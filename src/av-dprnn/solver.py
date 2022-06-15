import time
from utils import *
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
from stft_loss import MultiResolutionSTFTLoss

class Solver(object):
    def __init__(self, train_data, validation_data, test_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.amp = amp
        self.stft_loss = MultiResolutionSTFTLoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                        opt_level=args.opt_level,
                                                        patch_torch_functions=args.patch_torch_functions)

        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict.pt' % self.args.continue_from, map_location='cpu')

            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.prev_val_loss = checkpoint['prev_val_loss']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            self.prev_val_loss = float("inf")
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            start = time.time()
            tr_loss, tr_snr_loss, tr_stft_loss = self._run_one_epoch(data_loader = self.train_data)
            reduced_tr_loss = self._reduce_tensor(tr_loss)
            reduced_tr_snr_loss = self._reduce_tensor(tr_snr_loss)
            reduced_tr_stft_loss = self._reduce_tensor(tr_stft_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, reduced_tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_snr_loss, val_stft_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                reduced_val_snr_loss = self._reduce_tensor(val_snr_loss)
                reduced_val_stft_loss = self._reduce_tensor(val_stft_loss)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_val_loss))

            # test
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                test_loss, test_snr_loss, test_stft_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                reduced_test_loss = self._reduce_tensor(test_loss)
                reduced_test_snr_loss = self._reduce_tensor(test_snr_loss)
                reduced_test_stft_loss = self._reduce_tensor(test_stft_loss)

            if self.print: print('Test Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Test Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_test_loss))


            # Check whether to adjust learning rate and early stop
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0

            if self.val_no_impv == 6: #(epoch %2) == 0:
                self.halving = True

            # Halfing the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * 0.5
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = reduced_val_loss

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)
                    self.writer.add_scalar('Test_loss', reduced_test_loss, epoch)
                    self.writer.add_scalar('Train_snr_loss', reduced_tr_snr_loss, epoch)
                    self.writer.add_scalar('Validation_snr_loss', reduced_val_snr_loss, epoch)
                    self.writer.add_scalar('Test_snr_loss', reduced_test_snr_loss, epoch)
                    self.writer.add_scalar('Train_stft_loss', reduced_tr_stft_loss, epoch)
                    self.writer.add_scalar('Validation_stft_loss', reduced_val_stft_loss, epoch)
                    self.writer.add_scalar('Test_stft_loss', reduced_test_stft_loss, epoch)

                # Save model
                if reduced_val_loss < self.best_val_loss:
                    self.best_val_loss = reduced_val_loss
                    checkpoint = {'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'amp': self.amp.state_dict(),
                                    'epoch': epoch+1,
                                    'prev_val_loss': self.prev_val_loss,
                                    'best_val_loss': self.best_val_loss,
                                    'val_no_impv': self.val_no_impv}
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict.pt")
                    print("Fund new best model, dict saved")

    def _run_one_epoch(self, data_loader, state='train'):
        total_loss = 0
        total_snr_loss = 0
        total_stft_loss = 0
        for i, (a_mix, a_tgt, v_tgt) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()

            a_tgt_est = self.model(a_mix, v_tgt)

            pos_snr = cal_SISNR(a_tgt, a_tgt_est)
            snr_loss = 0 - torch.mean(pos_snr)

            # print(loss.item())

            stft_loss_sc, stft_loss_mag = self.stft_loss(a_tgt_est, a_tgt)
            stft_loss = stft_loss_mag + stft_loss_sc

            gamma = 1
            loss = snr_loss + gamma* stft_loss
  
            if state=='train':
                # self._adjust_lr()
                self.optimizer.zero_grad()
                with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                               self.args.max_norm)
                self.optimizer.step()
            if state=='test':
                loss = 0 - torch.mean(pos_snr[::self.args.C])

            total_loss += loss.data
            total_snr_loss += snr_loss
            total_stft_loss += stft_loss
            
        return total_loss / (i+1), total_snr_loss / (i+1), total_stft_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
