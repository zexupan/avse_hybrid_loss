import os
import numpy as np 
import argparse
import csv
import tqdm
import scipy.io.wavfile as wavfile

np.random.seed(0)

def extract_wav_from_mp4(line):
	# Extract .wav file from mp4
	video_from_path=args.data_direc + line[0]+'/'+line[1]+'/'+line[2]+'.mp4'
	audio_save_direc=args.audio_data_direc + line[0]+'/'+line[1]+'/'
	if not os.path.exists(audio_save_direc):
		os.makedirs(audio_save_direc)
	audio_save_path=audio_save_direc+line[2]+'.wav'
	if not os.path.exists(audio_save_path):
		os.system("ffmpeg -i %s %s"%(video_from_path, audio_save_path))

	sr, audio = wavfile.read(audio_save_path)
	assert sr==args.sampling_rate , "sampling_rate mismatch"
	sample_length = audio.shape[0]/args.sampling_rate
	return sample_length # In seconds


def main(args):
	# read the datalist and separate into train, val and test set
	train_list=[]
	val_list=[]
	test_list=[]

	test_data_list = open(args.test_list).read().splitlines()
	for line in tqdm.tqdm(test_data_list, desc='Processing Test List'):
		line = line.split(' ')
		line = line[0].split('/')
		ln = ('main',line[0],line[1])
		sample_length = extract_wav_from_mp4(ln)
		test_list.append(('main',line[0],line[1], sample_length))

	val_data_list = open(args.val_list).read().splitlines()
	for line in tqdm.tqdm(val_data_list, desc='Processing Validation List'):
		line = line.split('/')
		ln = ('main',line[0],line[1])
		sample_length = extract_wav_from_mp4(ln)
		val_list.append(('main',line[0],line[1], sample_length))

	train_data_list = open(args.train_list).read().splitlines()
	for line in tqdm.tqdm(train_data_list, desc='Processing Train List'):
		line = line.split('/')
		ln = ('main',line[0],line[1])
		sample_length = extract_wav_from_mp4(ln)
		ln=('main',line[0],line[1], sample_length)
		train_list.append(ln)

	# Create mixture list
	f=open(args.mixture_data_list,'w')
	w=csv.writer(f)
	create_mixture_list(args, 'test', args.test_samples, test_list, w)
	create_mixture_list(args, 'val', args.val_samples, val_list, w)
	create_mixture_list(args, 'train', args.train_samples, train_list, w)
	f.close()


def create_mixture_list(args, data, length, data_list, w):
	# data_list = sorted(data_list, key=lambda data: data[3], reverse=True)
	for _ in range(length):
		mixtures=[data]
		cache = []

		# target speaker
		idx = np.random.randint(0, len(data_list))
		cache.append(idx)
		mixtures = mixtures + list(data_list[idx])
		shortest = mixtures[-1]
		del mixtures[-1]
		mixtures.append(0)

		while len(cache) < args.C:
			idx = np.random.randint(0, len(data_list))
			if idx in cache:
				continue
			cache.append(idx)
			mixtures = mixtures + list(data_list[idx])
			del mixtures[-1]
			db_ratio = np.random.uniform(-args.mix_db,args.mix_db)
			mixtures.append(db_ratio)
		mixtures.append(shortest)
		w.writerow(mixtures)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LRS2 dataset')
	parser.add_argument('--data_direc', type=str)
	parser.add_argument('--pretrain_list', type=str)
	parser.add_argument('--train_list', type=str)
	parser.add_argument('--val_list', type=str)
	parser.add_argument('--test_list', type=str)
	parser.add_argument('--C', type=int)
	parser.add_argument('--mix_db', type=float)
	parser.add_argument('--train_samples', type=int)
	parser.add_argument('--val_samples', type=int)
	parser.add_argument('--test_samples', type=int)
	parser.add_argument('--audio_data_direc', type=str)
	parser.add_argument('--sampling_rate', type=int)
	parser.add_argument('--mixture_data_list', type=str)
	args = parser.parse_args()
	
	main(args)