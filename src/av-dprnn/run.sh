#!/bin/sh

gpu_id=11,12

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avDprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=2191 \
main.py \
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/LRS2/audio_clean/' \
--visual_direc '/home/panzexu/datasets/LRS2/visual_embedding/lip/' \
--mix_lst_path '/home/panzexu/datasets/LRS2/audio_mixture/2_mix_min_asr/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/LRS2/audio_mixture/2_mix_min_asr/' \
--C 3 \
\
--batch_size 24 \
--num_workers 2 \
\
--epochs 100 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \

