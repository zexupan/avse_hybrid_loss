#!/bin/bash 

direc=/home/panzexu/datasets/LRS2/

data_direc=${direc}mvlrs_v1/
pretrain_list=${data_direc}pretrain_list.txt
train_list=${data_direc}train_list.txt
val_list=${data_direc}val_list.txt
test_list=${data_direc}test_list.txt

train_samples=200000 # no. of train mixture samples simulated
val_samples=5000 # no. of validation mixture samples simulated
test_samples=3000 # no. of test mixture samples simulated
C=2 # no. of speakers in the mixture
mix_db=10 # random db ratio from -10 to 10db
mixture_data_list=mixture_data_list_${C}mix.csv #mixture datalist
sampling_rate=16000 # audio sampling rate

audio_data_direc=${direc}audio_clean/ # Target audio saved directory
mixture_audio_direc=${direc}audio_mixture/${C}_mix_min_asr/ # Audio mixture saved directory

#stage 1: create mixture list
echo 'stage 1: create mixture list'
python 1_create_mixture_list.py \
--data_direc $data_direc \
--pretrain_list $pretrain_list \
--train_list $train_list \
--val_list $val_list \
--test_list $test_list \
--C $C \
--mix_db $mix_db \
--train_samples $train_samples \
--val_samples $val_samples \
--test_samples $test_samples \
--audio_data_direc $audio_data_direc \
--sampling_rate $sampling_rate \
--mixture_data_list $mixture_data_list \

# stage 2: create audio mixture from list
echo 'stage 2: create mixture audios'
python 2_create_mixture.py \
--C $C \
--audio_data_direc $audio_data_direc \
--mixture_audio_direc $mixture_audio_direc \
--mixture_data_list $mixture_data_list \

