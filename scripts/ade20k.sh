#!/bin/bash

# Training
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
	     --class_cond True --no_instance True

# Classifier-free Finetune
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-FINETUNE'
mpiexec -n 8 python image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 2e-5 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
	     --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 \
	     --class_cond True --no_instance True --drop_rate 0.2 --resume_checkpoint OUTPUT/ADE20K-SDM-256CH/model.pt

# Testing
export OPENAI_LOGDIR='OUTPUT/ADE20K-SDM-256CH-TEST'
mpiexec -n 8 python image_sample.py --data_dir ./data/ade20k --dataset_mode ade20k --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
       --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --num_classes 151 \
       --class_cond True --no_instance True --batch_size 2 --num_samples 2000 --model_path OUTPUT/ADE20K-SDM-256CH-FINETUNE/ema_0.9999_best.pt --results_path RESULTS/ADE20K-SDM-256CH --s 1.5




python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /bigtemp/as3ek/datasets/ade20k/ADEChallengeData2016/ --log_path /u/as3ek/github/semantic-diffusion-model/outputs/pannuke_1/ --dataset_mode ade20k --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 151 --class_cond True --no_instance True

# CS
python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /localtmp/as3ek/datasets/pannuke/ --log_path /u/as3ek/github/semantic-diffusion-model/outputs/pannuke_1/ --dataset_mode pannuke --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False

python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /bigtemp/as3ek/datasets/pannuke/ --log_path /u/as3ek/github/semantic-diffusion-model/outputs/pannuke_2 --dataset_mode pannuke --lr 1e-4 --batch_size 4 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_anneal_steps 50000


python -m torch.distributed.run --nproc_per_node=8 image_train.py --data_dir /bigtemp/as3ek/datasets/pannuke/ --log_path /bigtemp/as3ek/p/histofusion/outputs/pannuke_20k_1e4_50k_cos/ --dataset_mode pannuke --lr 1e-4 --batch_size 2 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 20000 --lr_total_steps 50000


## Rivanna
# Train
python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /scratch/as3ek/datasets/pannuke/ --log_path /scratch/as3ek/github/histofusion/outputs/pannuke_50k_1e4nd/ --dataset_mode pannuke --lr 1e-4 --batch_size 40 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 200 --lr_total_steps 50000 --log_interval 100 --save_interval 2000 --lr_decay none
# Sample
python -m torch.distributed.run --nproc_per_node=4 image_sample.py --data_dir /scratch/as3ek/datasets/pannuke/ --dataset_mode pannuke --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --model_path /scratch/as3ek/github/histofusion/outputs/pannuke_10k_2e4_bs32/ema_0.9999_010000.pt --results_path /scratch/as3ek/github/histofusion/outputs/pannuke_10k_2e4_bs32/results --batch_size 32 --num_samples 100
# Resume
python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /scratch/as3ek/datasets/pannuke/ --log_path /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_1e4nd/ --dataset_mode pannuke --lr 1e-4 --batch_size 40 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 200 --lr_total_steps 50000 --log_interval 100 --save_interval 2000 --lr_decay none --resume_checkpoint /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_1e4nd/model010000.pt




python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /scratch/as3ek/datasets/pannuke/ --log_path /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_2e4cos/ --dataset_mode pannuke --lr 2e-4 --batch_size 40 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 10500 --lr_total_steps 50000 --log_interval 100 --save_interval 2000 --lr_decay cosine --resume_checkpoint /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_1e4nd/model010000.pt