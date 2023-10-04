## Rivanna
# Train
python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /scratch/as3ek/datasets/pannuke/ --log_path /scratch/as3ek/github/histofusion/outputs/pannuke_50k_1e4nd/ --dataset_mode pannuke --lr 1e-4 --batch_size 40 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 200 --lr_total_steps 50000 --log_interval 100 --save_interval 2000 --lr_decay none
# Sample
python -m torch.distributed.run --nproc_per_node=4 image_sample.py --data_dir /scratch/as3ek/datasets/pannuke/ --dataset_mode pannuke --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --model_path /scratch/as3ek/github/histofusion/outputs/pannuke_10k_2e4_bs32/ema_0.9999_010000.pt --results_path /scratch/as3ek/github/histofusion/outputs/pannuke_10k_2e4_bs32/results --batch_size 32 --num_samples 100
# Resume
python -m torch.distributed.run --nproc_per_node=4 image_train.py --data_dir /scratch/as3ek/datasets/pannuke/ --log_path /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_1e4nd/ --dataset_mode pannuke --lr 1e-4 --batch_size 40 --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False --lr_warmup_steps 200 --lr_total_steps 50000 --log_interval 100 --save_interval 2000 --lr_decay none --resume_checkpoint /scratch/as3ek/github/histofusion/outputs/pannuke_50k__10k_2e4cos__40k_1e4nd/model010000.pt

## UVA CS

# Sample
python -m torch.distributed.run --nproc_per_node=8 image_sample.py --data_dir /bigtemp/as3ek/datasets/pannuke/ \
        --dataset_mode pannuke --attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True \
        --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
        --use_scale_shift_norm True --use_checkpoint True --num_classes 6 --class_cond True --no_instance False \
        --model_path /bigtemp/as3ek/p/histofusion/outputs/pannuke_20k_1e4__40k_1e4cos/model050000.pt \
        --results_path /bigtemp/as3ek/p/histofusion/outputs/pannuke_20k_1e4__40k_1e4cos/results --batch_size 2 --num_samples 100