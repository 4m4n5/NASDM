"""
Train a diffusion model on images.
"""

import os
import argparse
import torch
import wandb

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.init_distributed_mode(args)        
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_util.get_rank()

    if dist_util.is_main_process():
        wandb.init(project="histofusion", config=vars(args))

    logger.configure(args.log_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(dist_util.dev())
    # model.to(device)
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) 
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        is_train=args.is_train,
        rank=dist_util.get_rank(),
        world_size=dist_util.get_world_size(),
    )
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_total_steps=args.lr_total_steps,
        lr_decay=args.lr_decay,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        lr=1e-4,
        lr_warmup_steps=500,
        lr_total_steps=50000,
        lr_decay="none", # ["cosine", "none"]
        weight_decay=0.0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        device='cuda',
        seed=42,
        world_size=8,    
        dist_url='env://',
        distributed=True,
        log_path='/scratch/as3ek/github/histofusion/outputs'
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
