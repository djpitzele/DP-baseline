"""
Diffusion Policy (DP) for joints_50demos: RGB images -> joint angles (+ gripper).
No gym env; training only (or inference with external env).
"""
import hydra
import os
import sys
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

# Run from repo root so cleandiffuser and utils are found (same as dp_robomimic)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn

from utils import set_seed, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from cleandiffuser.dataset.joints_dataset import JointsImageDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters


@hydra.main(config_path="../configs/dp/joints/dit", config_name="joints_image")
def pipeline(args):
    set_seed(args.seed)
    logger = Logger(pathlib.Path(args.work_dir), args)

    # No gym env for joints-image (no sim); skip inference or use external env
    envs = None

    # ---------------- Create Dataset ----------------
    dataset_path = os.path.expanduser(args.dataset_path)
    dataset = JointsImageDataset(
        dataset_path,
        horizon=args.horizon,
        obs_keys=args.obs_keys,
        pad_before=args.obs_steps - 1,
        pad_after=args.action_steps - 1,
        abs_action=args.abs_action,
    )
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # --------------- Create Diffusion Model -----------------
    if args.nn == "dit":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import DiT1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = DiT1d(
            args.action_dim,
            emb_dim=256 * args.obs_steps,
            d_model=320,
            n_heads=10,
            depth=2,
            timestep_emb_type="fourier",
        ).to(args.device)

    elif args.nn == "chi_unet":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiUNet1d

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
        ).to(args.device)
        nn_diffusion = ChiUNet1d(
            args.action_dim,
            256,
            args.obs_steps,
            model_dim=256,
            emb_dim=256,
            dim_mult=[1, 2, 2],
            obs_as_global_cond=True,
            timestep_emb_type="positional",
        ).to(args.device)

    elif args.nn == "chi_transformer":
        from cleandiffuser.nn_condition import MultiImageObsCondition
        from cleandiffuser.nn_diffusion import ChiTransformer

        nn_condition = MultiImageObsCondition(
            shape_meta=args.shape_meta,
            emb_dim=256,
            rgb_model_name=args.rgb_model,
            resize_shape=args.resize_shape,
            crop_shape=args.crop_shape,
            random_crop=args.random_crop,
            use_group_norm=args.use_group_norm,
            use_seq=args.use_seq,
            keep_horizon_dims=True,
        ).to(args.device)
        nn_diffusion = ChiTransformer(
            args.action_dim,
            256,
            args.horizon,
            args.obs_steps,
            d_model=256,
            nhead=4,
            num_layers=4,
            timestep_emb_type="positional",
        ).to(args.device)
    else:
        raise ValueError(f"Invalid nn type {args.nn}")

    print("======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print("==============================================================================")

    if args.diffusion == "ddpm":
        from cleandiffuser.diffusion.ddpm import DDPM

        x_max = torch.ones((1, args.horizon, args.action_dim), device=args.device) * +1.0
        x_min = torch.ones((1, args.horizon, args.action_dim), device=args.device) * -1.0
        agent = DDPM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            diffusion_steps=args.sample_steps,
            x_max=x_max,
            x_min=x_min,
            optim_params={"lr": args.lr},
        )
    elif args.diffusion == "edm":
        from cleandiffuser.diffusion.edm import EDM

        agent = EDM(
            nn_diffusion=nn_diffusion,
            nn_condition=nn_condition,
            device=args.device,
            optim_params={"lr": args.lr},
        )
    else:
        raise NotImplementedError
    lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=args.gradient_steps)

    if args.mode == "train":
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            nobs = batch["obs"]
            condition = {}
            for k in nobs.keys():
                # image: (B, T, C, H, W) -> keep first obs_steps
                condition[k] = nobs[k][:, : args.obs_steps, :].to(args.device)

            naction = batch["action"].to(args.device)

            diffusion_loss = agent.update(naction, condition)["loss"]
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)

            if n_gradient_step % args.log_freq == 0:
                metrics = {
                    "step": n_gradient_step,
                    "total_time": time.time() - start_time,
                    "avg_diffusion_loss": np.mean(diffusion_loss_list),
                }
                logger.log(metrics, category="train")
                diffusion_loss_list = []

            if n_gradient_step % args.save_freq == 0:
                logger.save_agent(agent=agent, identifier=n_gradient_step)

            if envs is not None and n_gradient_step > 0 and n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                # inference(args, envs, dataset, agent, logger)  # no env for joints
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                logger.finish(agent)
                break

    elif args.mode == "inference":
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")
        agent.model.eval()
        agent.model_ema.eval()
        if envs is not None:
            # inference(args, envs, dataset, agent, logger)
            pass
        print("No env for joints-image; skipping rollout.")
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
