"""
DiT policy for joints_50demos: RGB images + free-form text prompt
-> joint angles (+ gripper).

Same as dp_joints_image but uses a condition that combines two RGB images with a
text embedding from a frozen T5 encoder. No gym env; training only (or inference
with external env).
"""
import hydra
import os
import sys
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

# Run from repo root so cleandiffuser and utils are found (same as dp_joints_image)
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
from cleandiffuser.nn_condition import BaseNNCondition

from transformers import T5Tokenizer, T5EncoderModel

# -----------------------------------------------------------------------------
# Free-form text prompt (for DiT with text conditioning)
# Set TEXT_PROMPT to any instruction string you like. The frozen T5 encoder
# turns this into an embedding that is fused with the image-based condition.
# -----------------------------------------------------------------------------
TEXT_PROMPT = "push the block onto the green paper"

# T5 encoder config (can be overridden by Hydra config if desired)
T5_MODEL_NAME = "t5-small"
T5_MAX_LENGTH = 64


def build_t5_encoder(device: str):
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    encoder = T5EncoderModel.from_pretrained(T5_MODEL_NAME)
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return tokenizer, encoder


@torch.no_grad()
def encode_text(prompt: str, tokenizer: T5Tokenizer, encoder: T5EncoderModel, device: str) -> torch.Tensor:
    """
    Encode a single prompt string into a pooled text embedding using T5.

    Returns:
        text_emb: (1, hidden_dim)
    """
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=T5_MAX_LENGTH,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    # Masked mean pooling over tokens
    mask = attention_mask.unsqueeze(-1)  # (1, seq_len, 1)
    masked_hidden = last_hidden * mask
    sum_hidden = masked_hidden.sum(dim=1)  # (1, hidden_dim)
    lengths = mask.sum(dim=1).clamp(min=1)  # (1, 1)
    pooled = sum_hidden / lengths
    return pooled


class MultiImageAndTextCondition(BaseNNCondition):
    """
    Condition that combines image observations (MultiImageObsCondition) with a
    text embedding. Expects condition dict to contain image keys
    (e.g. image_cam0, image_cam1).
    Output shape matches the image-only embedding dim so it plugs into DiT.
    """

    def __init__(
        self,
        image_condition: nn.Module,
        image_emb_dim: int,
        text_proj: nn.Module,
    ):
        super().__init__()
        self.image_condition = image_condition
        self.image_emb_dim = image_emb_dim
        self.text_proj = text_proj
        self._text_emb: torch.Tensor = None

    def set_text_embedding(self, text_emb: torch.Tensor):
        """
        Set the pooled text embedding from T5.

        Args:
            text_emb: (1, hidden_dim) or (B, hidden_dim)
        """
        if text_emb.dim() == 2 and text_emb.shape[0] != 1:
            # Average across batch if a batch is given
            text_emb = text_emb.mean(dim=0, keepdim=True)
        self._text_emb = text_emb

    def forward(self, condition: dict, mask: torch.Tensor = None):
        # Don't mutate caller's dict
        cond_copy = dict(condition)
        image_emb = self.image_condition(cond_copy, mask)  # (B, image_emb_dim)
        if self._text_emb is None:
            return image_emb

        B = image_emb.shape[0]
        text_emb = self._text_emb.to(image_emb.device)  # (1, hidden_dim)
        text_emb = self.text_proj(text_emb)  # (1, image_emb_dim)
        text_emb = text_emb.expand(B, -1)  # (B, image_emb_dim)
        return image_emb + text_emb


@hydra.main(config_path="../configs/dp/joints/dit", config_name="joints_image_dit_text")
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
        image_augmentations=getattr(args, "image_augmentations", None),
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

    # --------------- Create DiT + Image+Text Condition -----------------
    global T5_MODEL_NAME, T5_MAX_LENGTH
    t5_model_name = getattr(args, "t5_model_name", T5_MODEL_NAME)
    t5_max_length = getattr(args, "t5_max_length", T5_MAX_LENGTH)
    T5_MODEL_NAME = t5_model_name
    T5_MAX_LENGTH = t5_max_length

    image_emb_dim = 256 * args.obs_steps  # same as vanilla DiT condition size

    from cleandiffuser.nn_condition import MultiImageObsCondition
    from cleandiffuser.nn_diffusion import DiT1d

    image_condition = MultiImageObsCondition(
        shape_meta=args.shape_meta,
        emb_dim=256,
        rgb_model_name=args.rgb_model,
        resize_shape=args.resize_shape,
        crop_shape=args.crop_shape,
        random_crop=args.random_crop,
        use_group_norm=args.use_group_norm,
        use_seq=args.use_seq,
    ).to(args.device)

    # Build frozen T5 encoder and projection into image_emb_dim
    tokenizer, t5_encoder = build_t5_encoder(args.device)
    text_hidden_dim = t5_encoder.config.d_model
    text_proj = nn.Linear(text_hidden_dim, image_emb_dim).to(args.device)

    nn_condition = MultiImageAndTextCondition(
        image_condition=image_condition,
        image_emb_dim=image_emb_dim,
        text_proj=text_proj,
    ).to(args.device)

    nn_diffusion = DiT1d(
        args.action_dim,
        emb_dim=image_emb_dim,
        d_model=320,
        n_heads=10,
        depth=2,
        timestep_emb_type="fourier",
    ).to(args.device)

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

    # Precompute text embedding for the global TEXT_PROMPT
    text_emb = encode_text(TEXT_PROMPT, tokenizer, t5_encoder, device=args.device)
    nn_condition.set_text_embedding(text_emb)

    if args.mode == "train":
        n_gradient_step = 0
        diffusion_loss_list = []
        start_time = time.time()
        for batch in loop_dataloader(dataloader):
            nobs = batch["obs"]
            condition = {}
            for k in nobs.keys():
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
                stats_path = logger._model_dir / f"action_normalizer_stats_{n_gradient_step}.npz"
                np.savez(
                    stats_path,
                    min=dataset.normalizer["action"].min,
                    max=dataset.normalizer["action"].max,
                )
                print(f"Action normalizer stats saved to {stats_path}")

            if envs is not None and n_gradient_step > 0 and n_gradient_step % args.eval_freq == 0:
                print("Evaluate model...")
                agent.model.eval()
                agent.model_ema.eval()
                agent.model.train()
                agent.model_ema.train()

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                final_stats_path = logger._model_dir / "action_normalizer_stats_final.npz"
                np.savez(
                    final_stats_path,
                    min=dataset.normalizer["action"].min,
                    max=dataset.normalizer["action"].max,
                )
                print(f"Action normalizer stats (final) saved to {final_stats_path}")
                logger.finish(agent)
                break

        full_model_path = os.path.join(args.work_dir, "agent_full.pth")
        torch.save(agent, full_model_path)
        print(f"Full agent model saved to {full_model_path}")
        full_stats_path = os.path.join(args.work_dir, "action_normalizer_stats.npz")
        np.savez(
            full_stats_path,
            min=dataset.normalizer["action"].min,
            max=dataset.normalizer["action"].max,
        )
        print(f"Action normalizer stats saved to {full_stats_path}")

    elif args.mode == "inference":
        if args.model_path:
            agent.load(args.model_path)
        else:
            raise ValueError("Empty model for inference")

        # Re-encode text prompt for inference and update condition
        text_emb = encode_text(TEXT_PROMPT, tokenizer, t5_encoder, device=args.device)
        nn_condition.set_text_embedding(text_emb)

        agent.model.eval()
        agent.model_ema.eval()
        print("No env for joints-image; skipping rollout.")
        print(
            f"Inference uses TEXT_PROMPT={TEXT_PROMPT!r}."
        )
        # When sampling, build condition with same image keys as in training:
        # condition = { "image_cam0": ..., "image_cam1": ... }
        # -------------------------------------------------------------------------
        # Action unnormalization at inference (same as dp_joints_image)
        # -------------------------------------------------------------------------
        # 1) Load action_normalizer_stats.npz
        # 2) naction, _ = agent.sample(..., condition_cfg=condition, ...)
        # 3) action_raw = (naction + 1.0) / 2.0 * action_range + action_min
        # 4) Use first action_steps for the robot.
        # -------------------------------------------------------------------------
    else:
        raise ValueError("Illegal mode")


if __name__ == "__main__":
    pipeline()
