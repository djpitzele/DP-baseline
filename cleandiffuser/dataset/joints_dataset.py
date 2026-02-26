"""
Dataset for joints_50demos-style data: RGB images as observation, joint angles (+ gripper) as action.
Zarr must have keys: img (N, H, W, 3), action (N, action_dim) for single-cam;
  or img_cam0 (N, H, W, 3), img_cam1 (N, H, W, 3), action (N, action_dim) for dual-cam (scene + wrist).
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import (
    SequenceSampler,
    MinMaxNormalizer,
    ImageNormalizer,
    dict_apply,
)


def _image_zarr_to_obs_key(zarr_key: str) -> str:
    """Map zarr image key to obs key used in shape_meta / condition."""
    if zarr_key == "img":
        return "image"
    if zarr_key.startswith("img_"):
        return "image_" + zarr_key[4:]
    return zarr_key


class JointsImageDataset(BaseDataset):
    """RGB images -> joint actions. No low-dim state (e.g. no agent_pos).
    Supports single-cam (img) or dual-cam (img_cam0, img_cam1) from rgb_cam0 / rgb_cam1 folders.
    """

    def __init__(
        self,
        zarr_path,
        obs_keys=("img", "action"),
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
    ):
        super().__init__()
        self.obs_keys = list(obs_keys)
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=list(obs_keys))
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._image_zarr_keys = [
            k for k in self.obs_keys
            if k == "img" or (k.startswith("img_") and k != "action")
        ]
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        obs_normalizers = {}
        for zarr_key in self._image_zarr_keys:
            obs_key = _image_zarr_to_obs_key(zarr_key)
            obs_normalizers[obs_key] = ImageNormalizer()
        return {
            "obs": obs_normalizers,
            "action": action_normalizer,
        }

    def __str__(self) -> str:
        return (
            f"Keys: {list(self.replay_buffer.keys())} "
            f"Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
        )

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # Each image: (T, H, W, C) -> (T, C, H, W), [0,255] -> [0,1], then normalize
        obs = {}
        for zarr_key in self._image_zarr_keys:
            obs_key = _image_zarr_to_obs_key(zarr_key)
            image = np.moveaxis(sample[zarr_key], -1, 1).astype(np.float32) / 255.0
            image = self.normalizer["obs"][obs_key].normalize(image)
            obs[obs_key] = image
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": obs,
            "action": action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.tensor)
