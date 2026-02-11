"""
Dataset for joints_50demos-style data: RGB images as observation, joint angles (+ gripper) as action.
Zarr must have keys: img (N, H, W, 3), action (N, action_dim).
"""
from typing import Dict

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


class JointsImageDataset(BaseDataset):
    """RGB images -> joint actions. No low-dim state (e.g. no agent_pos)."""

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
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        image_normalizer = ImageNormalizer()
        action_normalizer = MinMaxNormalizer(self.replay_buffer["action"][:])
        return {
            "obs": {
                "image": image_normalizer,
            },
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
        # image: (T, H, W, C) -> (T, C, H, W), [0,255] -> [0,1], then normalize
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0
        image = self.normalizer["obs"]["image"].normalize(image)
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": {
                "image": image,
            },
            "action": action,
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.tensor)
