"""
Convert joints_50demos folder (per-run rgb + joint_trajectory) into a single zarr
replay buffer compatible with CleanDiffuser DP image pipelines.

Usage (from repo root; activate conda env first):
  conda activate cleandiffuser
  python scripts/joints_50demos_to_zarr.py --data_dir joints_50demos --out_path joints_50demos_replay.zarr
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow importing cleandiffuser when run from repo root without pip install -e
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image

from cleandiffuser.dataset.replay_buffer import ReplayBuffer

try:
    import zarr
    ZARR_V3 = hasattr(zarr, "storage")
except Exception:
    ZARR_V3 = False


def _save_buffer_to_zarr(buffer: ReplayBuffer, zarr_path: str) -> None:
    """Write buffer to zarr on disk (compatible with zarr v2 and v3)."""
    path = os.path.expanduser(zarr_path)
    if ZARR_V3:
        store = zarr.storage.LocalStore(path)
        root = zarr.open_group(store, mode="w")
        meta = root.create_group("meta")
        data_grp = root.create_group("data")
        ep = buffer.episode_ends[:]
        meta.create_array("episode_ends", shape=ep.shape, dtype=ep.dtype)[...] = ep
        for key in buffer.keys():
            arr = buffer[key][:]
            chunk_len = min(100, max(1, arr.shape[0] // 10))
            chunks = (chunk_len,) + arr.shape[1:]
            z = data_grp.create_array(key, shape=arr.shape, dtype=arr.dtype, chunks=chunks)
            z[...] = arr
    else:
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store)
        meta = root.require_group("meta")
        meta.array("episode_ends", data=buffer.episode_ends[:], overwrite=True)
        data_grp = root.require_group("data")
        for key in buffer.keys():
            arr = buffer[key][:]
            chunk_len = min(100, max(1, arr.shape[0] // 10))
            chunks = (chunk_len,) + arr.shape[1:]
            data_grp.array(key, data=arr, chunks=chunks, overwrite=True)


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one run: rgb images and actions (joint_angles + gripper). Returns (img, action) or None if invalid."""
    rgb_dir = run_dir / "rgb"
    joint_dir = run_dir / "joint_trajectory"
    if not rgb_dir.is_dir() or not joint_dir.is_dir():
        return None

    # Single npz per run
    npz_files = list(joint_dir.glob("*.npz"))
    if not npz_files:
        return None
    data = np.load(npz_files[0])

    joint_angles = data["joint_angles"]  # (T, 7)
    gripper = data["gripper_widths"] if "gripper_widths" in data else data["gripper_states"].astype(np.float64)
    if gripper.ndim == 1:
        gripper = gripper[:, None]  # (T, 1)
    action = np.concatenate([joint_angles, gripper], axis=-1).astype(np.float32)  # (T, 8)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    n = min(len(rgb_files), action.shape[0])
    if n == 0:
        return None
    rgb_files = rgb_files[:n]
    action = action[:n]

    imgs = []
    for f in rgb_files:
        im = np.array(Image.open(f))
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        imgs.append(im)
    img = np.stack(imgs, axis=0)  # (T, H, W, 3) uint8

    return img, action


def main():
    parser = argparse.ArgumentParser(description="Convert joints_50demos to zarr replay buffer.")
    parser.add_argument("--data_dir", type=str, default="joints_50demos", help="Path to joints_50demos folder")
    parser.add_argument("--out_path", type=str, default="joints_50demos_replay.zarr", help="Output zarr path")
    parser.add_argument("--max_runs", type=int, default=None, help="Max number of run_* dirs to process (default: all)")
    parser.add_argument("--resize", type=int, nargs=2, default=(96, 96), help="(H, W) to resize images; default 96 96 for DP")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
        print(f"Processing first {len(run_dirs)} runs (--max_runs={args.max_runs})")
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories in {data_dir}")

    # Use in-memory numpy buffer to avoid zarr API differences (v2 vs v3); save to zarr at the end
    buffer = ReplayBuffer.create_empty_numpy()
    for run_dir in run_dirs:
        out = load_run(run_dir)
        if out is None:
            print(f"Skip (invalid or empty): {run_dir.name}")
            continue
        img, action = out
        if args.resize:
            from PIL import Image as PILImage
            h, w = args.resize
            resized = np.stack([
                np.array(PILImage.fromarray(img[i]).resize((w, h), PILImage.BILINEAR))
                for i in range(len(img))
            ], axis=0)
            img = resized
        buffer.add_episode({"img": img, "action": action})

    out_path = os.path.expanduser(args.out_path)
    _save_buffer_to_zarr(buffer, out_path)
    print(f"Saved zarr to {out_path}  steps={buffer.n_steps}  episodes={buffer.n_episodes}")


if __name__ == "__main__":
    main()
