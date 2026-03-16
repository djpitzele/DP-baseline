"""
Convert a raw push dataset (video format) into a ReplayBuffer zarr.

Expected input directory structure (per episode):

    data/0309_push_150demos/
        run_YYYYMMDD_HHMMSS/
            cam0.mp4
            cam1.mp4
            trajectory_joints.npz

This script creates a zarr directory that is compatible with
cleandiffuser.dataset.joints_dataset.JointsImageDataset for the dual-camera
setting, with the following zarr keys:

    /data/img_cam0  -> (N, H, W, 3) uint8
    /data/img_cam1  -> (N, H, W, 3) uint8
    /data/action    -> (N, action_dim) float32
    /meta/episode_ends -> (E,) int64 cumulative step counts

Requires imageio with the FFMPEG backend:
    pip install imageio[ffmpeg]

Run from the repository root (DP-baseline), for example:

    python scripts/convert_push_to_zarr.py \\
        --input_dir data/0309_push_150demos \\
        --output_path data/joints_150demos_replay.zarr

You can then point configs like configs/dp/joints/dit/joints_image.yaml
to the produced zarr via dataset_path.
"""

import argparse
import os
import sys
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image

import imageio.v3 as iio
import zarr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cleandiffuser.dataset.replay_buffer import ReplayBuffer

try:
    # Match zarr version handling used in joints_50demos_to_zarr.py
    ZARR_V3 = hasattr(zarr, "storage")
except Exception:
    ZARR_V3 = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert push_100demos-style runs into a ReplayBuffer zarr "
        "for JointsImageDataset (dual RGB cameras)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/0309_push_150demos",
        help="Path to folder containing run_* episode directories.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/joints_150demos_replay.zarr",
        help="Path to output zarr directory (will be created).",
    )
    parser.add_argument(
        "--resize_hw",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="Resize images to (H, W) before writing to zarr. "
        "Defaults to 256 256 to match joints_image shape_meta.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to convert.",
    )
    parser.add_argument(
        "--allow_truncate",
        action="store_true",
        help=(
            "If set, when action and image sequence lengths differ, truncate all "
            "to the minimum length instead of raising an error."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "If set, remove any existing zarr directory at output_path before "
            "creating a new one."
        ),
    )
    return parser.parse_args()


def enumerate_episodes(input_dir: str) -> List[str]:
    episodes = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            episodes.append(full)
    episodes.sort()
    return episodes


def load_action_array(run_dir: str) -> np.ndarray:
    npz_path = os.path.join(run_dir, "trajectory_joints.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing trajectory file: {npz_path}")
    data = np.load(npz_path)

    # Expected keys:
    # ['timestamps', 'joint_angles', 'ee_positions', 'ee_orientations',
    #  'gripper_widths', 'gripper_states', 'joint_names', 'capture_interval']
    if "joint_angles" not in data:
        raise KeyError(f"'joint_angles' not found in {npz_path}")

    joint_angles = np.array(data["joint_angles"])  # (T, 7)
    if joint_angles.ndim != 2:
        raise ValueError(
            f"Expected 'joint_angles' to be 2D (T, D) in {npz_path}, "
            f"got shape {joint_angles.shape}"
        )

    if "gripper_widths" in data:
        gripper = np.array(data["gripper_widths"])
    elif "gripper_states" in data:
        gripper = np.array(data["gripper_states"]).astype(np.float64)
    else:
        raise KeyError(
            f"Neither 'gripper_widths' nor 'gripper_states' found in {npz_path}"
        )

    if gripper.ndim == 1:
        gripper = gripper[:, None]
    elif gripper.ndim != 2:
        raise ValueError(
            f"Expected gripper array to be 1D or 2D in {npz_path}, "
            f"got shape {gripper.shape}"
        )

    if gripper.shape[0] != joint_angles.shape[0]:
        raise ValueError(
            f"Length mismatch between 'joint_angles' ({joint_angles.shape[0]}) "
            f"and gripper array ({gripper.shape[0]}) in {npz_path}"
        )

    action = np.concatenate([joint_angles, gripper], axis=-1).astype(np.float32)
    if action.shape[0] <= 0:
        raise ValueError(f"Empty trajectory in {npz_path} (shape {action.shape})")
    return action


def subsample_to_length(arr: np.ndarray, T: int) -> np.ndarray:
    """Return arr subsampled to exactly T steps along axis 0.

    Indices are chosen with np.linspace so that dropped entries are spread
    evenly across the full sequence rather than all removed from the end.
    If arr already has T steps it is returned unchanged.
    """
    if arr.shape[0] == T:
        return arr
    indices = np.round(np.linspace(0, arr.shape[0] - 1, T)).astype(int)
    return arr[indices]


def load_video_frames(
    run_dir: str,
    cam_name: str,
    resize_hw: Tuple[int, int],
) -> np.ndarray:
    """Decode all frames from a .mp4 video file and return as (T, H, W, 3) uint8."""
    video_path = os.path.join(run_dir, f"{cam_name}.mp4")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Missing video file: {video_path}")

    # iio.imread with index=None reads all frames as (T, H, W, C)
    frames = iio.imread(video_path, plugin="FFMPEG", index=None)

    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"Unexpected video frame shape {frames.shape} from {video_path}"
        )

    H, W = resize_hw
    if (frames.shape[1], frames.shape[2]) != (H, W):
        T = frames.shape[0]
        resized = np.empty((T, H, W, 3), dtype=np.uint8)
        for i in range(T):
            im = Image.fromarray(frames[i]).resize((W, H), resample=Image.BILINEAR)
            resized[i] = np.array(im, dtype=np.uint8)
        frames = resized

    return frames.astype(np.uint8)


def _save_buffer_to_zarr(buffer: ReplayBuffer, zarr_path: str) -> None:
    """Write buffer to zarr on disk (mirror joints_50demos_to_zarr layout/chunking)."""
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


def main() -> None:
    args = parse_args()

    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    resize_hw = (int(args.resize_hw[0]), int(args.resize_hw[1]))

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if os.path.exists(output_path):
        if args.overwrite:
            print(f"[INFO] Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                f"Use --overwrite to replace it."
            )

    episodes = enumerate_episodes(input_dir)
    if not episodes:
        raise RuntimeError(f"No run_* episodes found under {input_dir}")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"[INFO] Found {len(episodes)} episodes under {input_dir}")

    # Use in-memory numpy buffer to match joints_50demos_to_zarr behavior;
    # we write to zarr on disk once at the end.
    replay_buffer = ReplayBuffer.create_empty_numpy()

    total_steps = 0
    for idx, run_dir in enumerate(episodes):
        run_name = os.path.basename(run_dir)
        print(f"[INFO] Processing episode {idx + 1}/{len(episodes)}: {run_name}")

        # Load actions.
        action = load_action_array(run_dir)
        T_action = action.shape[0]

        # Decode video frames.
        img_cam0 = load_video_frames(run_dir, "cam0", resize_hw=resize_hw)
        img_cam1 = load_video_frames(run_dir, "cam1", resize_hw=resize_hw)
        T_cam0 = img_cam0.shape[0]
        T_cam1 = img_cam1.shape[0]

        if args.allow_truncate:
            T = min(T_action, T_cam0, T_cam1)
            if T <= 0:
                raise RuntimeError(f"Episode {run_name} has no usable steps.")
            if T < T_action or T < T_cam0 or T < T_cam1:
                print(
                    f"[WARN] Length mismatch in {run_name}: "
                    f"action={T_action}, cam0={T_cam0}, cam1={T_cam1}. "
                    f"Subsampling all to {T} steps (evenly spaced)."
                )
            action = subsample_to_length(action, T)
            img_cam0 = subsample_to_length(img_cam0, T)
            img_cam1 = subsample_to_length(img_cam1, T)
        else:
            if not (T_action == T_cam0 == T_cam1):
                raise RuntimeError(
                    f"Length mismatch in {run_name}: "
                    f"action={T_action}, cam0={T_cam0}, cam1={T_cam1}. "
                    f"Use --allow_truncate to truncate to the minimum length."
                )
            T = T_action

        episode_data = {
            "img_cam0": img_cam0,
            "img_cam1": img_cam1,
            "action": action,
        }
        replay_buffer.add_episode(episode_data)

        total_steps += T
        print(
            f"[INFO] Added episode {run_name} with {T} steps. "
            f"Cumulative steps: {total_steps}."
        )

    # Save to zarr using the same layout and chunking as joints_50demos_to_zarr.py
    _save_buffer_to_zarr(replay_buffer, output_path)

    # Basic validation of resulting zarr without going through ReplayBuffer
    # (zarr v3 groups do not always implement the same mapping API expected by ReplayBuffer).
    print(f"[INFO] Finished conversion. Validating zarr at {output_path} ...")
    group = zarr.open(os.path.expanduser(output_path), mode="r")
    if "data" not in group or "meta" not in group:
        raise RuntimeError(f"Zarr at {output_path} is missing 'data' or 'meta' groups.")

    data_grp = group["data"]
    # Prefer array_keys if available (zarr v2), otherwise fall back to keys().
    if hasattr(data_grp, "array_keys"):
        keys = list(data_grp.array_keys())
    else:
        keys = list(data_grp.keys())
    print(f"[INFO] Zarr keys: {keys}")

    episode_ends = np.array(group["meta"]["episode_ends"])
    if episode_ends.size == 0:
        n_steps = 0
        n_episodes = 0
    else:
        n_steps = int(episode_ends[-1])
        n_episodes = int(episode_ends.shape[0])
    print(f"[INFO] Total steps (n_steps): {n_steps}")
    print(f"[INFO] Total episodes (n_episodes): {n_episodes}")

    if "img_cam0" not in keys or "img_cam1" not in keys or "action" not in keys:
        raise RuntimeError(
            "Zarr is missing required keys; expected 'img_cam0', 'img_cam1', and 'action'."
        )


if __name__ == "__main__":
    main()

