"""
Convert the HuggingFace LeRobot-format LIBERO dataset (parquet + videos)
into a ReplayBuffer zarr for the diffusion policy training pipeline.

Supports two local data layouts (auto-detected):

  1. **Video-based** (standard LeRobot clone):
       <dataset_dir>/
         meta/info.json, meta/tasks.parquet
         data/chunk-000/file-*.parquet   (metadata only)
         videos/observation.images.image/chunk-000/file-*.mp4
         videos/observation.images.image2/chunk-000/file-*.mp4

  2. **Inline-image** (HuggingFace auto-converted parquet):
       <dataset_dir>/
         meta/tasks.parquet
         data/chunk-000/file-*.parquet   (images embedded as bytes)

Output zarr layout (compatible with JointsImageDataset):

    /data/img_cam0       -> (N, H, W, 3) uint8   wrist / eye-in-hand camera
    /data/img_cam1       -> (N, H, W, 3) uint8   environment / agentview camera
    /data/action         -> (N, 7)       float32
    /data/task_index     -> (N,)         int64
    /meta/episode_ends   -> (E,)         int64    cumulative step counts
    /meta/task_descriptions -> JSON string stored as zarr attribute

Camera mapping (from original LIBERO conventions):
    observation.images.image2  (eye-in-hand / wrist)    -> img_cam0
    observation.images.image   (agentview / environment) -> img_cam1

Memory-efficient: episodes are flushed to the on-disk zarr one at a time
so the full dataset never needs to fit in RAM.

Example usage (run from DP-baseline root):

    python scripts/convert_libero_to_zarr.py \
        --input_path ../LIBERO/libero/datasets \
        --output_path data/libero_replay.zarr \
        --resize_hw 96 96

Requires: pandas, pyarrow, imageio[ffmpeg], Pillow, zarr, numpy
"""

import argparse
import gc
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import zarr
from cleandiffuser.dataset.replay_buffer import ReplayBuffer

try:
    ZARR_V3 = hasattr(zarr.storage, "LocalStore")
except Exception:
    ZARR_V3 = False

WRIST_IMAGE_KEY = "observation.images.image2"
ENV_IMAGE_KEY = "observation.images.image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local LIBERO dataset (LeRobot parquet format) "
        "into a ReplayBuffer zarr for JointsImageDataset.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the local LIBERO dataset root directory "
        "(should contain meta/ and data/ subdirectories).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/libero_replay.zarr",
        help="Path to output zarr directory (will be created).",
    )
    parser.add_argument(
        "--resize_hw",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="Resize images to (H, W). Defaults to 256 256.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to convert.",
    )
    parser.add_argument(
        "--task_indices",
        type=int,
        nargs="+",
        default=None,
        help="If set, only convert episodes belonging to these task indices.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory before creating a new one.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_image_from_parquet(img_data) -> Image.Image:
    """Decode an image stored in a HuggingFace parquet image column."""
    import io

    if isinstance(img_data, dict):
        raw = img_data.get("bytes") or img_data.get("image")
        if raw is None:
            raise ValueError(f"Image dict has no 'bytes' key. Keys: {list(img_data.keys())}")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(img_data, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    raise TypeError(f"Cannot decode image from type {type(img_data)}")


def resize_image(img: Image.Image, hw: Tuple[int, int]) -> np.ndarray:
    """Resize a PIL image to (H, W) and return as uint8 numpy array."""
    h, w = hw
    if (img.height, img.width) != (h, w):
        img = img.resize((w, h), resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_video_frames(video_path: str, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Read all frames from an mp4 and return as (T, H, W, 3) uint8."""
    import imageio.v3 as iio

    frames = iio.imread(video_path, plugin="FFMPEG", index=None)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"Unexpected frame shape {frames.shape} from {video_path}")

    H, W = resize_hw
    if (frames.shape[1], frames.shape[2]) != (H, W):
        T = frames.shape[0]
        resized = np.empty((T, H, W, 3), dtype=np.uint8)
        for i in range(T):
            im = Image.fromarray(frames[i]).resize((W, H), resample=Image.BILINEAR)
            resized[i] = np.array(im, dtype=np.uint8)
        return resized
    return frames.astype(np.uint8)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(input_path: str) -> str:
    """Return 'video' if video files are present, else 'inline'."""
    video_dir = os.path.join(input_path, "videos")
    if os.path.isdir(video_dir):
        return "video"
    return "inline"


def find_parquet_files(input_path: str) -> List[str]:
    """Find and sort all data parquet files."""
    data_dir = os.path.join(input_path, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = []
    for root, _dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.endswith(".parquet"):
                files.append(os.path.join(root, fn))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir}")
    return files


def parquet_to_video_path(
    input_path: str, parquet_path: str, image_key: str
) -> str:
    """Map a data parquet path to its corresponding video file."""
    rel = os.path.relpath(parquet_path, os.path.join(input_path, "data"))
    video_rel = rel.replace(".parquet", ".mp4")
    return os.path.join(input_path, "videos", image_key, video_rel)


# ---------------------------------------------------------------------------
# Task descriptions
# ---------------------------------------------------------------------------

def load_task_descriptions(input_path: str) -> Dict[int, str]:
    """Load task_index -> description mapping from meta/tasks.parquet.

    In the LeRobot LIBERO format the task text is stored as the DataFrame
    *index* and the numeric task_index is a column.
    """
    tasks_path = os.path.join(input_path, "meta", "tasks.parquet")
    if not os.path.isfile(tasks_path):
        print(f"[WARN] tasks.parquet not found at {tasks_path}; "
              "task descriptions will be empty.")
        return {}
    tasks_df = pd.read_parquet(tasks_path)
    mapping = {}
    for description, row in tasks_df.iterrows():
        mapping[int(row["task_index"])] = str(description)
    return mapping


# ---------------------------------------------------------------------------
# Episode flushing helper
# ---------------------------------------------------------------------------

def flush_episode(
    replay_buffer: ReplayBuffer,
    ep_data: Dict[str, list],
) -> None:
    """Stack accumulated per-step lists into arrays and write one episode."""
    episode = {
        "img_cam0": np.stack(ep_data["img_cam0"]),
        "img_cam1": np.stack(ep_data["img_cam1"]),
        "action": np.stack(ep_data["action"]),
        "task_index": np.array(ep_data["task_index"], dtype=np.int64),
    }
    replay_buffer.add_episode(episode)


def _new_accum() -> Dict[str, list]:
    return {"img_cam0": [], "img_cam1": [], "action": [], "task_index": []}


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def process_video_format(
    input_path: str,
    parquet_files: List[str],
    resize_hw: Tuple[int, int],
    task_filter: Optional[set],
    max_episodes: Optional[int],
    replay_buffer: ReplayBuffer,
) -> Tuple[set, int]:
    """Process data when images are stored in video files.

    Episodes are flushed to the on-disk zarr one at a time so that the
    full dataset never needs to reside in RAM simultaneously.
    """
    seen_tasks: set = set()
    episodes_added = 0
    cur_ep_idx: Optional[int] = None
    cur_ep_data: Optional[Dict[str, list]] = None

    for file_idx, pf in enumerate(parquet_files):
        print(f"[INFO] Reading parquet file {file_idx + 1}/{len(parquet_files)}: "
              f"{os.path.basename(pf)}")

        df = pd.read_parquet(pf)

        wrist_video = parquet_to_video_path(input_path, pf, WRIST_IMAGE_KEY)
        env_video = parquet_to_video_path(input_path, pf, ENV_IMAGE_KEY)

        if not os.path.isfile(wrist_video):
            print(f"[WARN] Missing video {wrist_video}, skipping file.")
            continue
        if not os.path.isfile(env_video):
            print(f"[WARN] Missing video {env_video}, skipping file.")
            continue

        wrist_frames = load_video_frames(wrist_video, resize_hw)
        env_frames = load_video_frames(env_video, resize_hw)

        n_rows = len(df)
        if wrist_frames.shape[0] != n_rows or env_frames.shape[0] != n_rows:
            min_len = min(n_rows, wrist_frames.shape[0], env_frames.shape[0])
            print(f"[WARN] Length mismatch in {os.path.basename(pf)}: "
                  f"parquet={n_rows}, wrist={wrist_frames.shape[0]}, "
                  f"env={env_frames.shape[0]}. Truncating to {min_len}.")
            df = df.iloc[:min_len]
            wrist_frames = wrist_frames[:min_len]
            env_frames = env_frames[:min_len]

        for local_idx in range(len(df)):
            row = df.iloc[local_idx]
            ep_idx = int(row["episode_index"])
            t_idx = int(row["task_index"])

            if cur_ep_idx is not None and ep_idx != cur_ep_idx:
                if cur_ep_data and cur_ep_data["img_cam0"]:
                    seen_tasks.add(cur_ep_data["task_index"][0])
                    flush_episode(replay_buffer, cur_ep_data)
                    episodes_added += 1
                    if episodes_added == 1 or episodes_added % 50 == 0:
                        print(f"[INFO] Flushed {episodes_added} episodes to disk")
                cur_ep_data = None
                cur_ep_idx = None

            if task_filter is not None and t_idx not in task_filter:
                continue

            if max_episodes is not None and episodes_added >= max_episodes:
                break

            if cur_ep_data is None:
                cur_ep_data = _new_accum()
                cur_ep_idx = ep_idx

            cur_ep_data["img_cam0"].append(wrist_frames[local_idx])
            cur_ep_data["img_cam1"].append(env_frames[local_idx])
            cur_ep_data["action"].append(
                np.array(row["action"], dtype=np.float32)
            )
            cur_ep_data["task_index"].append(t_idx)

        del wrist_frames, env_frames, df
        gc.collect()

        if max_episodes is not None and episodes_added >= max_episodes:
            break

    if cur_ep_data and cur_ep_data["img_cam0"]:
        if max_episodes is None or episodes_added < max_episodes:
            seen_tasks.add(cur_ep_data["task_index"][0])
            flush_episode(replay_buffer, cur_ep_data)
            episodes_added += 1

    return seen_tasks, episodes_added


def process_inline_format(
    parquet_files: List[str],
    resize_hw: Tuple[int, int],
    task_filter: Optional[set],
    max_episodes: Optional[int],
    replay_buffer: ReplayBuffer,
) -> Tuple[set, int]:
    """Process data when images are embedded inline in the parquet files.

    Episodes are flushed to the on-disk zarr one at a time.
    """
    seen_tasks: set = set()
    episodes_added = 0
    cur_ep_idx: Optional[int] = None
    cur_ep_data: Optional[Dict[str, list]] = None

    for file_idx, pf in enumerate(parquet_files):
        print(f"[INFO] Reading parquet file {file_idx + 1}/{len(parquet_files)}: "
              f"{os.path.basename(pf)}")

        df = pd.read_parquet(pf)

        has_wrist = WRIST_IMAGE_KEY in df.columns
        has_env = ENV_IMAGE_KEY in df.columns
        if not has_wrist or not has_env:
            raise ValueError(
                f"Parquet file {pf} missing image columns. "
                f"Expected '{WRIST_IMAGE_KEY}' and '{ENV_IMAGE_KEY}'. "
                f"Found columns: {list(df.columns)}"
            )

        for local_idx in range(len(df)):
            row = df.iloc[local_idx]
            ep_idx = int(row["episode_index"])
            t_idx = int(row["task_index"])

            if cur_ep_idx is not None and ep_idx != cur_ep_idx:
                if cur_ep_data and cur_ep_data["img_cam0"]:
                    seen_tasks.add(cur_ep_data["task_index"][0])
                    flush_episode(replay_buffer, cur_ep_data)
                    episodes_added += 1
                    if episodes_added == 1 or episodes_added % 50 == 0:
                        print(f"[INFO] Flushed {episodes_added} episodes to disk")
                cur_ep_data = None
                cur_ep_idx = None

            if task_filter is not None and t_idx not in task_filter:
                continue

            if max_episodes is not None and episodes_added >= max_episodes:
                break

            if cur_ep_data is None:
                cur_ep_data = _new_accum()
                cur_ep_idx = ep_idx

            wrist_img = decode_image_from_parquet(row[WRIST_IMAGE_KEY])
            env_img = decode_image_from_parquet(row[ENV_IMAGE_KEY])

            cur_ep_data["img_cam0"].append(
                resize_image(wrist_img, resize_hw)
            )
            cur_ep_data["img_cam1"].append(
                resize_image(env_img, resize_hw)
            )
            cur_ep_data["action"].append(
                np.array(row["action"], dtype=np.float32)
            )
            cur_ep_data["task_index"].append(t_idx)

        del df
        gc.collect()

        if max_episodes is not None and episodes_added >= max_episodes:
            break

    if cur_ep_data and cur_ep_data["img_cam0"]:
        if max_episodes is None or episodes_added < max_episodes:
            seen_tasks.add(cur_ep_data["task_index"][0])
            flush_episode(replay_buffer, cur_ep_data)
            episodes_added += 1

    return seen_tasks, episodes_added


def main() -> None:
    args = parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input_path))
    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    resize_hw = (int(args.resize_hw[0]), int(args.resize_hw[1]))

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    if os.path.exists(output_path):
        if args.overwrite:
            print(f"[INFO] Removing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                f"Use --overwrite to replace it."
            )

    task_filter = set(args.task_indices) if args.task_indices else None

    task_descriptions = load_task_descriptions(input_path)
    print(f"[INFO] Loaded {len(task_descriptions)} task descriptions.")

    parquet_files = find_parquet_files(input_path)
    print(f"[INFO] Found {len(parquet_files)} parquet data files.")

    fmt = detect_format(input_path)
    print(f"[INFO] Detected data format: {fmt}")

    # Create on-disk zarr so episodes stream to disk incrementally.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if ZARR_V3:
        store = zarr.storage.LocalStore(output_path)
    else:
        store = zarr.DirectoryStore(output_path)
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)

    if fmt == "video":
        seen_tasks, episodes_added = process_video_format(
            input_path, parquet_files, resize_hw, task_filter,
            args.max_episodes, replay_buffer,
        )
    else:
        seen_tasks, episodes_added = process_inline_format(
            parquet_files, resize_hw, task_filter,
            args.max_episodes, replay_buffer,
        )

    if replay_buffer.n_episodes == 0:
        raise RuntimeError("No episodes were processed. Check input path and filters.")

    print(f"[INFO] Total episodes: {replay_buffer.n_episodes}")
    print(f"[INFO] Total steps: {replay_buffer.n_steps}")
    print(f"[INFO] Unique tasks seen: {sorted(seen_tasks)}")

    relevant_descriptions = {
        k: v for k, v in task_descriptions.items() if k in seen_tasks
    }
    if relevant_descriptions:
        replay_buffer.root.attrs["task_descriptions"] = json.dumps(
            relevant_descriptions
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print(f"[INFO] Validating zarr at {output_path} ...")
    group = zarr.open(os.path.expanduser(output_path), mode="r")

    if "data" not in group or "meta" not in group:
        raise RuntimeError("Zarr is missing 'data' or 'meta' groups.")

    data_grp = group["data"]
    if hasattr(data_grp, "array_keys"):
        keys = list(data_grp.array_keys())
    else:
        keys = list(data_grp.keys())
    print(f"[INFO] Zarr data keys: {keys}")

    episode_ends = np.array(group["meta"]["episode_ends"])
    n_steps = int(episode_ends[-1]) if episode_ends.size > 0 else 0
    n_episodes = int(episode_ends.shape[0])
    print(f"[INFO] Zarr total steps:    {n_steps}")
    print(f"[INFO] Zarr total episodes: {n_episodes}")

    for required_key in ("img_cam0", "img_cam1", "action", "task_index"):
        if required_key not in keys:
            raise RuntimeError(f"Zarr missing required key '{required_key}'.")

    cam0_shape = group["data"]["img_cam0"].shape
    cam1_shape = group["data"]["img_cam1"].shape
    action_shape = group["data"]["action"].shape
    print(f"[INFO] img_cam0 shape: {cam0_shape}")
    print(f"[INFO] img_cam1 shape: {cam1_shape}")
    print(f"[INFO] action shape:   {action_shape}")

    expected_img = (n_steps, resize_hw[0], resize_hw[1], 3)
    if cam0_shape != expected_img:
        print(f"[WARN] img_cam0 shape {cam0_shape} != expected {expected_img}")
    if cam1_shape != expected_img:
        print(f"[WARN] img_cam1 shape {cam1_shape} != expected {expected_img}")

    if "task_descriptions" in (group.attrs if hasattr(group, "attrs") else {}):
        td = json.loads(group.attrs["task_descriptions"])
        print(f"[INFO] Task descriptions stored: {len(td)} tasks")
        for idx, desc in sorted(td.items(), key=lambda x: int(x[0])):
            print(f"       task {idx}: {desc}")

    print("[INFO] Conversion complete.")


if __name__ == "__main__":
    main()