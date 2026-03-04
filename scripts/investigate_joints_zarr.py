"""
Investigate joints zarr dataset: raw action stats and whether values look like
absolute angles vs deltas. Also reports JointsImageDataset normalizer and a sample batch.
Usage: python scripts/investigate_joints_zarr.py [path/to/replay.zarr]
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np

def main():
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else "joints_50demos_replay.zarr"
    zarr_path = os.path.expanduser(zarr_path)
    if not os.path.exists(zarr_path):
        print(f"Zarr not found: {zarr_path}")
        print("Usage: python scripts/investigate_joints_zarr.py [path/to/replay.zarr]")
        return

    import zarr
    print("=" * 60)
    print("1. RAW ZARR CONTENTS")
    print("=" * 60)
    root = zarr.open(zarr_path, mode="r")
    print("Top-level keys:", list(root.keys()))
    if "data" in root:
        print("data keys:", list(root["data"].keys()))
    if "meta" in root:
        print("meta keys:", list(root["meta"].keys()))
        if "episode_ends" in root["meta"]:
            ends = np.array(root["meta"]["episode_ends"])
            print("episode_ends shape:", ends.shape, "  first 5:", ends[:5], "  last 5:", ends[-5:])

    # Assume direct keys or data/action, data/img
    if "data" in root and "action" in root["data"]:
        action_arr = np.array(root["data"]["action"][:])
    elif "action" in root:
        action_arr = np.array(root["action"][:])
    else:
        print("Could not find 'action' under root or root['data']")
        return

    print()
    print("2. RAW ACTION ARRAY (from zarr)")
    print("-" * 60)
    print("shape:", action_arr.shape)
    print("dtype:", action_arr.dtype)
    n_steps, action_dim = action_arr.shape[0], action_arr.shape[-1]
    print("action_dim:", action_dim)
    print("min (per dim):", action_arr.min(axis=0))
    print("max (per dim):", action_arr.max(axis=0))
    print("mean (per dim):", action_arr.mean(axis=0))
    print("std  (per dim):", action_arr.std(axis=0))
    print()
    print("First 5 rows (raw):")
    print(action_arr[:5])
    print("Last 5 rows (raw):")
    print(action_arr[-5:])

    # Consecutive differences: if these are small and raw values are large, it's absolute; if raw is small and diffs similar, could be delta
    diffs = np.diff(action_arr, axis=0)
    print()
    print("3. CONSECUTIVE DIFFERENCES (action[t+1] - action[t])")
    print("-" * 60)
    print("diffs shape:", diffs.shape)
    print("diffs min (per dim):", diffs.min(axis=0))
    print("diffs max (per dim):", diffs.max(axis=0))
    print("diffs mean (per dim):", diffs.mean(axis=0))
    print("diffs std  (per dim):", diffs.std(axis=0))
    print("Sample diffs (first 5):")
    print(diffs[:5])

    # Heuristic: if range of action is small (e.g. all dims in [-0.2, 0.2]) and similar to diff range -> likely deltas
    raw_range = action_arr.max(axis=0) - action_arr.min(axis=0)
    diff_range = diffs.max(axis=0) - diffs.min(axis=0)
    print()
    print("4. INTERPRETATION (absolute vs delta heuristic)")
    print("-" * 60)
    print("Raw action range (max - min) per dim:", raw_range)
    print("Diff range per dim:                 ", diff_range)
    if np.all(raw_range < 1.0) and np.all(np.abs(action_arr).max() < 2.0):
        print("--> Values are SMALL (typical of joint angle DELTAS in rad). Likely stored as DELTAS.")
    else:
        print("--> Values span a larger range (typical of ABSOLUTE joint angles in rad). Likely ABSOLUTE.")
    print()

    # JointsImageDataset path and normalizer
    print("5. JOINTS IMAGE DATASET (cleandiffuser) NORMALIZER & SAMPLE")
    print("-" * 60)
    from cleandiffuser.dataset.joints_dataset import JointsImageDataset

    dataset = JointsImageDataset(
        zarr_path,
        obs_keys=("img", "action"),
        horizon=16,
        pad_before=1,
        pad_after=7,
        abs_action=False,
    )
    norm = dataset.normalizer["action"]
    print("MinMaxNormalizer min (from replay_buffer['action']):", norm.min)
    print("MinMaxNormalizer max (from replay_buffer['action']):", norm.max)
    print("MinMaxNormalizer range:                             ", norm.range)
    print()
    # One batch: raw sample -> normalized -> unnormalized should match
    idx = 0
    sample = dataset.sampler.sample_sequence(idx)
    raw_action = sample["action"].astype(np.float32)
    normalized = norm.normalize(raw_action)
    unnormalized = norm.unnormalize(normalized)
    print("Sample sequence idx=0:")
    print("  raw action (first step):     ", raw_action[0])
    print("  normalized (first step):    ", normalized[0])
    print("  unnormalized (first step):   ", unnormalized[0])
    print("  max |raw - unnormalized|:   ", np.abs(raw_action - unnormalized).max())
    print()
    print("Done.")

if __name__ == "__main__":
    main()
