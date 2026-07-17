#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 5
DT = 1.0 / FPS
IMG_H, IMG_W = 224, 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="NavVLA dataset dir containing episodeNN/ folders")
    p.add_argument("--repo-id", required=True,
                   help="LeRobotDataset repo id, e.g. open-rdc/tsudanuma_nav6")
    p.add_argument("--root", type=Path, default=None,
                   help="local output dir (omit -> default HF cache, still local/no push)")
    return p.parse_args()


def build_features() -> dict:
    """Feature schema for the LeRobotDataset."""
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["v", "omega"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "cos_yaw", "sin_yaw"],
        },
    }


def load_episode(ep_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one NavVLA episode: positions (N,2), yaws (N,), per-frame prompts."""
    with (ep_dir / "traj_data.pkl").open("rb") as f:
        d = pickle.load(f)
    position = np.asarray(d["position"], dtype=np.float32)  # (N, 2) global meters
    yaw = np.asarray(d["yaw"], dtype=np.float32)            # (N,)   unwrapped rad
    prompts = (ep_dir / "traj_prompt.txt").read_text(encoding="utf-8").splitlines()
    return position, yaw, prompts


def main() -> None:
    args = parse_args()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=build_features(),
        root=args.root,
        robot_type="diffdrive",
        use_videos=True,
    )

    ep_dirs = sorted(p for p in args.input.iterdir() if p.is_dir() and (p / "traj_data.pkl").exists())
    for ep_dir in ep_dirs:
        position, yaw, prompts = load_episode(ep_dir)
        n = len(position)

        for t in range(n):
            # --- image: NavVLA saved BGR via cv2 -> convert to RGB ---
            img = cv2.imread(str(ep_dir / f"{t}.jpg"))           # HWC, BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # HWC, RGB, uint8

            # --- action: frame t's own absolute pose [x, y, cos(yaw), sin(yaw)] ---
            # 各waypointを共通原点(chunkの先頭)基準に変換する処理は
            # WaypointRebaseProcessorStep(学習時)が担う。ここでは絶対姿勢を保存するだけ。
            action = np.array(
                [position[t, 0], position[t, 1], np.cos(yaw[t]), np.sin(yaw[t])],
                dtype=np.float32,
            )

            # --- state: 常にゼロ = vision+language だけで予測させる（stateless化）---
            # 実機側 navigation.py も state=zeros 固定で推論するため、学習も分布を合わせる。
            state = np.zeros(2, dtype=np.float32)

            dataset.add_frame({
                "observation.images.front": img,
                "observation.state": state,
                "action": action,
                # Cosmos天候拡張エピソードは traj_prompt.txt が jpg より1行少ない
                # (オフバイワン)ことがあるため、最終フレームは直前の指示を使い回す。
                "task": prompts[min(t, len(prompts) - 1)].strip(),
            })

        dataset.save_episode()
        print(f"[convert] saved {ep_dir.name}: {n} frames")


if __name__ == "__main__":
    main()
