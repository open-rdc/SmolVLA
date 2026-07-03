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
            "shape": (2,),
            "names": ["dx_body", "dyaw"],
        },
    }


def to_body_frame(delta_xy_global: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotmat = np.array([[c, -s], [s, c]])
    return delta_xy_global.dot(rotmat)


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

    for ep_dir in sorted(args.input.glob("episode*")):
        position, yaw, prompts = load_episode(ep_dir)
        n = len(position)

        for t in range(n - 1):  # drop last frame: no t+1 -> no action
            # --- image: NavVLA saved BGR via cv2 -> convert to RGB ---
            img = cv2.imread(str(ep_dir / f"{t}.jpg"))           # HWC, BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # HWC, RGB, uint8

            # --- action: body-frame increment t -> t+1 ---  [Δx_body, Δyaw]
            dxy_body = to_body_frame(position[t + 1] - position[t], yaw[t])
            dyaw     = yaw[t + 1] - yaw[t]          # yaw is unwrapped -> plain diff
            action   = np.array([dxy_body[0], dyaw], dtype=np.float32)

            # --- state: PREVIOUS increment (t-1 -> t) / dt ---  [v, ω]
            # (copycat note: never use the SAME-step increment here = that is the label)
            if t == 0:
                state = np.zeros(2, dtype=np.float32)
            else:
                v     = to_body_frame(position[t] - position[t - 1], yaw[t - 1])[0] / DT
                omega = (yaw[t] - yaw[t - 1]) / DT
                state = np.array([v, omega], dtype=np.float32)

            dataset.add_frame({
                "observation.images.front": img,
                "observation.state": state,
                "action": action,
                "task": prompts[t].strip(),
            })

        dataset.save_episode()
        print(f"[convert] saved {ep_dir.name}: {n - 1} frames")


if __name__ == "__main__":
    main()
