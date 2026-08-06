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

# 時系列画像コンテキスト（camera1=現在 / camera2=1秒前 / camera3=2秒前）のラグ。
# 5 フレーム @ FPS=5 = 1 秒。後で調整できるよう定数化してある。
# 設計: ~/.company/engineering/docs/smolvla-temporal-context-architecture.md 決定3
HISTORY_STRIDE_FRAMES = 5


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
    """Feature schema for the LeRobotDataset.

    SmolVLA が元々持つ 3 カメラ視点スロットを「複数視点」ではなく「時間軸」に転用する:
    camera1=現在(t) / camera2=1秒前(t-5) / camera3=2秒前(t-10)。
    こうするとモデル本体（SmolVLM2 の visual encoder）の改造が不要になる。
    front キーは使わないので、学習コマンドの --rename_map も不要。
    詳細設計: ~/.company/engineering/docs/smolvla-temporal-context-architecture.md 決定2,3。
    """
    image_feature = {
        "dtype": "video",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.camera1": dict(image_feature),  # 現在 (t)
        "observation.images.camera2": dict(image_feature),  # 1秒前 (t - HISTORY_STRIDE_FRAMES)
        "observation.images.camera3": dict(image_feature),  # 2秒前 (t - 2*HISTORY_STRIDE_FRAMES)
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

    ep_dirs = sorted(p for p in args.input.iterdir() if p.is_dir() and (p / "traj_data.pkl").exists())
    for ep_dir in ep_dirs:
        position, yaw, prompts = load_episode(ep_dir)
        n = len(position)

        # エピソード内の画像キャッシュ。参照するのは t / t-5 / t-10 の3枚だけなので、
        # 直近 2*HISTORY_STRIDE_FRAMES+1 枚だけ保持して古いものから捨てる。
        # 各フレームの jpg 読み込みは1回で済む（I/Oを3倍にしない）。
        cache: dict[int, np.ndarray] = {}

        def load_img(i: int, ep_dir: Path = ep_dir, cache: dict = cache) -> np.ndarray:
            """i 番目のフレームを RGB uint8 で返す（キャッシュ経由）。"""
            img = cache.get(i)
            if img is None:
                bgr = cv2.imread(str(ep_dir / f"{i}.jpg"))       # HWC, BGR
                if bgr is None:
                    raise FileNotFoundError(f"cannot read {ep_dir / f'{i}.jpg'}")
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)       # HWC, RGB, uint8
                cache[i] = img
            return img

        for t in range(n - 1):  # drop last frame: no t+1 -> no action
            # --- images: 現在(t) / 1秒前(t-5) / 2秒前(t-10) ---
            # 負のインデックスになるエピソード先頭は 0 にクランプ = 先頭フレームを複製して padding。
            # 設計: ~/.company/engineering/docs/smolvla-temporal-context-architecture.md 決定3,4
            img    = load_img(t)                                     # 現在 (t)
            img_t1 = load_img(max(t - HISTORY_STRIDE_FRAMES, 0))     # 1秒前
            img_t2 = load_img(max(t - 2 * HISTORY_STRIDE_FRAMES, 0)) # 2秒前
            cache.pop(t - 2 * HISTORY_STRIDE_FRAMES - 1, None)       # もう参照しない分を解放

            # --- action: body-frame increment t -> t+1 ---  [Δx_body, Δyaw]
            dxy_body = to_body_frame(position[t + 1] - position[t], yaw[t])
            dyaw     = yaw[t + 1] - yaw[t]          # yaw is unwrapped -> plain diff
            action   = np.array([dxy_body[0], dyaw], dtype=np.float32)

            # --- state: 常にゼロ = vision+language だけで予測させる（stateless化）---
            # 実機側 navigation.py も state=zeros 固定で推論するため、学習も分布を合わせる。
            state = np.zeros(2, dtype=np.float32)

            dataset.add_frame({
                "observation.images.camera1": img,      # 現在 (t)
                "observation.images.camera2": img_t1,   # 1秒前
                "observation.images.camera3": img_t2,   # 2秒前
                "observation.state": state,
                "action": action,
                # Cosmos天候拡張エピソードは traj_prompt.txt が jpg より1行少ないことがあるので
                # 末尾でクランプして直前の指示を使い回す（IndexError 防止）。
                "task": prompts[min(t, len(prompts) - 1)].strip(),
            })

        dataset.save_episode()
        print(f"[convert] saved {ep_dir.name}: {n - 1} frames")


if __name__ == "__main__":
    main()
