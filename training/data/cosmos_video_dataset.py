#!/usr/bin/env python3
"""Cosmos-Transfer で天候拡張された動画 -> LeRobotDataset。

動画はカメラ画像のみを差し替えたもの（幾何・エゴモーションは元エピソードと同一）。
軌跡(traj_data.pkl)・言語指示(traj_prompt.txt)は元の実データをそのまま流用し、
動画フレーム数 M と実フレーム数 N の差は両端が一致する線形リサンプリングで対応付ける:
    real_index = round(video_index * (N - 1) / (M - 1))
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 5
IMG_H, IMG_W = 224, 224

VIDEO_NAME_RE = re.compile(r"^(?P<prefix>[a-z]+)_episode(?P<num>\d+)__.+\.mp4$")


def build_features() -> dict:
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


def load_raw_episode(ep_dir: Path):
    with (ep_dir / "traj_data.pkl").open("rb") as f:
        d = pickle.load(f)
    position = np.asarray(d["position"], dtype=np.float32)
    yaw = np.asarray(d["yaw"], dtype=np.float32)
    prompts = (ep_dir / "traj_prompt.txt").read_text(encoding="utf-8").splitlines()
    return position, yaw, prompts


def find_episode_dir(root: Path, local_ep: int) -> Path | None:
    for fmt in (f"episode{local_ep}", f"episode{local_ep:02d}", f"episode{local_ep:03d}"):
        p = root / fmt
        if p.exists():
            return p
    return None


def read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (IMG_H, IMG_W):
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        frames.append(frame)
    cap.release()
    return frames


# --- global cosmos episode番号 -> (raw_root配下の相対パス, ローカルepisode番号) ---
def resolve_junction(n: int):
    return ("navvla_tsudanuma_junction", n) if 1 <= n <= 48 else None


def resolve_linestop(n: int):
    return ("navvla_tsudanuma_linestop", n) if 1 <= n <= 30 else None


def resolve_nav(n: int):
    # 1-201 (nav201本体) は raw が手元に無いため skip
    if 202 <= n <= 229:
        return ("candidate_dataset/navvla_tsudanuma_nav4", n - 201)
    if 230 <= n <= 256:
        return ("candidate_dataset/navvla_tsudanuma_nav5", n - 229)
    if 257 <= n <= 286:
        return ("candidate_dataset/navvla_tsudanuma_nav6", n - 256)
    return None


def resolve_random(n: int):
    if 1 <= n <= 43:
        return ("candidate_dataset/navvla_tsudanuma_random1", n)
    if 44 <= n <= 105:
        return ("candidate_dataset/navvla_tsudanuma_random2-3", n - 43)
    if 106 <= n <= 220:
        return ("candidate_dataset/navvla_tsudanuma_random4-5", n - 105)
    if 221 <= n <= 249:
        return ("candidate_dataset/navvla_tsudanuma_random6", n - 220)
    if 250 <= n <= 341:
        return ("candidate_dataset/navvla_tsudanuma_random7", n - 249)
    return None


RESOLVERS = {
    "junction": resolve_junction,
    "linestop": resolve_linestop,
    "nav": resolve_nav,
    "random": resolve_random,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", required=True, type=Path)
    p.add_argument("--raw-root", required=True, type=Path, help="training/data/raw")
    p.add_argument("--prefix", required=True, choices=list(RESOLVERS))
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resolver = RESOLVERS[args.prefix]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=build_features(),
        root=args.root,
        robot_type="diffdrive",
        use_videos=True,
    )

    videos = sorted(args.video_dir.glob(f"{args.prefix}_episode*.mp4"))
    cache: dict[Path, tuple] = {}
    n_saved = n_skipped_no_raw = n_skipped_short = 0

    for vid_path in videos:
        m = VIDEO_NAME_RE.match(vid_path.name)
        if not m:
            continue
        num = int(m["num"])
        resolved = resolver(num)
        if resolved is None:
            n_skipped_no_raw += 1
            continue

        rel_root, local_ep = resolved
        ep_dir = find_episode_dir(args.raw_root / rel_root, local_ep)
        if ep_dir is None:
            n_skipped_no_raw += 1
            continue

        if ep_dir not in cache:
            cache[ep_dir] = load_raw_episode(ep_dir)
        position, yaw, prompts = cache[ep_dir]
        n_real = len(position)

        frames = read_video_frames(vid_path)
        n_video = len(frames)
        if n_video < 2 or n_real < 2:
            n_skipped_short += 1
            continue

        for video_idx in range(n_video):
            real_idx = round(video_idx * (n_real - 1) / (n_video - 1))
            # action: そのフレーム自身の絶対姿勢 [x, y, cos(yaw), sin(yaw)]。
            # 共通原点への変換はWaypointRebaseProcessorStep(学習時)が担う。
            action = np.array(
                [position[real_idx, 0], position[real_idx, 1], np.cos(yaw[real_idx]), np.sin(yaw[real_idx])],
                dtype=np.float32,
            )
            state = np.zeros(2, dtype=np.float32)

            dataset.add_frame({
                "observation.images.front": frames[video_idx],
                "observation.state": state,
                "action": action,
                "task": prompts[real_idx].strip(),
            })

        dataset.save_episode()
        n_saved += 1
        print(f"[cosmos] saved {vid_path.name}: {n_video} frames <- {rel_root}/{ep_dir.name} ({n_real} frames)")

    print(f"[cosmos] done: saved={n_saved} skipped_no_raw={n_skipped_no_raw} skipped_short={n_skipped_short}")


if __name__ == "__main__":
    main()
