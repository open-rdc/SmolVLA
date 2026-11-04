#!/usr/bin/env python3
"""注釈済みの周回エピソードを、指示が変わる位置で区切って 1 指示 1 エピソードにする。

  - MERGE_GAP フレーム以下の未割当("No language instruction")の隙間は、直前の指示で埋める
  - 画像はハードリンクで置く(容量を食わない)。番号は 0 から振り直す
  - 未割当のまま残ったエピソードだけを集めた review/ (シンボリックリンク)を作る。
    lang_anotation_tool.py に review/ を渡せば、手作業の要るものだけを順に開ける
    (ツールの書き込みはリンク先の本体に入る)

出力:
    <out_root>/<bag名>/episode001/0.jpg ... traj_data.pkl traj_prompt.txt timestamps.npy
    <out_root>/review/<bag名>_episode001 -> ../<bag名>/episode001

    python training/data/split_lap_by_prompt.py ~/tsudanuma_data/laps/tsudanuma_20261010_* \
        --out-root ~/tsudanuma_data/episodes
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

DUMMY = "No language instruction"
MERGE_GAP = 10  # 2 秒


def segments(prompts: list[str]) -> list[list]:
    segs: list[list] = []
    for i, s in enumerate(prompts):
        if segs and segs[-1][2] == s:
            segs[-1][1] = i
        else:
            segs.append([i, i, s])
    return segs


def fill_short_gaps(prompts: list[str]) -> list[str]:
    out = list(prompts)
    for k, (a, b, s) in enumerate(segments(prompts)):
        if s == DUMMY and k > 0 and b - a + 1 <= MERGE_GAP:
            for i in range(a, b + 1):
                out[i] = out[a - 1]
    return out


def split(lap: Path, out_root: Path, review: Path) -> None:
    src = lap / "episode01"
    prompts = (src / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")
    traj = pickle.load(open(src / "traj_data.pkl", "rb"))
    stamps = np.load(src / "timestamps.npy")
    assert len(prompts) == len(traj["position"]) == len(stamps)
    prompts = fill_short_gaps(prompts)
    dst_root = out_root / lap.name
    dst_root.mkdir(parents=True, exist_ok=False)
    n_review = 0
    segs = segments(prompts)
    for k, (a, b, s) in enumerate(segs, start=1):
        ep = dst_root / f"episode{k:03d}"
        ep.mkdir()
        for j, i in enumerate(range(a, b + 1)):
            os.link(src / f"{i}.jpg", ep / f"{j}.jpg")
        with (ep / "traj_data.pkl").open("wb") as f:
            pickle.dump({"position": traj["position"][a:b + 1], "yaw": traj["yaw"][a:b + 1]}, f)
        (ep / "traj_prompt.txt").write_text("\n".join([s] * (b - a + 1)), encoding="utf-8")
        np.save(ep / "timestamps.npy", stamps[a:b + 1])
        if s == DUMMY:
            (review / f"{lap.name}_{ep.name}").symlink_to(Path("..") / lap.name / ep.name)
            n_review += 1
    lens = [b - a + 1 for a, b, _ in segs]
    print(f"{lap.name}: {len(segs)} エピソード (長さ 中央 {int(np.median(lens))} / 最短 {min(lens)} / 最長 {max(lens)} フレーム)"
          f" / 要手作業 {n_review} 本")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laps", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    out_root = args.out_root.expanduser()
    review = out_root / "review"
    review.mkdir(parents=True, exist_ok=True)
    for lap in args.laps:
        split(lap.expanduser(), out_root, review)


if __name__ == "__main__":
    main()
