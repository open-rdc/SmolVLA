#!/usr/bin/env python3
"""検証用周回のフレームのうち、分岐(学習用の周回が同じ場所・同じ向きから別々の方向へ進んだ地点)を探す。

1. すべての周回を旧カメラの nav データの座標へ合わせる(transfer_prompts_from_old.py の align と同じ)。
   旧 nav と新しい bag は同じ地点で LIO を起動しているので、合わせた後は全周回がほぼ同じ座標になる。
   ただし旧 nav は何本もの走行が道幅いっぱいに重なっているので、2〜4° 回った位置でも「どれかに近い」ことになり
   1.5〜2 m ずれる周回があった。そこで続けて、検証用周回の近く(REFINE_M 以内)の点だけで検証用周回へ合わせ直す。
2. 検証用周回の各フレームについて、学習用の周回のフレームのうち
     距離 RADIUS_M 以内・向きの差 MAX_DYAW 以内(同じ向きに通った)
   のものを集め、それぞれ AHEAD_M 進んだ先の方向(今の向きから測った角度)を求める。
   時間ではなく進んだ距離で測るので、曲がり始めのタイミングの差に左右されにくい。
3. 周回ごとの方向(中央値)を 左(> TURN_DEG)/ 直進(|角度| < STRAIGHT_DEG)/ 右(< -TURN_DEG)に分け、
   2 種類以上が別々の周回で出ていれば分岐とする。直進と旋回の間の周回は数えない
   (普通の角で曲がり始めの位置が周回ごとに違うだけのものを分岐と誤判定しないため)。

4. 行ける方向ごとに、その方向へ進んだ周回の「一番近いフレームから H フレーム先までの向き変化 Ψ」の中央値を
   psi_ref_{L,S,R} に入れる(その方向の周回が無ければ NaN)。指示どおりに進めたかを Ψ で判定するときの正解。
   同じフレームの学習時の指示文(traj_prompt.txt)の多数決を prompt_ref_{L,S,R} に入れる(無ければ "")。

出力(npz): t, branch(bool), options("LS" "LR" "LSR" など、その地点から行ける方向), n_laps, aligned_xy, psi_ref_{L,S,R}, prompt_ref_{L,S,R},
          xy_<周回名>(合わせた後の全周回の軌跡)
    python training/data/find_branch_points.py ~/tsudanuma_data/laps --test tsudanuma_20261010_134655 \\
        --old-pkl-root ~/tsudanuma_data/old_nav/navvla_tsudanuma_nav --out branch_134655.npz
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_prompts_from_old import align, trimmed_icp  # noqa: E402

RADIUS_M = 2.5
REFINE_M = 4.0           # 検証用周回へ合わせ直すときに使う点の距離の上限 [m]
MAX_DYAW = np.radians(45)
AHEAD_M = 10.0
TURN_DEG = 45.0          # これより大きく向きを変えたら旋回
STRAIGHT_DEG = 15.0      # これ以内なら直進。間の周回は曲がり始めの位置の違いで揺れるので数えない
H = 24                   # 方向ごとの正解 Ψ を測る長さ(4.8 秒 = 24 フレーム, eval_monitor.py と同じ)


def load(lap: Path):
    t = pickle.load(open(lap / "episode01" / "traj_data.pkl", "rb"))
    return np.asarray(t["position"], float), np.unwrap(np.asarray(t["yaw"], float))


def ahead_angle(xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """各フレームから AHEAD_M 進んだ地点の方向を、今の向きから測った角度[度]で返す(届かなければ NaN)。"""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    j = np.searchsorted(s, s + AHEAD_M)
    out = np.full(len(xy), np.nan)
    ok = j < len(xy)
    d = xy[j[ok]] - xy[ok]
    a = np.arctan2(d[:, 1], d[:, 0]) - yaw[ok]
    out[ok] = np.degrees((a + np.pi) % (2 * np.pi) - np.pi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("laps_root", type=Path)
    ap.add_argument("--test", required=True)
    ap.add_argument("--old-pkl-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dense = np.concatenate([np.asarray(pickle.load(open(d / "traj_data.pkl", "rb"))["position"], float)
                            for d in sorted(a.old_pkl_root.glob("episode*"))])
    laps = sorted(p for p in a.laps_root.iterdir() if (p / "episode01" / "traj_data.pkl").exists())
    laps.sort(key=lambda p: p.name == a.test)  # 検証用は最後に。うまく合った周回を手がかり(ref)にして合わせる
    aligned, ref = {}, None
    for lap in laps:
        xy, yaw = load(lap)
        R, tr, d = align(xy, dense, ref)
        rot = np.arctan2(R[1, 0], R[0, 0])
        aligned[lap.name] = (xy @ R.T + tr, yaw + rot)
        print(f"{lap.name}: 回転 {np.degrees(rot):6.1f}° / 旧データとの残差 中央 {np.median(d):.2f} m"
              f" / 1 m 以内 {np.mean(d < 1.0) * 100:.0f}%")
        if ref is None and np.median(d) < 0.5:
            ref = aligned[lap.name][0]

    txy, tyaw = aligned[a.test]
    ttree = cKDTree(txy)
    for name, (xy, yaw) in list(aligned.items()):
        if name == a.test:
            continue
        R, tr, d = trimmed_icp(xy, ttree, txy, np.eye(2), np.zeros(2), max_d=REFINE_M, iters=60)
        aligned[name] = (xy @ R.T + tr, yaw + np.arctan2(R[1, 0], R[0, 0]))
        near = d < 6
        print(f"{name}: 検証用周回へ合わせ直し 回転 {np.degrees(np.arctan2(R[1, 0], R[0, 0])):5.2f}°"
              f" / 6 m 以内の点のずれ 中央 {np.median(d[near]):.2f} m")
    pts, yaws, angs, ids, psis, prompts = [], [], [], [], [], []
    for k, (name, (xy, yaw)) in enumerate(aligned.items()):
        if name == a.test:
            continue
        psi = np.full(len(xy), np.nan)
        psi[:-H] = yaw[H:] - yaw[:-H]
        pr = (a.laps_root / name / "episode01" / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")
        prompts += [pr[j].strip() if j < len(pr) else "" for j in range(len(xy))]
        pts.append(xy); yaws.append(yaw); angs.append(ahead_angle(xy, yaw)); ids.append(np.full(len(xy), k)); psis.append(psi)
    pts, yaws, angs, ids, psis = map(np.concatenate, (pts, yaws, angs, ids, psis))
    tree = cKDTree(pts)
    n = len(txy)
    branch = np.zeros(n, bool)
    options = np.full(n, "", dtype=object)
    n_laps = np.zeros(n, int)
    psi_ref = {c: np.full(n, np.nan) for c in "LSR"}
    prompt_ref = {c: np.full(n, "", dtype=object) for c in "LSR"}
    for i, nb in enumerate(tree.query_ball_point(txy, RADIUS_M)):
        nb = np.array(nb, int)
        if len(nb) == 0:
            continue
        dy = np.abs((yaws[nb] - tyaw[i] + np.pi) % (2 * np.pi) - np.pi)
        nb = nb[(dy < MAX_DYAW) & ~np.isnan(angs[nb])]
        lap_dirs, lap_psi, lap_prompt = {}, {}, {}
        for k in np.unique(ids[nb]):
            m = float(np.median(angs[nb][ids[nb] == k]))
            if m > TURN_DEG or m < -TURN_DEG or abs(m) < STRAIGHT_DEG:
                lap_dirs[k] = "L" if m > TURN_DEG else "R" if m < -TURN_DEG else "S"
                mine = nb[ids[nb] == k]
                j = mine[np.argmin(np.linalg.norm(pts[mine] - txy[i], axis=1))]
                lap_psi[k], lap_prompt[k] = psis[j], prompts[j]
        for c in "LSR":
            v = [lap_psi[k] for k, d in lap_dirs.items() if d == c and not np.isnan(lap_psi[k])]
            if v:
                psi_ref[c][i] = np.median(v)
            w = [lap_prompt[k] for k, d in lap_dirs.items() if d == c and lap_prompt[k]]
            if w:
                prompt_ref[c][i] = Counter(w).most_common(1)[0][0]
        n_laps[i] = len(lap_dirs)
        opts = "".join(c for c in "LSR" if c in lap_dirs.values())
        options[i] = opts
        branch[i] = len(opts) >= 2
    np.savez(a.out, t=np.arange(n), branch=branch, options=options.astype(str), n_laps=n_laps, aligned_xy=txy,
             **{f"psi_ref_{c}": v for c, v in psi_ref.items()},
             **{f"prompt_ref_{c}": v.astype(str) for c, v in prompt_ref.items()},
             **{f"xy_{k}": v[0] for k, v in aligned.items()})  # 合わせた後の全周回の軌跡(目視確認用)
    print(f"検証用 {a.test}: {n} フレーム中 分岐 {branch.sum()} ({branch.mean() * 100:.1f}%)"
          f" / 近くを通った学習用周回が 2 本以上 {np.mean(n_laps >= 2) * 100:.0f}%")
    for o in sorted(set(options[branch])):
        print(f"  行ける方向 {o}: {np.sum(options == o)} フレーム")


if __name__ == "__main__":
    main()
