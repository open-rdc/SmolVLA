#!/usr/bin/env python3
"""旧カメラの nav データの言語指示を、新しい周回エピソードへ位置で転写する。

旧 nav データ(navvla_tsudanuma_nav)と新しい bag は同じ地点から LIO を起動しているため、
オドメトリ座標がほぼ一致する。ずれ(LIO の再起動やドリフト)は ICP で新→旧へ合わせる。

各新フレームについて、旧フレームのうち
  - 距離 RADIUS_M 以内
  - 向きの差 MAX_DYAW 以内
  - 4.8 秒先までの向き変化 Ψ の行動クラス(左折/右折/直進)が一致
のものを候補とし、その指示を候補にする。行動クラスで絞るのは、分岐の手前で
旧データが別方向に曲がった指示を拾わないため。

指示の決め方:
  - 今の指示が候補に残っている間は使い続ける(区間の途中で指示が揺れないように)
  - 切り替えるときは、前に処理した bag が近く(DIVERSITY_M 以内)で使った指示を避け、
    残りの中で支持する旧エピソード数が最も多い指示を選ぶ
    → 同じ場所でも bag ごとにできるだけ違う指示になる
  - 候補が無いフレームは "No language instruction" にし、手作業の対象として報告する

    python training/data/transfer_prompts_from_old.py \
        ~/tsudanuma_data/laps/tsudanuma_20261010_150555 ... \
        --old-pkl-root <raw_pkl/navvla_tsudanuma_nav> --old-prompts nav_prompts_all.json [--write]

  注釈済みの周回を書き換えずに，新しい周回だけに指示を入れるときは --fixed に注釈済みのほうを並べる
  (固定した周回の指示は「前に処理した bag」として，同じ場所で同じ指示を避けるのに使う):
    python training/data/transfer_prompts_from_old.py ~/tsudanuma_data/laps/tsudanuma_20261011_* \
        --fixed ~/tsudanuma_data/laps/tsudanuma_20261010_* --old-pkl-root ... --old-prompts ... --write
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DUMMY = "No language instruction"
H = 24                    # Ψ の先読み(4.8 s)。eval_monitor.py と同じ
THETA = 0.090             # 行動クラスのしきい値 [rad]。eval_monitor.py と同じ
RADIUS_M = 2.5
MAX_DYAW = np.radians(45)
DIVERSITY_M = 10.0
HOLD_GAP = 5              # 今の指示が候補から消えても、この フレーム数までは使い続ける
TRIM_M = 2.0              # trimmed_icp で使う点の距離の上限 [m]
INLIER_M = 1.0            # 旧データと「重なった」とみなす距離 [m]
GRID_M = 1.0              # global_align の格子 [m]
ROT_STEP = 3              # global_align の回転の刻み [度]
MIN_SEG = 5               # これより短い区間は前の区間に吸収する(1 秒)


def psi_and_class(yaw: np.ndarray) -> np.ndarray:
    idx = np.minimum(np.arange(len(yaw)) + H, len(yaw) - 1)
    psi = yaw[idx] - yaw
    return np.where(psi > THETA, 0, np.where(psi < -THETA, 1, 2))


def load_old(pkl_root: Path, prompts_path: Path):
    prompts = json.load(open(prompts_path))
    xy, yaw, cls, ins, ep = [], [], [], [], []
    dense = []
    for d in sorted(pkl_root.glob("episode*")):
        t = pickle.load(open(d / "traj_data.pkl", "rb"))
        p = np.asarray(t["position"], float)
        y = np.unwrap(np.asarray(t["yaw"], float))
        dense.append(p)
        pr = prompts.get(f"{pkl_root.name}/{d.name}")
        if pr is None:
            continue
        n = min(len(pr), len(p))
        c = psi_and_class(y[:n])
        for i in range(n):
            if pr[i].strip() and pr[i] != DUMMY:
                xy.append(p[i]); yaw.append(y[i]); cls.append(c[i]); ins.append(pr[i]); ep.append(d.name)
    return (np.array(xy), np.array(yaw), np.array(cls), ins, np.array(ep)), np.concatenate(dense)


def icp(src: np.ndarray, tree: cKDTree, dst: np.ndarray, R: np.ndarray, t: np.ndarray, iters: int = 60):
    for _ in range(iters):
        s = src @ R.T + t
        d, idx = tree.query(s)
        keep = d < np.percentile(d, 80)
        a, b = s[keep], dst[idx[keep]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rk = Vt.T @ U.T
        if np.linalg.det(Rk) < 0:
            Vt[1] *= -1
            Rk = Vt.T @ U.T
        R, t = Rk @ R, Rk @ t + mb - Rk @ ma
    d, _ = tree.query(src @ R.T + t)
    return R, t, d


def trimmed_icp(src: np.ndarray, tree: cKDTree, dst: np.ndarray, R: np.ndarray, t: np.ndarray,
                max_d: float = TRIM_M, iters: int = 40):
    """旧データの近く(max_d 以内)にある点だけで合わせる ICP。旧データが通っていない場所を
    走った点(ランダム走行に多い)に引っぱられないようにする。"""
    for _ in range(iters):
        s = src @ R.T + t
        d, idx = tree.query(s)
        keep = d < max_d
        if keep.sum() < 10:
            break
        a, b = s[keep], dst[idx[keep]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Rk = Vt.T @ U.T
        if np.linalg.det(Rk) < 0:
            Vt[1] *= -1
            Rk = Vt.T @ U.T
        R, t = Rk @ R, Rk @ t + mb - Rk @ ma
    d, _ = tree.query(src @ R.T + t)
    return R, t, d


def inlier(d: np.ndarray) -> float:
    return float(np.mean(d < INLIER_M))


def global_align(nxy: np.ndarray, dense: np.ndarray, tree: cKDTree):
    """出発地点も向きも分からない周回を旧データに合わせる。

    回転を ROT_STEP 度刻みで総当たりし、それぞれ旧データと新周回の軌跡を GRID_M 格子に塗って
    相互相関(FFT)が最大になる平行移動を求める。上位の候補を trimmed_icp で仕上げ、
    旧データと INLIER_M 以内で重なる点の割合が最大のものを返す。
    """
    lo = dense.min(0) - 50
    occ = np.zeros(np.ceil((dense.max(0) + 50 - lo) / GRID_M).astype(int) + 1)
    ij = ((dense - lo) / GRID_M).astype(int)
    occ[ij[:, 0], ij[:, 1]] = 1
    occ = np.maximum.reduce([np.roll(np.roll(occ, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    sub = nxy[:: max(1, len(nxy) // 3000)]
    cands = []
    for th in np.radians(np.arange(0, 360, ROT_STEP)):
        R0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        r = sub @ R0.T
        rlo = r.min(0)
        src = np.zeros(np.ceil((r.max(0) - rlo) / GRID_M).astype(int) + 1)
        k = ((r - rlo) / GRID_M).astype(int)
        src[k[:, 0], k[:, 1]] = 1
        shape = np.array(occ.shape) + np.array(src.shape)
        corr = np.fft.irfft2(np.fft.rfft2(occ, shape) * np.conj(np.fft.rfft2(src, shape)), shape)
        sh = np.unravel_index(np.argmax(corr), corr.shape)
        off = np.where(np.array(sh) > occ.shape, np.array(sh) - shape, np.array(sh))
        cands.append((corr[sh], R0, lo + off * GRID_M - rlo))
    cands.sort(key=lambda c: -c[0])
    best = None
    for _, R0, t0 in cands[:8]:
        R, t, d = trimmed_icp(nxy, tree, dense, R0, t0, max_d=5.0, iters=20)
        R, t, d = trimmed_icp(nxy, tree, dense, R, t)
        if best is None or inlier(d) > inlier(best[2]):
            best = (R, t, d)
    return best


def align(nxy: np.ndarray, dense: np.ndarray, ref: np.ndarray | None):
    """新の軌跡を旧の座標へ合わせる剛体変換。

    まず旧データ全体に恒等変換から合わせる。駄目なら(LIO の再起動で座標がずれた bag)、
    合わせ込み済みの別の周回 ref に回転の総当たりで合わせる。旧データは複数の経路が
    重なっているので総当たりの相手にすると局所解に落ちやすいが、同じ経路の周回なら形が一致する。
    """
    tree = cKDTree(dense)
    R, t, d = icp(nxy, tree, dense, np.eye(2), np.zeros(2))
    if np.median(d) < 0.5 or ref is None:
        return R, t, d
    rtree = cKDTree(ref)
    best = None
    for th in np.radians(np.arange(0, 360, 5)):
        R0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t0 = ref.mean(0) - (nxy @ R0.T).mean(0)
        R, t, d = icp(nxy, rtree, ref, R0, t0)
        if best is None or np.median(d) < np.median(best[2]):
            best = (R, t, d)
    R, t, _ = best
    R, t, d = icp(nxy, tree, dense, R, t, iters=10)
    # 旧データとの重なりが少なければ(出発地点も向きも違うランダム走行)、全域から探し直す
    if inlier(d) < 0.5:
        g = global_align(nxy, dense, tree)
        if inlier(g[2]) > inlier(d):
            return g
    return R, t, d


def assign(nxy, nyaw, ncls, old, used):
    oxy, oyaw, ocls, oins, oep = old
    tree = cKDTree(oxy)
    utree = cKDTree(np.array([u[0] for u in used])) if used else None
    out, cur, miss = [], None, 0
    for i in range(len(nxy)):
        idx = tree.query_ball_point(nxy[i], RADIUS_M)
        support = defaultdict(set)
        for j in idx:
            if ocls[j] == ncls[i] and abs((nyaw[i] - oyaw[j] + np.pi) % (2 * np.pi) - np.pi) < MAX_DYAW:
                support[oins[j]].add(oep[j])
        near_used = Counter()
        if utree is not None:
            near_used = Counter(used[k][1] for k in utree.query_ball_point(nxy[i], DIVERSITY_M))
        # 今の指示を前の bag が近くで使っていて、使われていない候補があるなら乗り換える
        fresh = [s for s in support if near_used[s] == 0]
        if cur is not None and (cur in support or miss < HOLD_GAP) and not (near_used[cur] and fresh):
            miss = 0 if cur in support else miss + 1
            out.append(cur)
            continue
        miss = 0
        if not support:
            cur = None
            out.append(None)
            continue
        cur = max(support, key=lambda s: (-near_used[s], len(support[s])))
        out.append(cur)
    # 短すぎる区間を前の区間に吸収
    segs = []
    for i, s in enumerate(out):
        if segs and segs[-1][2] == s:
            segs[-1][1] = i
        else:
            segs.append([i, i, s])
    for k in range(1, len(segs)):
        if segs[k][1] - segs[k][0] + 1 < MIN_SEG and segs[k][2] is not None:
            for i in range(segs[k][0], segs[k][1] + 1):
                out[i] = out[segs[k][0] - 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laps", nargs="+", type=Path, help="bag_to_lap_episode.py の出力 <out_root>/<bag名>")
    ap.add_argument("--old-pkl-root", type=Path, required=True)
    ap.add_argument("--old-prompts", type=Path, required=True)
    ap.add_argument("--fixed", nargs="*", type=Path, default=[],
                    help="注釈済みの周回。書き換えず，今の traj_prompt.txt を「前に処理した bag」として扱う")
    ap.add_argument("--write", action="store_true", help="traj_prompt.txt を書く")
    args = ap.parse_args()

    old, dense = load_old(args.old_pkl_root.expanduser(), args.old_prompts.expanduser())
    print(f"旧データ: {len(old[3])} フレーム / 指示 {len(set(old[3]))} 種")
    used: list[tuple[np.ndarray, str]] = []
    per_bag = {}
    for lap in [*args.fixed, *args.laps]:
        fixed = lap in args.fixed
        ep_dir = lap.expanduser() / "episode01"
        t = pickle.load(open(ep_dir / "traj_data.pkl", "rb"))
        nxy = np.asarray(t["position"], float)
        nyaw = np.asarray(t["yaw"], float)
        ref = np.concatenate([v[0] for v in per_bag.values()]) if per_bag else None
        R, tr, d = align(nxy, dense, ref)
        rot = np.arctan2(R[1, 0], R[0, 0])
        axy, ayaw = nxy @ R.T + tr, nyaw + rot
        if fixed:
            prompts = [None if x == DUMMY else x
                       for x in (ep_dir / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")]
        else:
            prompts = assign(axy, ayaw, psi_and_class(nyaw), old, used)
        used += [(axy[i], s) for i, s in enumerate(prompts) if s is not None and i % 5 == 0]
        n_dummy = sum(s is None for s in prompts)
        n_seg = sum(1 for i, s in enumerate(prompts) if i == 0 or s != prompts[i - 1])
        per_bag[lap.name] = (axy, prompts)
        print(f"{lap.name}{'（固定）' if fixed else ''}: 合わせ込み 回転 {np.degrees(rot):.1f}° 残差中央 {np.median(d):.2f} m | "
              f"区間 {n_seg} / 指示 {len(set(prompts) - {None})} 種 / 未割当 {n_dummy} フレーム ({n_dummy / len(prompts) * 100:.1f}%)")
        if args.write and not fixed:
            (ep_dir / "traj_prompt.txt").write_text("\n".join(s or DUMMY for s in prompts), encoding="utf-8")

    # bag 間で同じ場所の指示がどれだけ重なったか
    names = list(per_bag)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            xa, pa = per_bag[names[a]]
            xb, pb = per_bag[names[b]]
            _, j = cKDTree(xb).query(xa)
            both = [(pa[i], pb[j[i]]) for i in range(len(pa)) if pa[i] and pb[j[i]]]
            same = np.mean([x == y for x, y in both]) * 100 if both else float("nan")
            print(f"  同じ場所で同じ指示: {names[a]} と {names[b]} = {same:.1f}%")


if __name__ == "__main__":
    main()
