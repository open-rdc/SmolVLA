#!/usr/bin/env python3
"""収録したエピソードの破損チェックと、学習に使えないエピソードの選別。

create_data.py が吐いた <dataset>/episodeNN/ を検査する。判定に使う dx_body は
lerobot_dataset.py と同じ式（body系1ステップ増分の前進成分）で計算するので、
「変換したらどうなるか」と一致する。

    # 検査してレポートするだけ（既定・何も変更しない）
    python training/data/check_episodes.py <dataset_dir> [<dataset_dir> ...]

    # 弾く対象を実際に退避する（--move-to 配下へ移動。削除はしない）
    python training/data/check_episodes.py <dataset_dir> --move-to <dataset_dir>/_rejected

除外条件（閾値は引数で変更可）:
  short    : フレーム数が --min-frames 未満
  backward : dx_body < -stop_eps のフレーム比率が --max-backward 超
  stopped  : |dx_body| <= stop_eps のフレーム比率が --max-stopped 超
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_H, IMG_W = 224, 224
FPS = 5
DT = 1.0 / FPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", type=Path, help="episodeNN/ を含むデータセットディレクトリ")
    p.add_argument("--min-frames", type=int, default=25,
                   help="これ未満のフレーム数のエピソードは short として弾く")
    p.add_argument("--stop-eps", type=float, default=0.02,
                   help="|dx_body| がこれ以下なら停止フレームとみなす[m/step] (0.02m/step = 0.1m/s)")
    p.add_argument("--max-stopped", type=float, default=0.30, help="停止フレーム比率の上限")
    p.add_argument("--max-backward", type=float, default=0.10, help="後退フレーム比率の上限")
    p.add_argument("--teleport", type=float, default=1.0,
                   help="1ステップの移動量がこれを超えたらオドメトリ異常[m]")
    p.add_argument("--move-to", type=Path, default=None,
                   help="指定すると弾いたエピソードをここへ移動する（未指定ならレポートのみ）")
    p.add_argument("--skip-jpeg", action="store_true", help="jpg のデコード検査を省く（高速）")
    return p.parse_args()


def to_body_frame(delta_xy_global: np.ndarray, yaw: float) -> np.ndarray:
    """lerobot_dataset.py と同一実装（world -> body の回転）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return delta_xy_global.dot(np.array([[c, -s], [s, c]]))


def check_episode(ep: Path, args: argparse.Namespace) -> dict:
    """1エピソードを検査して結果 dict を返す。errors が空なら破損なし。"""
    r: dict = {"dir": ep, "errors": [], "warns": [], "reasons": []}

    # --- traj_data.pkl ---
    pkl = ep / "traj_data.pkl"
    if not pkl.exists():
        r["errors"].append("traj_data.pkl が無い")
        return r
    try:
        with pkl.open("rb") as f:
            d = pickle.load(f)
    except Exception as e:                                    # noqa: BLE001
        r["errors"].append(f"traj_data.pkl が読めない: {e}")
        return r

    for k in ("position", "yaw"):
        if k not in d:
            r["errors"].append(f"traj_data.pkl に '{k}' が無い")
    if r["errors"]:
        return r

    pos = np.asarray(d["position"], dtype=np.float64)
    yaw = np.asarray(d["yaw"], dtype=np.float64)
    n = len(pos)
    r["n"] = n

    if pos.ndim != 2 or pos.shape[1] != 2:
        r["errors"].append(f"position の形が変: {pos.shape}")
    if yaw.ndim != 1:
        r["errors"].append(f"yaw の形が変: {yaw.shape}")
    if len(yaw) != n:
        r["errors"].append(f"position({n}) と yaw({len(yaw)}) の長さが違う")
    if not np.isfinite(pos).all() or not np.isfinite(yaw).all():
        r["errors"].append("position/yaw に NaN か inf がある")
    if r["errors"]:
        return r

    # --- jpg の枚数と連番 ---
    jpgs = {int(p.stem): p for p in ep.glob("*.jpg") if p.stem.isdigit()}
    if len(jpgs) != n:
        r["errors"].append(f"jpg {len(jpgs)}枚 と フレーム数 {n} が一致しない")
    missing = [i for i in range(n) if i not in jpgs]
    if missing:
        r["errors"].append(f"jpg が欠番: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # --- jpg が読めるか・サイズ ---
    if not args.skip_jpeg and not missing:
        bad = []
        for i in range(n):
            im = cv2.imread(str(jpgs[i]))
            if im is None:
                bad.append(f"{i}(読めない)")
            elif im.shape != (IMG_H, IMG_W, 3):
                bad.append(f"{i}{im.shape}")
        if bad:
            r["errors"].append(f"jpg が壊れている/サイズ不正: {bad[:5]}{'...' if len(bad) > 5 else ''}")

    if n < 2:
        r["errors"].append(f"フレームが {n} 枚しか無い（action を作れない）")
        return r

    # --- yaw が unwrap されているか ---
    dyaw = np.diff(yaw)
    if np.any(np.abs(dyaw) > np.pi):
        r["errors"].append(f"yaw に 2pi の飛びがある（unwrap されていない）: max|dyaw|={np.abs(dyaw).max():.2f}")

    # --- 動きの統計（変換器と同じ dx_body）---
    dxy = np.diff(pos, axis=0)                                     # (n-1, 2) world
    dx_body = np.array([to_body_frame(dxy[t], yaw[t])[0] for t in range(n - 1)])
    step_len = np.hypot(dxy[:, 0], dxy[:, 1])

    if np.any(step_len > args.teleport):
        r["errors"].append(
            f"オドメトリが飛んでいる: max移動量 {step_len.max():.2f} m/step (>{args.teleport})")

    r["dx_mean"] = float(dx_body.mean())
    r["dx_med"] = float(np.median(dx_body))
    r["path_len"] = float(step_len.sum())
    r["net"] = float(np.hypot(*(pos[-1] - pos[0])))
    r["back_frac"] = float((dx_body < -args.stop_eps).mean())
    r["stop_frac"] = float((np.abs(dx_body) <= args.stop_eps).mean())
    r["dyaw_absmean"] = float(np.abs(dyaw).mean())

    # --- 除外判定 ---
    if n < args.min_frames:
        r["reasons"].append(f"short({n}f)")
    if r["back_frac"] > args.max_backward:
        r["reasons"].append(f"backward({r['back_frac']:.0%})")
    if r["stop_frac"] > args.max_stopped:
        r["reasons"].append(f"stopped({r['stop_frac']:.0%})")
    return r


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []

    for ds in args.datasets:
        eps = sorted((p for p in ds.iterdir() if p.is_dir() and not p.name.startswith("_")),
                     key=lambda p: p.name)
        print(f"\n{'=' * 78}\n{ds}  ({len(eps)} episodes)\n{'=' * 78}")
        rows = [check_episode(ep, args) for ep in eps]
        all_rows += rows

        broken = [r for r in rows if r["errors"]]
        if broken:
            print(f"\n■ 破損 {len(broken)} 件")
            for r in broken:
                print(f"  {r['dir'].name}: " + " / ".join(r["errors"]))
        else:
            print("\n■ 破損: なし")

        ok = [r for r in rows if not r["errors"]]
        if ok:
            n = np.array([r["n"] for r in ok])
            print(f"\n■ フレーム数    min={n.min()} med={int(np.median(n))} max={n.max()} 合計={n.sum()}")
            for name, key, unit in [("経路長", "path_len", "m"), ("正味移動", "net", "m"),
                                    ("dx_body平均", "dx_mean", "m/step"),
                                    ("後退フレーム率", "back_frac", ""), ("停止フレーム率", "stop_frac", "")]:
                v = np.array([r[key] for r in ok])
                print(f"  {name:<14} min={v.min():7.3f} med={np.median(v):7.3f} "
                      f"max={v.max():7.3f} {unit}")

        rejected = [r for r in rows if not r["errors"] and r["reasons"]]
        print(f"\n■ 除外対象 {len(rejected)} / {len(ok)} 件"
              f"（min_frames<{args.min_frames}, backward>{args.max_backward:.0%}, "
              f"stopped>{args.max_stopped:.0%}）")
        for r in rejected:
            print(f"  {r['dir'].name:<12} n={r['n']:>3} 経路長={r['path_len']:6.2f}m "
                  f"dx平均={r['dx_mean']:+.3f} 後退={r['back_frac']:.0%} 停止={r['stop_frac']:.0%}"
                  f"  -> {', '.join(r['reasons'])}")

        keep = [r for r in rows if not r["errors"] and not r["reasons"]]
        print(f"\n■ 採用 {len(keep)} 件 / 合計 {sum(r['n'] for r in keep)} フレーム")

    # --- 退避（--move-to 指定時のみ）---
    if args.move_to is not None:
        targets = [r for r in all_rows if r["errors"] or r["reasons"]]
        args.move_to.mkdir(parents=True, exist_ok=True)
        for r in targets:
            dst = args.move_to / f"{r['dir'].parent.name}__{r['dir'].name}"
            shutil.move(str(r["dir"]), str(dst))
        print(f"\n{len(targets)} 件を {args.move_to} へ退避しました（削除はしていません）")
    else:
        print("\n（レポートのみ。実際に弾くには --move-to <退避先> を付けてください）")


if __name__ == "__main__":
    main()
