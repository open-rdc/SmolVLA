#!/usr/bin/env python3
"""周回エピソードの指示を，実際の動き(直進/右折/左折)の区間ごとに 1 つにまとめ直す。

自動付与(transfer_prompts_from_old.py)の指示は似た英文へ数フレームごとに切り替わり細切れになる。
lap_annotation_web.py と同じ Ψ(4.8 秒先までの向き変化)で各フレームを L/R/S に分け，
min_len 未満の短い区間は長い方の隣に吸収してから，区間ごとに
  - ダミーが区間の半分を超える → 区間全体をダミー
  - それ以外 → 方向が区間の動きと合う指示のうち一番多いものを区間全体に(合うものが無ければダミー)
とする。確認済みのフレーム(review_state.json)は触らず，未確認の連続部分ごとに処理する。

    python training/data/resegment_by_motion.py ~/tsudanuma_data/laps --classes ~/tsudanuma_data/instructions_cls.json
    → 件数を見て問題なければ --write(元は traj_prompt.before_motionseg.txt.bak に残す)
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

DUMMY = "No language instruction"
H = 24          # lap_annotation_web.py と同じ
THETA = 0.090


def motion_cls(d: Path, n: int) -> list[str]:
    traj = pickle.load(open(d / "traj_data.pkl", "rb"))
    yaw = np.unwrap(np.asarray(traj["yaw"], dtype=float)[:n])
    psi = np.array([yaw[min(t + H, n - 1)] - yaw[t] for t in range(n)])
    return np.where(psi > THETA, "L", np.where(psi < -THETA, "R", "S")).tolist()


def runs_of(seq) -> list[list]:
    """[[値, 始点, 終点(含まない)], ...]"""
    out = []
    for i, v in enumerate(seq):
        if out and out[-1][0] == v:
            out[-1][2] = i + 1
        else:
            out.append([v, i, i + 1])
    return out


def merge_short(runs: list[list], min_len: int) -> list[list]:
    """一番短い区間から順に，長い方の隣へ吸収する"""
    runs = [r[:] for r in runs]
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda j: runs[j][2] - runs[j][1])
        if runs[k][2] - runs[k][1] >= min_len:
            break
        nb = [j for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: runs[j][2] - runs[j][1])
        runs[j][1], runs[j][2] = min(runs[j][1], runs[k][1]), max(runs[j][2], runs[k][2])
        del runs[k]
        # 吸収で同じ動きの区間が隣り合ったらつなぐ
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        runs = merged
    return runs


def matches(icls: dict, s: str, c: str) -> bool:
    k = icls.get(s)
    if k is None:
        return True          # 分類が無い指示は方向で弾かない
    return k == c or (k == "U" and c in "LR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laps_root", type=Path)
    ap.add_argument("--classes", type=Path, required=True, help="指示の動作分類 JSON {英語: L/R/S/U}")
    ap.add_argument("--min-len", type=int, default=15, help="これより短い動きの区間は隣に吸収 [フレーム](5 Hz で 3 秒)")
    ap.add_argument("--only", nargs="*", help="この文字列を名前に含む周回だけ処理する")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    icls = {k.strip(): v for k, v in json.loads(args.classes.read_text(encoding="utf-8")).items()}

    for lap in sorted(p for p in args.laps_root.iterdir() if (p / "episode01" / "traj_prompt.txt").exists()):
        if args.only and not any(o in lap.name for o in args.only):
            continue
        d = lap / "episode01"
        pr = (d / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")
        n = len(pr)
        rs = d / "review_state.json"
        rv = json.loads(rs.read_text())["reviewed"] if rs.exists() else [0] * n
        mc = motion_cls(d, n)
        segs = []
        for v, a0, b0 in runs_of([bool(x) for x in rv]):
            if v:
                segs.append((None, a0, b0))      # 確認済みはそのまま
            else:
                segs += [(c, a0 + a, a0 + b) for c, a, b in merge_short(runs_of(mc[a0:b0]), args.min_len)]
        new = []
        for c, a, b in segs:
            if c is None:
                new += pr[a:b]
                continue
            cnt = Counter(pr[a:b])
            if cnt[DUMMY] * 2 > b - a:
                t = DUMMY
            else:
                ok = [(m, s) for s, m in cnt.items() if s != DUMMY and matches(icls, s, c)]
                t = max(ok)[1] if ok else DUMMY
            new += [t] * (b - a)
        before = len(runs_of(pr))
        print(f"{lap.name}: 区間 {before} → {len(runs_of(new))}  ダミー {pr.count(DUMMY)} → {new.count(DUMMY)}"
              f"  変わったフレーム {sum(x != y for x, y in zip(pr, new))}/{n}")
        if args.write:
            bak = d / "traj_prompt.before_motionseg.txt.bak"
            if not bak.exists():
                bak.write_text("\n".join(pr), encoding="utf-8")
            (d / "traj_prompt.txt").write_text("\n".join(new), encoding="utf-8")


if __name__ == "__main__":
    main()
