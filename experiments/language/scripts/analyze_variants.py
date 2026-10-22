"""目印だけ/方向だけプローブ(probe_branch_variants.py の出力 npz)の集計。
使い方: python3 analyze_variants.py variants_omni_e22.npz variants_smol_e14.npz
方向が合う指示(orig の方向 = 行き先)のフレームだけ使う。"""
import sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec(open(HERE / "check_prompt_motion.py").read().split("def motion_dir")[0])
br = np.load(HERE.parent / "data/fisheye_tsudanuma_v2/branch_134655_ref.npz")
for path in sys.argv[1:]:
    d = np.load(path); t = d["t"]
    ref = np.stack([br["psi_ref_" + c][t] for c in "LSR"], 1)
    ok = np.ones(len(t), bool)
    for i in range(len(t)):
        r = ref[i][~np.isnan(ref[i])]
        dd = np.abs(r[:, None] - r[None]); dd[np.eye(len(r), dtype=bool)] = 9
        ok[i] = len(r) >= 2 and dd.min() >= 0.09
    def pick(psi):
        dist = np.abs(ref - psi[:, None]); dist[np.isnan(ref)] = np.inf
        return np.array(list("LSR"))[dist.argmin(1)]
    cons = {c: np.array([prompt_dir(str(p)) == c for p in d["text_orig_" + c]]) for c in "LSR"}
    print(path.split("/")[-1])
    print("  行き先の正解率(方向が合う指示のフレーム、括弧内フレーム数)")
    for k, name in [("orig", "元の文"), ("lm", "目印だけ"), ("dir", "方向だけ")]:
        row = []
        for c in "LSR":
            psi = d[f"psi_{k}_{c}"]
            m = ok & cons[c] & ~np.isnan(psi) & ~np.isnan(ref[:, "LSR".index(c)])
            row.append(f"{c}:{np.mean(pick(psi)[m] == c):.0%}({m.sum()})" if m.any() else f"{c}:-")
        print(f"    {name:6s} " + "  ".join(row))
    m = ok & cons["L"] & cons["R"]
    print(f"  左と右の指示を差し替えたときの予測Ψの差(左右とも方向が合う {m.sum()} フレーム, 平均 |ΔΨ| 度)")
    for k, name in [("orig", "元の文"), ("lm", "目印だけ"), ("dir", "方向だけ")]:
        a, b = d[f"psi_{k}_L"][m], d[f"psi_{k}_R"][m]
        v = ~np.isnan(a) & ~np.isnan(b)
        print(f"    {name:6s} {np.degrees(np.mean(np.abs(a[v] - b[v]))):5.1f}° (n={v.sum()})  左の方が左向き {np.mean(a[v] > b[v]):.0%}")


# ---- 行き先の違う 2 つの指示の組すべて(左↔右, 左↔直進, 直進↔右)で、分岐の場所ごとにまとめる
def pair_summary(path):
    d = np.load(path); t = d["t"]
    ref = {c: br["psi_ref_" + c][t] for c in "LSR"}
    cons = {c: np.array([prompt_dir(str(p)) == c for p in d["text_orig_" + c]]) for c in "LSR"}
    loc = np.concatenate([[0], np.cumsum(np.diff(t) > 1)])  # 連続したフレーム = 1 つの分岐の場所
    rows = []  # (場所, 組, 条件, |ΔΨ|, 向きが正しいか)
    for a, b in [("L", "R"), ("L", "S"), ("S", "R")]:
        m = ~np.isnan(ref[a]) & ~np.isnan(ref[b]) & cons[a] & cons[b] & (np.abs(ref[a] - ref[b]) >= 0.09)
        for k in ("orig", "lm", "dir"):
            pa, pb = d[f"psi_{k}_{a}"], d[f"psi_{k}_{b}"]
            for i in np.where(m & ~np.isnan(pa) & ~np.isnan(pb))[0]:
                rows.append((loc[i], a + b, k, abs(pa[i] - pb[i]), pa[i] > pb[i]))  # 左ほど Ψ が大きい
    return rows


for path in sys.argv[1:]:
    rows = pair_summary(path)
    print(path.split("/")[-1], "— 行き先の違う指示の組(左↔右・左↔直進・直進↔右)")
    for k, name in [("orig", "元の文"), ("lm", "目印だけ"), ("dir", "方向だけ")]:
        r = [x for x in rows if x[2] == k]
        locs = sorted({x[0] for x in r})
        per = [(np.mean([x[3] for x in r if x[0] == l]), np.mean([x[4] for x in r if x[0] == l])) for l in locs]
        dpsi = np.degrees([p[0] for p in per]); right = np.array([p[1] for p in per])
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(dpsi, len(dpsi))) for _ in range(2000)]
        print(f"  {name:6s} フレーム {len(r):3d} / 場所 {len(locs):2d} | 場所ごと平均 |ΔΨ| {dpsi.mean():5.1f}° "
              f"(95%区間 {np.percentile(bs, 2.5):4.1f}–{np.percentile(bs, 97.5):4.1f}°) | 向きが正しい場所 {np.mean(right > 0.5):.0%} "
              f"| フレーム単位で向きが正しい {np.mean([x[4] for x in r]):.0%}")
    by = {}
    for x in rows:
        if x[2] == "orig":
            by.setdefault(x[1], set()).add(x[0])
    print("  組ごとの場所数:", {k: len(v) for k, v in by.items()})
