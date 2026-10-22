"""分岐フレームで、学習時の指示文を「目印だけ」「方向だけ」に作り替えて予測 Ψ を出す。

probe_language_map.py の --branch と同じフレーム・同じ推論(eval_monitor.py の OmniVLAEval / SmolVLAEval)。
条件: {orig,lm,dir}_{L,S,R}。orig = その方向へ行った周回が使っていた指示文、lm = 目印だけ、dir = 方向だけ(variants.py)。
    python probe/probe_branch_variants.py --model omnivla --weights <pth> --out probe/variants_omni_e22.npz
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "SmolVLA/training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_monitor as em  # noqa: E402
from variants import direction_only, landmark_only  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["omnivla", "smolvla"], required=True)
ap.add_argument("--weights", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--branch", type=Path, default=HOME / "SmolVLA/probe/branch_134655_ref.npz")
ap.add_argument("--lap", default="tsudanuma_20261010_134655")
ap.add_argument("--raw-pkl", type=Path, default=HOME / "navvla_data/fisheye_tsudanuma_v2")
ap.add_argument("--val-root", type=Path, default=HOME / "smolvla_data/fisheye_tsudanuma_v2_test")
ap.add_argument("--val-repo-id", default="open-rdc/fisheye_tsudanuma_v2_test")
ap.add_argument("--omni-config", default="fisheye_tsudanuma_v2")
ap.add_argument("--omni-dataset", default="fisheye_tsudanuma_v2")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--all-seeds", action="store_true", default=True)
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

d = pickle.load(open(args.raw_pkl / args.lap / "traj_data.pkl", "rb"))
pos = np.asarray(d["position"], dtype=np.float64)
yaw = np.unwrap(np.asarray(d["yaw"], dtype=np.float64))
pr = [s.strip() for s in (args.raw_pkl / args.lap / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")]
L = len(pr) - 1
br = np.load(args.branch)
ts = [t for t in range(0, L - em.H) if br["branch"][t]]
frames = {"real": [dict(ep=0, name=args.lap, t=t, L=L, gidx=t, task=pr[t], psi=float(yaw[t + em.H] - yaw[t]),
                        pts=em.local(pos, yaw, t, [t + em.SPACING * k for k in range(1, em.H // em.SPACING + 1)]).tolist())
                   for t in ts], "cosmos": []}
make = {"orig": lambda s: s, "lm": landmark_only, "dir": direction_only}
conds, texts = [], {}
for i, f in enumerate(frames["real"]):
    for c in "LSR":
        p = str(br[f"prompt_ref_{c}"][f["t"]])
        if not p:
            continue
        for k, fn in make.items():
            s = fn(p)
            if s:
                conds.append((f"{k}_{c}", i, s))
                texts[(f"{k}_{c}", i)] = s
print(f"{args.model}: {len(ts)} フレーム / {len(conds)} 条件", flush=True)
ev = (em.OmniVLAEval if args.model == "omnivla" else em.SmolVLAEval)(args, frames)
preds, _ = ev.predict(args.weights, conds)
out = {"t": np.array(ts)}
for k in make:
    for c in "LSR":
        name = f"{k}_{c}"
        out["psi_" + name] = np.array([preds.get(name, {}).get(i, (np.nan,))[0] for i in range(len(ts))])
        out["text_" + name] = np.array([texts.get((name, i), "") for i in range(len(ts))])
np.savez(args.out, **out)
print("saved", args.out, flush=True)
