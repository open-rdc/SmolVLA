"""検証用周回の全フレームで、指示を差し替えたときの予測 Ψ(4.8 秒先までの向き変化)を出す。

分岐で指示が効いているかを地図で見るためのもの。条件は
  orig = 元の指示 / mirror = left<->right を入れ替えた指示(左右の語があるフレームだけ)
  left = "Turn left." / right = "Turn right." / straight = "Go straight." / noinst = 指示なし
Ψ(left) - Ψ(right) が大きい場所ほど、指示で行き先を変えられる。
--branch(find_branch_points.py の出力)を渡すと分岐フレームだけを見て、その方向へ進んだ学習用周回が
同じ場所で使っていた指示文(prompt_ref_{L,S,R})も train_L / train_S / train_R として入れる。

フレームの選び方・正解・推論は eval_monitor.py と同じ(OmniVLAEval / SmolVLAEval をそのまま使う)。

    python training/probe_language_map.py --model omnivla \\
        --weights ~/NavVLA/training/runs/fisheye_tsudanuma_v2_omnivla/model_epoch_022.pth \\
        --out probe_omnivla_e22.npz
    python training/probe_language_map.py --model smolvla --stride 3 \\
        --weights ~/SmolVLA/outputs/train/smolvla_fisheye_tsudanuma_v2/checkpoints/<step>/pretrained_model \\
        --out probe_smolvla.npz
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_monitor as em  # noqa: E402

HOME = Path.home()
PROBES = {"left": "Turn left.", "right": "Turn right.", "straight": "Go straight.", "noinst": ""}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["omnivla", "smolvla"], required=True)
ap.add_argument("--weights", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--stride", type=int, default=1, help="何フレームおきに見るか(SmolVLA は画像を全部メモリに載せるので 3 程度)")
ap.add_argument("--lap", default="tsudanuma_20261010_134655")
ap.add_argument("--raw-pkl", type=Path, default=HOME / "navvla_data/fisheye_tsudanuma_v2")
ap.add_argument("--val-root", type=Path, default=HOME / "smolvla_data/fisheye_tsudanuma_v2_test")
ap.add_argument("--val-repo-id", default="open-rdc/fisheye_tsudanuma_v2_test")
ap.add_argument("--omni-config", default="fisheye_tsudanuma_v2")
ap.add_argument("--omni-dataset", default="fisheye_tsudanuma_v2")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--all-seeds", action="store_true", help="orig 以外の条件も --seeds 本で平均する")
ap.add_argument("--branch", type=Path, help="find_branch_points.py の出力。分岐フレームだけを見て学習時の指示文も入れる")
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

d = pickle.load(open(args.raw_pkl / args.lap / "traj_data.pkl", "rb"))
pos = np.asarray(d["position"], dtype=np.float64)
yaw = np.unwrap(np.asarray(d["yaw"], dtype=np.float64))
pr = [s.strip() for s in (args.raw_pkl / args.lap / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")]
L = len(pr) - 1  # LeRobot 側は最終フレームを落としている
ts = list(range(0, L - em.H, args.stride))
br = np.load(args.branch) if args.branch else None
if br is not None:
    ts = [t for t in ts if br["branch"][t]]
frames = {"real": [dict(ep=0, name=args.lap, t=t, L=L, gidx=t, task=pr[t], psi=float(yaw[t + em.H] - yaw[t]),
                        pts=em.local(pos, yaw, t, [t + em.SPACING * k for k in range(1, em.H // em.SPACING + 1)]).tolist())
                   for t in ts], "cosmos": []}
conds = []
for i, f in enumerate(frames["real"]):
    conds.append(("orig", i, f["task"]))
    if em.MIRROR.search(f["task"]):
        conds.append(("mirror", i, em.mirror(f["task"])))
    for k, s in PROBES.items():
        conds.append((k, i, s))
    if br is not None:
        for c in "LSR":
            if br[f"prompt_ref_{c}"][f["t"]]:
                conds.append((f"train_{c}", i, str(br[f"prompt_ref_{c}"][f["t"]])))
print(f"{args.model}: {len(ts)} フレーム / {len(conds)} 条件", flush=True)

ev = (em.OmniVLAEval if args.model == "omnivla" else em.SmolVLAEval)(args, frames)
preds, _ = ev.predict(args.weights, conds)
out = {"t": np.array(ts), "pos": pos[ts], "yaw": yaw[ts], "psi_true": np.array([f["psi"] for f in frames["real"]]),
       "task": np.array([f["task"] for f in frames["real"]])}
if br is not None:
    for c in "LSR":
        out[f"prompt_ref_{c}"] = br[f"prompt_ref_{c}"][ts]
for c in ["orig", "mirror", *PROBES, "train_L", "train_S", "train_R"]:
    out["psi_" + c] = np.array([preds.get(c, {}).get(i, (np.nan,))[0] for i in range(len(ts))])
np.savez(args.out, **out)
gap = out["psi_left"] - out["psi_right"]
print(f"Ψ(left)-Ψ(right): 中央値 {np.median(gap):.3f} / 90%点 {np.percentile(gap, 90):.3f} / 最大 {gap.max():.3f} rad", flush=True)
