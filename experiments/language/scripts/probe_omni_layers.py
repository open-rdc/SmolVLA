"""評価案4(第2段階): OmniVLA-edge の中で、指示文の違い(特に left/right)がどの層で小さくなるかを測る。

検証用周回の分岐フレームで、同じ画像に指示文だけを差し替えて入れ、途中の出力を取り出す。
  film_p : FiLM の係数(γ, β)。指示文だけで決まる
  film   : FiLM で画像を変調した後の特徴(画像+指示文)
  lan    : それを圧縮した言語トークン(Transformer に入る)
  final  : Transformer の出力(行動を出す直前)
  psi    : 予測 Ψ(4.8 秒後の向きの変化)
指示文の条件: orig(元) / mirror(left<->right) / train_L, train_S, train_R(その分岐で各方向へ行った周回の指示文) /
              rand1, rand2(周回中の無関係な指示文)
各層で距離の平均を出す:
  D_mirror = |h(orig) - h(mirror)|, D_LR = |h(train_L) - h(train_R)|, D_rand = |h(rand1) - h(rand2)|(同じ画像)
  D_img = |h(フレーム i, orig) - h(フレーム j, orig)|(画像も指示も違う)
eval_monitor.py の OmniVLAEval でサンプルを作り、推論だけ自前で行って forward hook で取り出す。
    python probe_omni_layers.py --weights <model.pth> --branch probe/branch_134655_ref.npz --out probe/layers_e22.npz
"""
import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
sys.path.insert(0, str(HOME / "SmolVLA/training"))
import eval_monitor as em  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=Path, required=True)
ap.add_argument("--branch", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--lap", default="tsudanuma_20261010_134655")
ap.add_argument("--raw-pkl", type=Path, default=HOME / "navvla_data/fisheye_tsudanuma_v2")
ap.add_argument("--omni-config", default="fisheye_tsudanuma_v2")
ap.add_argument("--omni-dataset", default="fisheye_tsudanuma_v2")
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
pool = sorted({p for p in pr if p})
rng = random.Random(0)
conds = []
for i, f in enumerate(frames["real"]):
    conds.append(("orig", i, f["task"]))
    if em.MIRROR.search(f["task"]):
        conds.append(("mirror", i, em.mirror(f["task"])))
    for c in "LSR":
        if br[f"prompt_ref_{c}"][f["t"]]:
            conds.append((f"train_{c}", i, str(br[f"prompt_ref_{c}"][f["t"]])))
    r1, r2 = rng.sample([p for p in pool if p != f["task"]], 2)
    conds += [("rand1", i, r1), ("rand2", i, r2)]
print(f"{len(ts)} フレーム / {len(conds)} 条件", flush=True)

ev = em.OmniVLAEval(args, frames)
from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge  # noqa: E402
nc = ev.nc
model = OmniVLA_edge(
    context_size=nc["context_size"], len_traj_pred=nc["len_traj_pred"], learn_angle=nc["learn_angle"],
    obs_encoder=nc["obs_encoder"], obs_encoding_size=nc["obs_encoding_size"],
    late_fusion=nc["late_fusion"], mha_num_attention_heads=nc["mha_num_attention_heads"],
    mha_num_attention_layers=nc["mha_num_attention_layers"], mha_ff_dim_factor=nc["mha_ff_dim_factor"])
sd = torch.load(args.weights, map_location="cpu")
model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd)
model = model.to("cuda").eval()

grab = {}
def hook(name):
    def fn(_m, _i, o):
        grab[name] = (o[0] if isinstance(o, tuple) else o).detach().float().flatten(start_dim=1).cpu()
    return fn
model.film_model.film_param_generator.register_forward_hook(hook("film_p"))
model.film_model.register_forward_hook(hook("film"))
model.compress_goal_enc_lan.register_forward_hook(hook("lan"))
model.decoder.register_forward_hook(hook("final"))

LAYERS = ["film_p", "film", "lan", "final", "psi"]
H = {k: {} for k in LAYERS}
for s0 in range(0, len(conds), args.batch_size):
    sub = conds[s0:s0 + args.batch_size]
    batch = {k: torch.stack([ev.samples[("real", c[1])][k] for c in sub]).to("cuda") for k in ev.samples[("real", 0)]}
    batch["feat_text"] = torch.stack([ev.ds.encode_text(c[2] or em.DUMMY) for c in sub]).to("cuda")
    with torch.no_grad():
        a, _, _ = model(batch["obs_images"], batch["goal_pose"].float(), batch["map_images"], batch["goal_image"],
                        batch["goal_mask"].long(), batch["feat_text"].float(), batch["current_img"])
    a = a.float().cpu()
    psi = torch.tensor([[math.atan2(ai[-1, 3], ai[-1, 2])] for ai in a])
    for j, c in enumerate(sub):
        for k in LAYERS[:-1]:
            H[k][(c[0], c[1])] = grab[k][j]
        H["psi"][(c[0], c[1])] = psi[j]

n = len(ts)
pairs_img = [tuple(rng.sample(range(n), 2)) for _ in range(2000)]
def same_img(k, ca, cb):
    v = [(H[k][(ca, i)] - H[k][(cb, i)]).norm().item() for i in range(n) if (ca, i) in H[k] and (cb, i) in H[k]]
    return float(np.mean(v)) if v else float("nan"), len(v)
out = {"t": np.array(ts)}
print("layer\tD_mirror\tD_LR\tD_rand\tD_img\tmirror/rand\tLR/rand\trand/img", flush=True)
for k in LAYERS:
    dm, nm = same_img(k, "orig", "mirror")
    dlr, nlr = same_img(k, "train_L", "train_R")
    dr, _ = same_img(k, "rand1", "rand2")
    di = float(np.mean([(H[k][("orig", i)] - H[k][("orig", j)]).norm().item() for i, j in pairs_img]))
    print(f"{k}\t{dm:.4g}\t{dlr:.4g}\t{dr:.4g}\t{di:.4g}\t{dm / dr:.3f}\t{dlr / dr:.3f}\t{dr / di:.3f}", flush=True)
    out[f"{k}_D"] = np.array([dm, dlr, dr, di])
out["n_mirror"], out["n_LR"] = nm, nlr
np.savez(args.out, **out)
