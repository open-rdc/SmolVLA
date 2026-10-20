"""毎エポックの重みを検証用データで評価し、指標を TensorBoard に書く(SmolVLA と OmniVLA-edge 共通)。

holdout_v2 の検証用(orne_box_all_tc_val2)から固定のフレームを選び、2 つのモデルを
同じフレーム・同じ正解・同じ指標で評価する。テスト用データは読まない。

■ フレームと正解
  - 実走行 31 本から 1,000 フレーム、合成版(天候違い)から 300 フレームを固定のシードで選ぶ
  - 4.8 秒先(24 ステップ)まで同じエピソード内に収まるフレームだけ(OmniVLA-edge の予測範囲に合わせる)
  - 正解の位置と向きは元データの traj_data.pkl。Ψ = yaw[t+24] - yaw[t]、行動クラスは θ=0.090 rad
■ 推論
  - SmolVLA: Heun 法 4 ステップ(実機と同じ)。ノイズはフレームとシードごとに固定し、3 種類で平均
  - OmniVLA-edge: 決定的(ノイズなし)。指示なしは NavVLA の学習時と同じ "No language instruction"
■ 条件
  元の指示 / 指示なし / 方向語反転(left<->right) / 未知の言い換え L1・L2・L3(paraphrases_v3.json)
  / 合成版(元の指示)
■ TensorBoard (tb_root/<run>/、横軸は epoch x 10000)
  val/psi_mae, val/psi_mae_turn(エポック選択の基準), val/acc, val/acc_{left,right,straight},
  val/psi_corr, val/ade, val/fde, val/dx_err, val/acc_noinst, val/lang_gain, val/lang_gain_turn,
  val/mirror_flip, val/acc_{L1,L2,L3}, val/psi_mae_turn_{L1,L2,L3}, val/acc_cosmos,
  val/psi_mae_cosmos, val/psi_spread(SmolVLA のみ), val/flow_loss(SmolVLA のみ)
  画像: val/confusion(混同行列), val/trajectories(予測と正解の軌跡)
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HOME = Path.home()
H = 24            # 4.8 秒 = 24 ステップ(5Hz)
SPACING = 3       # OmniVLA-edge の点の間隔
THETA = 0.090     # 行動クラスの閾値 [rad](第5章と同じ)
SCALE = 10000     # 横軸 = epoch x SCALE
DUMMY = "No language instruction"
MIRROR = re.compile(r"\b(left|right)\b", re.I)


def mirror(s: str) -> str:
    return MIRROR.sub(lambda m: "right" if m.group(1).lower() == "left" else "left", s)


def cls_of(psi):
    psi = np.asarray(psi)
    return np.where(psi > THETA, 0, np.where(psi < -THETA, 1, 2))  # 0=左折 1=右折 2=直進


def local(pos, yaw, t, idx):
    c, s = math.cos(yaw[t]), math.sin(yaw[t])
    return (pos[idx] - pos[t]) @ np.array([[c, -s], [s, c]], dtype=np.float64)


# ---------------------------------------------------------------- フレームの選択と正解
def build_frames(args):
    root = args.val_root
    info = json.load(open(root / "meta/info.json"))
    eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_of = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
    data = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "task_index", "index"])
                      for p in sorted(glob.glob(str(root / "data/*/*.parquet")))])
    emap = json.load(open(args.map))["val"]
    assert len(emap) == info["total_episodes"]
    rng = random.Random(args.seed)
    frames = {"real": [], "cosmos": []}
    for e, row in eps.iterrows():
        e = int(row["episode_index"])
        L = int(row["length"])
        raw = emap[e]["raw"]
        name = emap[e].get("omni", f"ep{e:05d}")  # OmniVLA-edge 側のエピソード名
        d = pickle.load(open(args.raw_pkl / raw / "traj_data.pkl", "rb"))
        pos = np.asarray(d["position"], dtype=np.float64)
        yaw = np.asarray(d["yaw"], dtype=np.float64)
        g = data[data["episode_index"] == e].sort_values("frame_index")
        kind = "cosmos" if "cosmos" in raw else "real"
        for t in range(0, L - H):
            idx = [t + SPACING * k for k in range(1, H // SPACING + 1)]
            frames[kind].append(dict(
                ep=e, name=name, t=t, L=L, gidx=int(g["index"].iloc[t]), task=task_of[int(g["task_index"].iloc[t])],
                psi=float(yaw[t + H] - yaw[t]), pts=local(pos, yaw, t, idx).tolist(),
            ))
    out = {}
    for kind, n in [("real", args.n_real), ("cosmos", args.n_cosmos)]:
        pool = frames[kind]
        out[kind] = sorted(rng.sample(pool, min(n, len(pool))), key=lambda f: f["gidx"])
    print(f"frames: real {len(out['real'])} / cosmos {len(out['cosmos'])}", flush=True)
    return out


def conditions(frames, para):
    """(条件名, フレーム番号, 指示文) の一覧。"""
    rng = random.Random(1)
    out = []
    for i, f in enumerate(frames["real"]):
        out.append(("orig", i, f["task"]))
        out.append(("noinst", i, ""))
        if MIRROR.search(f["task"]):
            out.append(("mirror", i, mirror(f["task"])))
        if f["task"] in para:
            for lv in ("L1", "L2", "L3"):
                if para[f["task"]][lv]:
                    out.append((lv, i, rng.choice(para[f["task"]][lv])))
    for i, f in enumerate(frames["cosmos"]):
        out.append(("cosmos", i, f["task"]))
    return out


# ---------------------------------------------------------------- 指標
def metrics(frames, preds, writer, x, run, spread=None):
    """preds: {条件: {フレーム番号: (psi, pts(8,2))}}  (SmolVLA は 3 シードの平均)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = frames["real"]
    m = {}

    def arr(cond, fr):
        ids = sorted(preds.get(cond, {}))
        p = np.array([preds[cond][i][0] for i in ids])
        t = np.array([fr[i]["psi"] for i in ids])
        return ids, p, t

    ids, p, t = arr("orig", real)
    tc, pc = cls_of(t), cls_of(p)
    turn = tc != 2
    m["psi_mae"] = float(np.abs(p - t).mean())
    m["psi_mae_turn"] = float(np.abs(p - t)[turn].mean())
    m["acc"] = float((pc == tc).mean())
    for k, name in enumerate(["left", "right", "straight"]):
        m[f"acc_{name}"] = float((pc[tc == k] == k).mean()) if (tc == k).any() else float("nan")
    m["psi_corr"] = float(np.corrcoef(p, t)[0, 1])
    pts_p = np.array([preds["orig"][i][1] for i in ids])
    pts_t = np.array([real[i]["pts"] for i in ids])
    dist = np.linalg.norm(pts_p - pts_t, axis=-1)
    m["ade"] = float(dist.mean())
    m["fde"] = float(dist[:, -1].mean())
    m["dx_err"] = float(np.abs(pts_p[:, -1, 0] - pts_t[:, -1, 0]).mean())

    ids_n, p_n, t_n = arr("noinst", real)
    acc_n = cls_of(p_n) == cls_of(t_n)
    m["acc_noinst"] = float(acc_n.mean())
    m["lang_gain"] = m["acc"] - m["acc_noinst"]
    turn_n = cls_of(t_n) != 2
    m["lang_gain_turn"] = float((pc == tc)[turn].mean() - acc_n[turn_n].mean())

    # 方向語反転: 旋回フレームで、予測 Ψ の符号が元の指示のときと逆になった割合
    mi = [i for i in sorted(preds.get("mirror", {})) if cls_of(real[i]["psi"]) != 2]
    if mi:
        m["mirror_flip"] = float(np.mean([np.sign(preds["mirror"][i][0]) != np.sign(preds["orig"][i][0]) for i in mi]))

    for lv in ("L1", "L2", "L3"):
        if preds.get(lv):
            i_l, p_l, t_l = arr(lv, real)
            m[f"acc_{lv}"] = float((cls_of(p_l) == cls_of(t_l)).mean())
            tt = cls_of(t_l) != 2
            m[f"psi_mae_turn_{lv}"] = float(np.abs(p_l - t_l)[tt].mean()) if tt.any() else float("nan")

    if preds.get("cosmos"):
        i_c, p_c, t_c = arr("cosmos", frames["cosmos"])
        m["acc_cosmos"] = float((cls_of(p_c) == cls_of(t_c)).mean())
        m["psi_mae_cosmos"] = float(np.abs(p_c - t_c).mean())
    if spread is not None:
        m["psi_spread"] = spread

    for k, v in m.items():
        if v == v:  # NaN は書かない
            writer.add_scalar(f"val/{k}", v, x)

    # 混同行列
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(tc, pc):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    lab = ["L", "R", "S"]
    ax.set_xticks(range(3), lab)
    ax.set_yticks(range(3), lab)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for a in range(3):
        for b in range(3):
            ax.text(b, a, cm[a, b], ha="center", va="center")
    ax.set_title(f"{run} epoch {x / SCALE:.1f}")
    writer.add_figure("val/confusion", fig, x)
    plt.close(fig)

    # 軌跡(固定の 6 フレーム)
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    pick = ids[:: max(1, len(ids) // 6)][:6]
    for a, i in zip(axs.flat, pick):
        tp, pp = np.array(real[i]["pts"]), np.array(preds["orig"][i][1])
        a.plot(-tp[:, 1], tp[:, 0], "k.-", label="true")
        a.plot(-pp[:, 1], pp[:, 0], "r.-", label="pred")
        a.set_aspect("equal")
        a.set_title(real[i]["task"][:32], fontsize=7)
    axs.flat[0].legend(fontsize=7)
    writer.add_figure("val/trajectories", fig, x)
    plt.close(fig)
    writer.flush()
    return m


# ---------------------------------------------------------------- SmolVLA
class SmolVLAEval:
    def __init__(self, args, frames):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.args, self.frames = args, frames
        ds = LeRobotDataset(args.val_repo_id, root=args.val_root, video_backend="pyav")
        self.obs = {}
        for kind in ("real", "cosmos"):
            for i, f in enumerate(frames[kind]):
                it = ds[f["gidx"]]
                self.obs[(kind, i)] = {k: v for k, v in it.items()
                                       if k.startswith("observation.")}
        # flow loss 用(行動列つき)
        self.ds_chunk = None

    def predict(self, ckpt_pm: Path, conds):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        pol = SmolVLAPolicy.from_pretrained(str(ckpt_pm)).to("cuda").eval()
        pol.config.num_steps = 4  # Heun 法 4 ステップ(実機と同じ)
        pre, post = make_pre_post_processors(
            pol.config, pretrained_path=str(ckpt_pm),
            preprocessor_overrides={"device_processor": {"device": "cuda"},
                                    "smolvla_paraphrase_augment_processor": {"enabled": False}})
        CH, AD = pol.config.chunk_size, pol.config.max_action_dim
        seeds = range(self.args.seeds)
        raw = {}  # (cond, i) -> list of (psi, pts)
        bs = self.args.batch_size
        for s0 in range(0, len(conds), bs):
            chunk = conds[s0:s0 + bs]
            for sd in seeds:
                # orig 以外は 1 シードだけ(計算量を抑える)
                sub = [c for c in chunk if sd == 0 or c[0] == "orig"]
                if not sub:
                    continue
                batch = {}
                kinds = [("cosmos" if c[0] == "cosmos" else "real", c[1]) for c in sub]
                for k in self.obs[kinds[0]]:
                    batch[k] = torch.stack([self.obs[kk][k] for kk in kinds])
                batch["task"] = [c[2] for c in sub]
                noise = torch.stack([
                    torch.normal(0., 1., size=(CH, AD), generator=torch.Generator().manual_seed(
                        sd * 100003 + self.frames[kk[0]][kk[1]]["gidx"])) for kk in kinds]).to("cuda")
                with torch.no_grad():
                    act = post(pol.predict_action_chunk(pre(batch), noise=noise)).float().cpu().numpy()
                for c, a in zip(sub, act):
                    psi = float(a[:H, 1].sum())
                    x = y = th = 0.0
                    pts = []
                    for k in range(H):
                        x += a[k, 0] * math.cos(th)
                        y += a[k, 0] * math.sin(th)
                        th += a[k, 1]
                        if (k + 1) % SPACING == 0:
                            pts.append([x, y])
                    raw.setdefault((c[0], c[1]), []).append((psi, pts))
        del pol
        torch.cuda.empty_cache()
        preds, spreads = {}, []
        for (cond, i), v in raw.items():
            ps = [a for a, _ in v]
            preds.setdefault(cond, {})[i] = (float(np.mean(ps)), np.mean([b for _, b in v], axis=0).tolist())
            if cond == "orig" and len(ps) > 1:
                spreads.append(float(np.std(ps)))
        return preds, (float(np.mean(spreads)) if spreads else None)


# ---------------------------------------------------------------- OmniVLA-edge
class OmniVLAEval:
    def __init__(self, args, frames):
        import yaml
        sys.path.insert(0, str(HOME / "NavVLA"))
        from training.data.dataset import EdgeNavigationDataset
        self.args, self.frames = args, frames
        cdir = HOME / "NavVLA/training/config" / args.omni_config
        nc = yaml.safe_load(open(cdir / "network.yaml"))
        dc = yaml.safe_load(open(cdir / "dataset.yaml"))
        cfg = dc["datasets"][args.omni_dataset]
        root = Path(cfg["data_folder"])
        names = sorted({f["name"] for k in frames for f in frames[k]})
        self.ds = EdgeNavigationDataset(
            data_folder=root, traj_names=names, dataset_name="val2", image_size=tuple(dc["image_size"]),
            waypoint_spacing=cfg["waypoint_spacing"], len_traj_pred=nc["len_traj_pred"],
            learn_angle=nc["learn_angle"], context_size=nc["context_size"], end_slack=cfg["end_slack"],
            modality_id=cfg["modality_id"], metric_waypoint_spacing=cfg["metric_waypoint_spacing"],
            pad_episode_edges=dc["pad_episode_edges"])
        self.nc = nc
        self.scale = cfg["metric_waypoint_spacing"] * cfg["waypoint_spacing"]
        self.samples = {}
        for kind in ("real", "cosmos"):
            for i, f in enumerate(frames[kind]):
                s = self.ds.build_sample(f["name"], f["t"], min(f["t"] + H, f["L"] - 1))
                self.samples[(kind, i)] = s

    def predict(self, weights: Path, conds):
        from OmniVLA.inference.model_omnivla_edge import OmniVLA_edge
        nc = self.nc
        model = OmniVLA_edge(
            context_size=nc["context_size"], len_traj_pred=nc["len_traj_pred"], learn_angle=nc["learn_angle"],
            obs_encoder=nc["obs_encoder"], obs_encoding_size=nc["obs_encoding_size"],
            late_fusion=nc["late_fusion"], mha_num_attention_heads=nc["mha_num_attention_heads"],
            mha_num_attention_layers=nc["mha_num_attention_layers"], mha_ff_dim_factor=nc["mha_ff_dim_factor"])
        sd = torch.load(weights, map_location="cpu")
        model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd)
        model = model.to("cuda").eval()
        preds = {}
        bs = self.args.batch_size
        for s0 in range(0, len(conds), bs):
            sub = conds[s0:s0 + bs]
            kinds = [("cosmos" if c[0] == "cosmos" else "real", c[1]) for c in sub]
            batch = {k: torch.stack([self.samples[kk][k] for kk in kinds]).to("cuda")
                     for k in self.samples[kinds[0]]}
            batch["feat_text"] = torch.stack([self.ds.encode_text(c[2] or DUMMY) for c in sub]).to("cuda")
            with torch.no_grad():
                a, _, _ = model(batch["obs_images"], batch["goal_pose"].float(), batch["map_images"],
                                batch["goal_image"], batch["goal_mask"].long(), batch["feat_text"].float(),
                                batch["current_img"])
            a = a.float().cpu().numpy()
            for c, ai in zip(sub, a):
                pts = (ai[:, :2] * self.scale).tolist()
                psi = float(math.atan2(ai[-1, 3], ai[-1, 2]))
                preds.setdefault(c[0], {})[c[1]] = (psi, pts)
        del model
        torch.cuda.empty_cache()
        return preds, None


# ---------------------------------------------------------------- メイン
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smolvla-run", action="append", default=[])
    ap.add_argument("--omnivla-run", action="append", default=[])
    ap.add_argument("--omnivla-pretrained", action="store_true",
                    help="事前学習済み omnivla-edge.pth を epoch 0 として評価する")
    ap.add_argument("--val-root", type=Path, default=HOME / "smolvla_data/orne_box_all_tc_val2")
    ap.add_argument("--val-repo-id", default="open-rdc/orne_box_all_tc_val2")
    ap.add_argument("--omni-config", default="holdout_v2", help="NavVLA/training/config/ の下の設定フォルダ")
    ap.add_argument("--omni-dataset", default="orne_box_all_tc_val2", help="その dataset.yaml の中の検証用データの名前")
    ap.add_argument("--map", type=Path, default=HOME / "SmolVLA/training/data/splits/holdout_v2_episode_map.json")
    ap.add_argument("--raw-pkl", type=Path, default=HOME / "navvla_data/raw_pkl")
    ap.add_argument("--para", type=Path, default=HOME / "SmolVLA/training/data/paraphrases_v3.json")
    ap.add_argument("--tb-root", type=Path, default=HOME / "SmolVLA/tb_eval_ho2")
    ap.add_argument("--train-frames", type=int, default=293430)
    ap.add_argument("--n-real", type=int, default=1000)
    ap.add_argument("--n-cosmos", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    from torch.utils.tensorboard import SummaryWriter

    frames = build_frames(args)
    para = json.load(open(args.para))
    conds = conditions(frames, para)
    print(f"conditions: {len(conds)} ({', '.join(sorted({c[0] for c in conds}))})", flush=True)
    spe = args.train_frames / 8  # SmolVLA の 1 エポックあたり step

    evaluators = {}
    jobs = [("smolvla", r) for r in args.smolvla_run] + [("omnivla", r) for r in args.omnivla_run]
    writers = {r: SummaryWriter(str(args.tb_root / r)) for _, r in jobs}
    state = {}
    for _, r in jobs:
        f = args.tb_root / r / "eval_results.json"
        state[r] = json.load(open(f)) if f.exists() else {}

    def evaluate(kind, run, key, path, epoch):
        if kind not in evaluators:
            evaluators[kind] = (SmolVLAEval if kind == "smolvla" else OmniVLAEval)(args, frames)
        t0 = time.time()
        preds, spread = evaluators[kind].predict(path, conds)
        x = round(epoch * SCALE)
        m = metrics(frames, preds, writers[run], x, run, spread)
        m["epoch"] = epoch
        state[run][key] = m
        json.dump(state[run], open(args.tb_root / run / "eval_results.json", "w"), indent=1)
        best = min(state[run].items(), key=lambda kv: kv[1]["psi_mae_turn"])
        print(f"[{run}] epoch {epoch:.2f}: psi_mae_turn={m['psi_mae_turn']:.4f} acc={m['acc']:.3f} "
              f"({time.time() - t0:.0f}s) best=epoch {best[1]['epoch']:.2f}", flush=True)

    while True:
        if args.omnivla_pretrained and "pretrained" not in state.get(args.omnivla_run[0], {}):
            evaluate("omnivla", args.omnivla_run[0], "pretrained",
                     HOME / "NavVLA/training/weights/omnivla-edge.pth", 0.0)
        for kind, run in jobs:
            if kind == "smolvla":
                root = HOME / "SmolVLA/outputs/train" / f"smolvla_{run}" / "checkpoints"
                cands = []
                for p in root.glob("[0-9]*"):
                    w = p / "pretrained_model" / "model.safetensors"
                    if w.exists() and time.time() - w.stat().st_mtime > 120:
                        cands.append((p.name, p / "pretrained_model", int(p.name) / spe))
            else:
                root = HOME / "NavVLA/training/runs" / run
                cands = []
                for p in root.glob("model_epoch_*.pth"):
                    if time.time() - p.stat().st_mtime > 120:
                        cands.append((p.stem, p, float(int(p.stem.split("_")[-1]))))
            for key, path, epoch in sorted(cands, key=lambda c: c[2]):
                if key not in state[run]:
                    evaluate(kind, run, key, path, epoch)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
