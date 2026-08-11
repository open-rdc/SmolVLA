#!/usr/bin/env python3
"""トポロジカルマップとの照合で traj_prompt.txt を自動生成する（オフライン）。

復帰動作や障害物回避のエピソードは、通常走行のようにマップ上を順番に遷移しない
（短い区間を任意の場所で切り出したもの）。そのため place_prompt_node が使う
Bayesian filter（遷移モデルで前方へ信念を伝播させる）は前提が成り立たない。
このスクリプトは **信念の伝播を使わず、毎フレームをマップ全体に対して照合**する。

さらに、票はノード単位ではなく **言語指示単位に集計**する。マップは926ノード
あるが指示は104種類しかなく、連続する複数ノードが同じ指示を共有しているため、
「どのノードか」より「どの指示か」を直接求めた方が良い。

判定は「各フレームについてマップ全体で最も似ている上位K ノードを取り、その指示に
投票する」という kNN 投票。toponav の観測尤度(delta=4)はフィルタの事前分布と
組み合わせる前提でわざと平坦にしてあり、単独で使うと926ノードに確率が拡散して
確信度が解釈できない値になるため使わない。

既定ではエピソード単位で1つの指示に決める（短い復帰クリップは1箇所で起きるので、
フレームごとに指示がばらつく方が不自然）。--per-frame でフレーム単位にもできる。

    # 確認だけ（何も書かない）
    python training/data/annotate_from_topomap.py <dataset_dir> [...]

    # traj_prompt.txt を書き出す
    python training/data/annotate_from_topomap.py <dataset_dir> --write

信頼度(confidence)が低いエピソードは手で直す前提でレポートに出す。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = REPO_ROOT / "deployment"
sys.path.insert(0, str(DEPLOY_DIR))

from smolvla_nav.toponav import TopologicalNavigator  # noqa: E402

# NavVLA の lang_anotation_tool.py が未注釈フレームに入れるのと同じ文字列。
# 一致が怪しいエピソードにはこれを書き、手作業が要ることを明示する。
DUMMY_LANGUAGE = "No language instruction"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", type=Path, help="episodeNN/ を含むデータセットディレクトリ")
    p.add_argument("--config", type=Path, default=DEPLOY_DIR / "config" / "topomap_nav.yaml")
    p.add_argument("--weights", type=Path, default=None,
                   help="PlaceNet の重み（既定は config の placenet_weight_path）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--write", action="store_true", help="traj_prompt.txt を実際に書き出す")
    p.add_argument("--per-frame", action="store_true",
                   help="エピソード単位で1つに決めず、フレームごとの最尤指示を書く")
    p.add_argument("--knn", type=int, default=20,
                   help="1フレームあたり、マップ全体から取る近傍ノード数")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="得票率がこれ未満のエピソードは要確認として警告する")
    p.add_argument("--max-distance", type=float, default=0.48,
                   help=f"最良一致の距離がこれを超えたら自動判定を採用せず "
                        f"'{DUMMY_LANGUAGE}' を書く（手作業が要る印）")
    p.add_argument("--show-top3", action="store_true", help="次点の指示も表示する（票割れの確認用）")
    p.add_argument("--overwrite", action="store_true",
                   help="既に traj_prompt.txt があっても上書きする")
    return p.parse_args()


def build_navigator(args: argparse.Namespace) -> TopologicalNavigator:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else DEPLOY_DIR / p

    weights = args.weights or resolve(str(cfg.get("placenet_weight_path", "weights/placenet.pt")))
    if not Path(weights).exists():
        raise SystemExit(
            f"PlaceNet の重みが見つかりません: {weights}\n"
            f"  --weights で明示するか、{DEPLOY_DIR / 'weights'} に placenet.pt を置いてください。"
        )
    return TopologicalNavigator(
        topomap_path=resolve(str(cfg.get("topomap_path", "config/topomap/topomap.yaml"))),
        image_dir=resolve(str(cfg.get("topomap_image_dir", "config/topomap/images"))),
        weight_path=Path(weights),
        device=torch.device(args.device),
        image_size=(224, 224),
        crop_size=int(cfg.get("toponav_crop_size", 288)),
    )


def extract_features_batch(nav: TopologicalNavigator, images_bgr: list[np.ndarray],
                           batch: int = 64) -> np.ndarray:
    """複数フレームの特徴量をまとめて計算する（(N, D) L2正規化済み）。

    toponav.extract_feature は1枚ずつ forward するので、エピソード単位でまとめると
    大幅に速くなる。前処理は extract_feature と同一。
    """
    tensors = [nav.transform(cv2.cvtColor(nav._center_crop(b), cv2.COLOR_BGR2RGB))
               for b in images_bgr]
    x = torch.stack(tensors).to(nav.device, dtype=torch.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(nav.model(x[i : i + batch]))
    f = torch.cat(outs).detach().cpu().numpy().reshape(len(x), -1).astype(np.float32)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise RuntimeError("PlaceNet returned a zero-norm feature.")
    return f / norms


def votes_from_features(feats: np.ndarray, feature_matrix: np.ndarray,
                        instr_of_node: list[str], k: int):
    """全フレームの kNN 投票と、各フレームの最良一致距離を返す。

    マップ全体（926ノード）から距離の近い上位 k ノードを取り、その指示に投票する。
    票は 1/(1+距離) で重み付けし、似ているノードほど強く効かせる。
    """
    # L2正規化済み同士なので dot=コサイン類似度 -> コサイン距離（_compute_distances と同式）
    dots = np.clip(feats @ feature_matrix.T, -1.0, 1.0)        # (N, n_nodes)
    dists = np.sqrt(2.0 - 2.0 * dots)
    idx = np.argpartition(dists, min(k, dists.shape[1] - 1), axis=1)[:, :k]

    per_frame = []
    for f in range(len(feats)):
        votes = defaultdict(float)
        for i in idx[f]:
            votes[instr_of_node[i]] += 1.0 / (1.0 + float(dists[f, i]))
        total = sum(votes.values())
        per_frame.append({kk: v / total for kk, v in votes.items()})

    # 近傍ノードが経路上の1箇所に固まっているか（=場所として同定できているか）。
    # 指示は言い換えが多く票が割れやすいので、得票率が低くてもここが高ければ
    # 「場所は特定できていて指示の文言だけ割れている」と判断できる。
    med = np.median(idx, axis=1, keepdims=True)
    locality = float(np.mean(np.abs(idx - med) <= 10))
    return per_frame, dists.min(axis=1), locality


def process_episode(ep: Path, nav: TopologicalNavigator, instr_of_node: list[str],
                    args: argparse.Namespace) -> dict:
    jpgs = sorted((p for p in ep.glob("*.jpg") if p.stem.isdigit()), key=lambda p: int(p.stem))
    if not jpgs:
        return {"dir": ep, "error": "jpg が無い"}

    images = []
    for jp in jpgs:
        bgr = cv2.imread(str(jp))
        if bgr is None:
            return {"dir": ep, "error": f"jpg が読めない: {jp.name}"}
        images.append(bgr)

    feats = extract_features_batch(nav, images)
    per_frame, best_dists, locality = votes_from_features(
        feats, nav.feature_matrix, instr_of_node, args.knn)

    # エピソード全体で平均 -> 最も確率の高い指示
    total: dict[str, float] = defaultdict(float)
    for d in per_frame:
        for k, v in d.items():
            total[k] += v
    for k in total:
        total[k] /= len(per_frame)
    ranked = sorted(total.items(), key=lambda kv: -kv[1])

    mean_dist = float(np.mean(best_dists))
    if args.max_distance is not None and mean_dist > args.max_distance:
        # 一致が怪しいので自動判定は採用せず、手作業が要ることが分かる文字列にする。
        prompts = [DUMMY_LANGUAGE] * len(jpgs)
    elif args.per_frame:
        prompts = [max(d.items(), key=lambda kv: kv[1])[0] for d in per_frame]
    else:
        prompts = [ranked[0][0]] * len(jpgs)

    return {
        "dir": ep,
        "n": len(jpgs),
        "prompts": prompts,
        "best": ranked[0][0],
        "conf": ranked[0][1],
        "second": ranked[1][0] if len(ranked) > 1 else "",
        "second_conf": ranked[1][1] if len(ranked) > 1 else 0.0,
        "n_unique": len(set(prompts)),
        "dist": mean_dist,
        "dummy": args.max_distance is not None and mean_dist > args.max_distance,
        "top3": ranked[:3],
        "loc": locality,
    }


def main() -> None:
    args = parse_args()
    nav = build_navigator(args)
    instr_of_node = [n.instruction for n in nav.nodes]
    print(f"topomap: {len(nav.nodes)} ノード / {len(set(instr_of_node))} 種類の指示 "
          f"/ device={args.device}")

    low, written, skipped = [], 0, 0
    for ds in args.datasets:
        eps = sorted((p for p in ds.iterdir() if p.is_dir() and not p.name.startswith("_")),
                     key=lambda p: p.name)
        print(f"\n{'=' * 78}\n{ds}  ({len(eps)} episodes)\n{'=' * 78}")
        for ep in eps:
            r = process_episode(ep, nav, instr_of_node, args)
            if "error" in r:
                print(f"  {ep.name:<12} ★{r['error']}")
                continue

            far = r["dummy"]
            mark = "★" if far else "  "
            uniq = f" [{r['n_unique']}種]" if r["n_unique"] > 1 else ""
            print(f"{mark}{ep.name:<12} n={r['n']:>3} 得票={r['conf']:.2f} "
                  f"距離={r['dist']:.3f} 局所性={r['loc']:.2f}{uniq}  "
                  f"{DUMMY_LANGUAGE + ' <- 要手作業 (' + r['best'] + ')' if far else r['best']}")
            if args.show_top3:
                for instr, v in r["top3"][1:]:
                    print(f"      次点 {v:.2f}  {instr}")
            if far:
                low.append(r)

            if args.write:
                out = ep / "traj_prompt.txt"
                if out.exists() and not args.overwrite:
                    skipped += 1
                    continue
                # lang_anotation_tool.py と同じ書式（1行1フレーム、末尾に改行なし）
                out.write_text("\n".join(r["prompts"]), encoding="utf-8")
                written += 1

    print(f"\n{'=' * 78}")
    if low:
        print(f"★ 距離 {args.max_distance} 超が {len(low)} 件 -> "
              f"'{DUMMY_LANGUAGE}' を書いたので手で付けてください")
        for r in low:
            print(f"   {r['dir'].parent.name}/{r['dir'].name}: 得票 {r['conf']:.2f} "
                  f"距離 {r['dist']:.3f} \"{r['best']}\" / 次点 {r['second_conf']:.2f} \"{r['second']}\"")
    else:
        print("全エピソードが距離のしきい値を満たしました")

    if args.write:
        print(f"\ntraj_prompt.txt を {written} 件書き出しました"
              + (f"（既存のため {skipped} 件スキップ。上書きは --overwrite）" if skipped else ""))
    else:
        print("\n（確認のみ。書き出すには --write を付けてください）")


if __name__ == "__main__":
    main()
