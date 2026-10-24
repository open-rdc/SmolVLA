"""魚眼の津田沼(fisheye_tsudanuma_v2)用に、gen_para2.py が読む入力を周回データから作る。

  train_vocab_full.json  学習用 15 周に出てくる指示(言い換えがこれと一致したら漏れとして捨てる)
  train_words.json       その単語の出現数(L3 に学習語彙に無い語が入っているかの検査)
  data_cls.json          検証用周回の指示ごとの正解クラス。Ψ は eval_monitor.py と同じ
                         4.8 秒先(24 ステップ)・θ=0.090 rad で、フレームの多数決
  sel_instructions_v3.json  言い換えを作る指示 = 検証用周回に出てくる指示すべて

    python scripts/prep_fisheye_para.py ~/tsudanuma_data/laps --test tsudanuma_20261010_134655 \
        --out data/fisheye_tsudanuma_v2
    WORK=data/fisheye_tsudanuma_v2 python scripts/gen_para2.py
"""
import argparse
import collections
import json
import pickle
import re
from pathlib import Path

import numpy as np

H = 24
THETA = 0.090
DUMMY = "No language instruction"

ap = argparse.ArgumentParser()
ap.add_argument("laps_root", type=Path)
ap.add_argument("--test", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


def prompts(lap):
    return (lap / "episode01" / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")


laps = sorted(p for p in a.laps_root.iterdir() if (p / "episode01" / "traj_prompt.txt").exists())
train = [p for p in laps if p.name != a.test]
vocab = sorted({s.strip() for p in train for s in prompts(p)} - {"", DUMMY})
words = collections.Counter(w for s in vocab for w in re.findall(r"[a-z]+", s.lower()))

test = a.laps_root / a.test / "episode01"
pr = [s.strip() for s in prompts(a.laps_root / a.test)]
yaw = np.unwrap(np.asarray(pickle.load(open(test / "traj_data.pkl", "rb"))["yaw"], dtype=float))[:len(pr)]
n = len(pr)
psi = collections.defaultdict(list)
for t in range(n - H):
    psi[pr[t]].append(yaw[t + H] - yaw[t])
dc = {}
for task, ps in psi.items():
    p = np.array(ps)
    lab = np.where(p > THETA, "L", np.where(p < -THETA, "R", "S"))
    cnt = collections.Counter(lab)
    top, ntop = cnt.most_common(1)[0]
    dc[task] = dict(n=len(p), frac={k: cnt[k] / len(p) for k in "LRS"}, data_cls=top, purity=ntop / len(p),
                    mean=float(p.mean()), median=float(np.median(p)))

json.dump(vocab, open(a.out / "train_vocab_full.json", "w"), ensure_ascii=False, indent=1)
json.dump(dict(sorted(words.items())), open(a.out / "train_words.json", "w"), ensure_ascii=False, indent=1)
json.dump(dc, open(a.out / "data_cls.json", "w"), ensure_ascii=False, indent=1)
json.dump(sorted(dc), open(a.out / "sel_instructions_v3.json", "w"), ensure_ascii=False, indent=1)
cc = collections.Counter(v["data_cls"] for v in dc.values())
print(f"学習用 {len(train)} 周: 指示 {len(vocab)} 種 / 単語 {len(words)} 種")
print(f"検証用 {a.test}: 指示 {len(dc)} 種 (左折 {cc['L']} / 右折 {cc['R']} / 直進 {cc['S']})"
      f"、学習用にも出てくるもの {len(set(dc) & set(vocab))} 種")
