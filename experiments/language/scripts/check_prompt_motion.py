"""学習用の周回で、指示文の方向(left/right/straight/転回)と実際の動きが食い違っていないかを調べる。

単位は「同じ指示文が続く区間」。区間の向きの変化 = 区間の最後の向き - 最初の向き(unwrap 済み)。
  実際の動き: 区間の向きの変化が +45° 超 = 左, -45° 未満 = 右, ±20° 以内 = 直進, 135° 超 = 転回, それ以外 = あいまい
  指示文の方向: 曲がる動詞の近くの left/right(encoder_lr_test_v2.py の TURN と同じ)= 左/右,
                turn around / U-turn = 転回, straight/ahead だけ = 直進, それ以外(目印の側だけ・方向なし) = 対象外
使い方: python3 check_prompt_motion.py
"""
import collections
import pickle
import re
from pathlib import Path

import numpy as np

LAPS = Path("/home/kasai/tsudanuma_data/laps")
TRAIN = ["tsudanuma_20261010_142912", "tsudanuma_20261010_150555", "tsudanuma_20261011_094156",
         "tsudanuma_20261011_101738", "tsudanuma_20261011_105240", "tsudanuma_20261011_131250",
         "tsudanuma_20261011_132048", "tsudanuma_20261011_140558", "tsudanuma_20261011_143630",
         "tsudanuma_20261011_143944", "tsudanuma_20261013_102952", "tsudanuma_20261013_123146",
         "tsudanuma_20261013_130506", "tsudanuma_20261013_153231", "tsudanuma_20261013_161431"]
TURN = re.compile(r"\b(turn\w*|trun|curv\w*|bear|veer|swing|circl\w*)\b[^,.;]{0,25}?\b(left|right)\b"
                  r"|\b(left|right) turn\b|\b(toward|towards|into) the (left|right)\b", re.I)
UTURN = re.compile(r"\bturn(ing)? around\b|\bu-turn\b|\bturn back\b", re.I)
STRAIGHT = re.compile(r"\b(straight|ahead)\b", re.I)


def prompt_dir(p):
    if UTURN.search(p):
        return "U"
    m = TURN.search(p)
    if m:
        words = {w.lower() for w in re.findall(r"\b(left|right)\b", m.group(0), re.I)}
        if len(words) == 1:
            return "L" if "left" in words else "R"
    if STRAIGHT.search(p) and not re.search(r"\b(left|right)\b", p, re.I):
        return "S"
    return None


def motion_dir(dpsi):
    a = np.degrees(dpsi)
    if abs(a) > 135:
        return "U"
    if a > 45:
        return "L"
    if a < -45:
        return "R"
    if abs(a) <= 20:
        return "S"
    return "?"


segs = []
for lap in TRAIN:
    ep = LAPS / lap / "episode01"
    d = pickle.load(open(ep / "traj_data.pkl", "rb"))
    yaw = np.unwrap(np.asarray(d["yaw"], dtype=np.float64))
    pr = [s.strip() for s in (ep / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")][:len(yaw)]
    s = 0
    for t in range(1, len(pr) + 1):
        if t == len(pr) or pr[t] != pr[s]:
            segs.append(dict(lap=lap, start=s, end=t - 1, n=t - s, prompt=pr[s],
                             pdir=prompt_dir(pr[s]), dpsi=float(yaw[t - 1] - yaw[s])))
            s = t
for g in segs:
    g["mdir"] = motion_dir(g["dpsi"])

nfr = sum(g["n"] for g in segs)
cov = [g for g in segs if g["pdir"]]
print(f"区間 {len(segs)} / フレーム {nfr}。方向のある指示文の区間 {len(cov)} ({sum(g['n'] for g in cov) / nfr:.0%} のフレーム)")
print("\n指示文の方向 × 実際の動き(区間数、括弧内はフレーム数)")
cols = ["L", "S", "R", "U", "?"]
print("指示\\動き\t" + "\t".join(cols) + "\t一致率(区間)\t一致率(フレーム)")
for pd in ["L", "S", "R", "U"]:
    gs = [g for g in cov if g["pdir"] == pd]
    if not gs:
        continue
    cnt = collections.Counter(g["mdir"] for g in gs)
    fr = collections.Counter()
    for g in gs:
        fr[g["mdir"]] += g["n"]
    ok = cnt[pd] / len(gs)
    okf = fr[pd] / sum(fr.values())
    print(f"{pd}\t\t" + "\t".join(f"{cnt[c]}({fr[c]})" for c in cols) + f"\t{ok:.0%}\t\t{okf:.0%}")

print("\n逆向き(左の指示で右へ / 右の指示で左へ)の区間:")
rev = [g for g in cov if {g["pdir"], g["mdir"]} == {"L", "R"}]
for g in sorted(rev, key=lambda g: -g["n"]):
    print(f"  {g['lap'][10:]} {g['start']:5d}-{g['end']:5d} ({g['n']:4d}fr) Δψ={np.degrees(g['dpsi']):+6.0f}° | {g['prompt']}")

print("\n曲がる指示なのに直進(±20°以内)の区間(長い順に上位15):")
st = [g for g in cov if g["pdir"] in "LR" and g["mdir"] == "S"]
for g in sorted(st, key=lambda g: -g["n"])[:15]:
    print(f"  {g['lap'][10:]} {g['start']:5d}-{g['end']:5d} ({g['n']:4d}fr) Δψ={np.degrees(g['dpsi']):+6.0f}° | {g['prompt']}")
print(f"  (全 {len(st)} 区間 / {sum(g['n'] for g in st)} フレーム)")

print("\n直進の指示なのに曲がった(±45°超)区間(長い順に上位15):")
sc = [g for g in cov if g["pdir"] == "S" and g["mdir"] in "LRU"]
for g in sorted(sc, key=lambda g: -g["n"])[:15]:
    print(f"  {g['lap'][10:]} {g['start']:5d}-{g['end']:5d} ({g['n']:4d}fr) Δψ={np.degrees(g['dpsi']):+6.0f}° | {g['prompt']}")
print(f"  (全 {len(sc)} 区間 / {sum(g['n'] for g in sc)} フレーム)")

import json
json.dump(segs, open(Path(__file__).with_name("prompt_motion_segments.json"), "w"), ensure_ascii=False, indent=0)
