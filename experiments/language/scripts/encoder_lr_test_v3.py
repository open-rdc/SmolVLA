"""(v3 の土台は v2。モデルの読み込み・文の分類は v2 と同じ)
文章エンコーダ1つについて、指示文の left/right を文ベクトルから読み分けられるかを測る(v2)。

v1 からの変更:
  - left と right を両方含む文は除く(入れ替えても単語の組み合わせが同じになるため)
  - left/right が「曲がる方向」(turn)か「目印の側」(side, on your left など)かで分けて集計する
  - ベクトルが動いた量だけでなく、読み分けられるか(差の向きのそろい方・線形分類)も測る
  - 分岐で実際に使われた左/直進/右の指示文どうしの類似度も測る

集合ごと(old=旧カメラ, new=新カメラ)・種類ごと(turn/side)の指標:
  swap      : 元の文と left/right を入れ替えた文のコサイン類似度(中央値)
  ratio     : (1 - swap) / (1 - other)。other は無関係な指示どうしの類似度の平均
  dir_cons  : 差ベクトル(left 版 - right 版)の向きのそろい方(正規化した差どうしのコサインの平均)
  probe     : left 版か right 版かを当てる線形分類の正解率(文ごとに分けた 5 分割交差検証、chance 0.5)
分岐(検証周回 134655 の分岐フレームで使われた学習時の指示文 prompt_ref_{L,S,R}):
  br_LR 等  : 同じ分岐での左と右の指示文のコサイン類似度(異なる組ごとの平均)と ratio
使い方: python encoder_lr_test_v2.py <model名> <OUT.jsonl> <branch_ref.npz>
"""
import gc
import glob
import json
import random
import re
import sys

import numpy as np
import torch

torch.set_num_threads(4)
MIR = re.compile(r"\b(left|right)\b", re.I)
SIDE = re.compile(r"\b(on|to|at) (your|the) (left|right)\b|\b(left|right)[- ]hand\b|\b(left|right) side\b|\bon (left|right)\b|\bfar (left|right)\b",
                  re.I)
TURN = re.compile(r"\b(turn\w*|trun|curv\w*|bear|veer|swing|circl\w*)\b[^,.;]{0,25}?\b(left|right)\b"
                  r"|\b(left|right) turn\b|\b(toward|towards|into) the (left|right)\b", re.I)


def mirror(s):
    def sw(m):
        w = "right" if m.group(1).lower() == "left" else "left"
        return w if m.group(1).islower() else w.capitalize()
    return MIR.sub(sw, s)


def load_sets():
    old = set()
    for line in open("/home/kasai/tsudanuma_data/all_instructions.tsv", encoding="utf-8"):
        _, src, text = line.rstrip("\n").split("\t")
        if src != "(none)" and text.strip():
            old.add(text.strip())
    new = set()
    for f in glob.glob("/home/kasai/tsudanuma_data/laps/*/episode01/traj_prompt.txt"):
        new |= {l.strip() for l in open(f, encoding="utf-8") if l.strip()}
    return {"old": sorted(old), "new": sorted(new)}


def split_lr(prompts):
    """left/right を1種類だけ含む文を turn(曲がる動詞の近くに left/right) / side(on your left 等)に分ける。どちらでもない文は使わない。"""
    out = {"turn": [], "side": []}
    for p in prompts:
        words = {w.lower() for w in MIR.findall(p)}
        if len(words) != 1:
            continue
        if TURN.search(p):
            out["turn"].append(p)
        elif SIDE.search(p):
            out["side"].append(p)
    return out


def load_branch(path):
    br = np.load(path)
    ts = np.where(br["branch"])[0]
    pairs = {"LR": set(), "LS": set(), "SR": set()}
    for t in ts:
        p = {c: str(br[f"prompt_ref_{c}"][t]) for c in "LSR"}
        for k in pairs:
            a, b = p[k[0]], p[k[1]]
            if a and b and a != b:
                pairs[k].add((a, b))
    return {k: sorted(v) for k, v in pairs.items()}


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)


def make(name):
    from transformers import AutoTokenizer
    if name.startswith("openai/clip"):
        from transformers import CLIPTextModelWithProjection
        tok = AutoTokenizer.from_pretrained(name)
        mod = CLIPTextModelWithProjection.from_pretrained(name).eval()
        def enc(ts):
            return mod(**tok(ts, padding=True, truncation=True, return_tensors="pt")).text_embeds
        return enc
    if "siglip" in name:
        from transformers import SiglipTextModel
        tok = AutoTokenizer.from_pretrained(name)
        mod = SiglipTextModel.from_pretrained(name).eval()
        def enc(ts):
            x = tok([t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            return mod(input_ids=x["input_ids"]).pooler_output
        return enc
    if "SmolVLM" in name:
        from transformers import AutoModelForImageTextToText
        tok = AutoTokenizer.from_pretrained(name)
        full = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16)
        mod = full.model.text_model
        del full  # 画像側を先に捨ててから fp32 にする(メモリのピークを下げる)
        gc.collect()
        mod = mod.float().eval()
        def enc(ts):
            x = tok(ts, padding=True, return_tensors="pt")
            return mean_pool(mod(**x).last_hidden_state, x["attention_mask"])
        return enc
    from transformers import AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    if "t5" in name:
        from transformers import T5EncoderModel
        mod = T5EncoderModel.from_pretrained(name).eval()
    else:
        mod = AutoModel.from_pretrained(name).eval()
    def enc(ts):
        x = tok(ts, padding=True, truncation=True, return_tensors="pt")
        return mean_pool(mod(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).last_hidden_state,
                         x["attention_mask"])
    return enc


class Embedder:
    def __init__(self, enc):
        self.enc, self.cache = enc, {}

    def __call__(self, ts, bs=32):
        todo = [t for t in dict.fromkeys(ts) if t not in self.cache]
        with torch.no_grad():
            for i in range(0, len(todo), bs):
                v = torch.nn.functional.normalize(self.enc(todo[i:i + bs]).float(), dim=-1)
                self.cache.update(zip(todo[i:i + bs], v))
        return torch.stack([self.cache[t] for t in ts])


def probe_acc(xl, xr, folds=5, seed=0):
    """left 版(1)と right 版(0)を当てる線形分類。同じ文の2つの版は同じ fold に入れる。"""
    n = len(xl)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    correct = 0
    for k in range(folds):
        te = set(idx[k::folds])
        tr = [i for i in range(n) if i not in te]
        te = sorted(te)
        Xtr = torch.cat([xl[tr], xr[tr]])
        ytr = torch.cat([torch.ones(len(tr)), torch.zeros(len(tr))])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        w = torch.zeros(Xtr.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-3 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            correct += (((xl[te] - mu) / sd @ w + b) > 0).sum().item()
            correct += (((xr[te] - mu) / sd @ w + b) <= 0).sum().item()
    return correct / (2 * n)


def other_sim(emb, prompts, n=2000):
    rng = random.Random(0)
    pairs = [tuple(rng.sample(prompts, 2)) for _ in range(n)]
    return (emb([a for a, _ in pairs]) * emb([b for _, b in pairs])).sum(1).mean().item()


def lr_metrics(emb, group, other):
    if len(group) < 10:
        return {"n": len(group)}
    a, b = emb(group), emb([mirror(s) for s in group])
    swap = (a * b).sum(1).median().item()
    is_left = torch.tensor([bool(re.search(r"\bleft\b", s, re.I)) for s in group])
    xl = torch.where(is_left[:, None], a, b)  # left 版
    xr = torch.where(is_left[:, None], b, a)  # right 版
    d = torch.nn.functional.normalize(xl - xr, dim=-1)
    s = d.sum(0)
    n = len(group)
    dir_cons = ((s @ s - n) / (n * (n - 1))).item()
    return {"n": n, "swap": round(swap, 4), "ratio": round((1 - swap) / (1 - other), 4),
            "dir_cons": round(dir_cons, 4), "probe": round(probe_acc(xl, xr), 4)}



# ---------------------------------------------------------------- v3: 難しくした線形分類
# どれも「曲がる方向」(turn)の文だけを使い、入れ替えた文の組は使わない(元の文だけ)。ラベルは文中の left/right。
# 正解率は left と right それぞれの正解率の平均(バランス正解率、当てずっぽうで 0.5)。
PARA = "/home/kasai/ros2_ws/src/SmolVLA/experiments/language/data/fisheye_tsudanuma_v2/paraphrases_v3.json"


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def label(s):
    return 1 if re.search(r"\bleft\b", s, re.I) else 0


def fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    yt = torch.tensor(y, dtype=torch.float32)
    # クラスの偏りを打ち消す重み
    pw = (yt == 0).sum() / max((yt == 1).sum(), 1)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, yt, pos_weight=pw) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Z: (((Z - mu) / sd @ w + b) > 0).long().tolist()


def bal_acc(pred, y):
    accs = [np.mean([p == t for p, t in zip(pred, y) if t == c]) for c in (0, 1) if c in y]
    return round(float(np.mean(accs)), 4)


def cv_natural(emb, sents, folds=5, seed=0):
    groups = sorted({norm(s) for s in sents})
    random.Random(seed).shuffle(groups)
    gid = {g: i % folds for i, g in enumerate(groups)}
    pred, y = [], []
    for k in range(folds):
        tr = [s for s in sents if gid[norm(s)] != k]
        te = [s for s in sents if gid[norm(s)] == k]
        if not te:
            continue
        f = fit(emb(tr), [label(s) for s in tr])
        pred += f(emb(te))
        y += [label(s) for s in te]
    return {"n": len(sents), "n_left": sum(y), "acc": bal_acc(pred, y)}


name, out = sys.argv[1], sys.argv[2]
emb = Embedder(make(name))
sets = load_sets()
turn = {cam: split_lr(p)["turn"] for cam, p in sets.items()}
res = {"model": name}
# 1. 別の文どうし(5分割、ほぼ同じ文は同じ分割へ)
for cam in ("old", "new"):
    res[f"nat_{cam}"] = cv_natural(emb, turn[cam])
# 2. 旧カメラで学習 → 旧カメラに無い新カメラの文で評価
old_n = {norm(s) for s in turn["old"]}
te = [s for s in turn["new"] if norm(s) not in old_n]
f = fit(emb(turn["old"]), [label(s) for s in turn["old"]])
res["old2new"] = {"n_train": len(turn["old"]), "n_test": len(te), "acc": bal_acc(f(emb(te)), [label(s) for s in te])}
# 3. 言い換え: 元の文(と表記ゆれ)を学習から除き、L1/L2/L3 で評価
para = json.load(open(PARA))
src = [k for k in para if k in turn["new"] or k in turn["old"]]
src_n = {norm(k) for k in src}
tr = [s for s in dict.fromkeys(turn["old"] + turn["new"]) if norm(s) not in src_n]
f = fit(emb(tr), [label(s) for s in tr])
res["para_src"] = {"n_train": len(tr), "n_src": len(src),
                   "acc": bal_acc(f(emb(src)), [label(s) for s in src])}
for lv in ("L1", "L2", "L3"):
    ps = [(p, label(k)) for k in src for p in para[k][lv] if MIR.search(p) and label(p) == label(k)]
    if ps:
        res[f"para_{lv}"] = {"n": len(ps), "acc": bal_acc(f(emb([p for p, _ in ps])), [y for _, y in ps])}
print(json.dumps(res, ensure_ascii=False), flush=True)
with open(out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
