#!/usr/bin/env python3
"""OmniVLA-edge の言語エンコーダ(凍結 CLIP ViT-B/32)が left/right を区別できているかを測る。

left/right を含む指示について、元の指示と「left<->right を入れ替えた文」「L1/L2/L3 の言い換え」
とのコサイン類似度を比べる。反転文の方が言い換えより近ければ、左右の情報は埋め込みの時点で弱い。
OmniVLA-edge はこの埋め込み1本を FiLM で画像特徴に掛けるだけなので、言語が入力される時点の状態を見る目安になる。
注意: 類似度が高いことは線形分離できないことを意味しない(それには線形プローブが要る)。

使い方(GPU不要、CLIP の入った環境で):
    python scripts/clip_lr_check.py data/paraphrases_v3.json > data/clip_lr_check.txt
"""
import json, re, sys
import numpy as np, torch, clip

MIRROR = re.compile(r"\b(left|right)\b", re.I)
def mirror(s):
    return MIRROR.sub(lambda m: "right" if m.group(1).lower() == "left" else "left", s)

para = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/paraphrases_v3.json"))
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()
cache = {}
def emb(texts):
    new = [t for t in texts if t not in cache]
    if new:
        with torch.no_grad():
            e = model.encode_text(clip.tokenize(new, truncate=True)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        cache.update(zip(new, e.numpy()))
    return np.stack([cache[t] for t in texts])

rows = []
for inst, d in para.items():
    if not MIRROR.search(inst):
        continue
    o = emb([inst])[0]
    r = {"inst": inst, "cls": d["class"], "mirror": float(o @ emb([mirror(inst)])[0])}
    for lv in ("L1", "L2", "L3"):
        if d[lv]:
            r[lv] = float((emb(d[lv]) @ o).mean())
    rows.append(r)

print(f"left/right を含む指示: {len(rows)} / {len(para)}")
m = np.array([r["mirror"] for r in rows])
print(f"cos(元, 反転)        mean {m.mean():.4f}  min {m.min():.4f}  max {m.max():.4f}")
for lv in ("L1", "L2", "L3"):
    v = np.array([r[lv] for r in rows if lv in r])
    w = np.array([r["mirror"] > r[lv] for r in rows if lv in r])
    print(f"cos(元, {lv}言い換え)  mean {v.mean():.4f}   反転の方が近い指示: {w.mean()*100:.1f}% ({w.sum()}/{len(w)})")

# 参考: 別の指示同士(同じ向き/逆向き)の類似度
L = [r["inst"] for r in rows if r["cls"] == "L"]
R = [r["inst"] for r in rows if r["cls"] == "R"]
eL, eR = emb(L), emb(R)
same = (eL @ eL.T)[np.triu_indices(len(L), 1)]
diff = (eL @ eR.T).ravel()
print(f"参考: 別の左折指示同士 mean {same.mean():.4f} / 左折×右折 mean {diff.mean():.4f}  (L{len(L)} R{len(R)})")

print("\n反転が最も近い例:")
for r in sorted(rows, key=lambda r: -r["mirror"])[:5]:
    print(f"  {r['mirror']:.4f}  {r['inst']!r} -> {mirror(r['inst'])!r}")
