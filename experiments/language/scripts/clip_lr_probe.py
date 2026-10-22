#!/usr/bin/env python3
"""CLIP ViT-B/32 のテキスト埋め込みに left/right の情報が線形に読み出せる形で残っているかを測る。

clip_lr_check.py の続き。コサイン類似度が高くても、左右の違いが一定の向きに乗っていれば
線形分類器で読み出せる。left/right を含む左折・右折の指示(paraphrases_v3.json の class)で
ロジスティック回帰を学習し、学習に使っていない指示で次を見る(指示単位の10分割交差検証)。

  A: 元の指示だけで学習 → 元の指示の正解率 / 反転文で予測も反転した割合 / L1-L3 言い換えの正解率
     (OmniVLA-edge の学習と同じく、指示と向きの対応を見るだけの条件)
  B: 元の指示 + 反転文(ラベルも反転)で学習 → 同じ3つ
     (left/right の語だけで向きを決めるよう強制した条件。ここで当たれば情報は埋め込みに残っている)

使い方(GPU不要、CLIP の入った環境で):
    python scripts/clip_lr_probe.py data/paraphrases_v3.json > data/clip_lr_probe.txt
"""
import json, re, sys
import numpy as np, torch, clip

MIRROR = re.compile(r"\b(left|right)\b", re.I)
def mirror(s):
    return MIRROR.sub(lambda m: "right" if m.group(1).lower() == "left" else "left", s)

para = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/paraphrases_v3.json"))
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()
def emb(texts):
    with torch.no_grad():
        e = model.encode_text(clip.tokenize(texts, truncate=True)).float()
    return (e / e.norm(dim=-1, keepdim=True)).numpy()

items = [(k, 0 if d["class"] == "L" else 1) for k, d in para.items()
         if d["class"] in ("L", "R") and MIRROR.search(k)]
insts = [k for k, _ in items]
y = np.array([c for _, c in items])
X = emb(insts)
Xm = emb([mirror(k) for k in insts])
P = [(i, t) for i, k in enumerate(insts) for lv in ("L1", "L2", "L3") for t in para[k][lv]]
Xp = emb([t for _, t in P])
p_idx = np.array([i for i, _ in P])
print(f"指示: 左折 {(y == 0).sum()} / 右折 {(y == 1).sum()}  言い換え {len(P)}本")


def fit(Xtr, ytr, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64)
    w = torch.zeros(Xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + lam * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    w_, b_ = w.detach().numpy(), b.item()
    return lambda Z: (((Z - mu) / sd) @ w_ + b_ > 0).astype(int)


fold = np.random.default_rng(0).permutation(len(y)) % 10
for lam in (1e-3, 1e-2, 1e-1):
    print(f"\n== L2 係数 {lam}")
    for name, aug in (("A 元の指示だけ", False), ("B 元+反転", True)):
        o = m = p = 0
        for f in range(10):
            tr, te = fold != f, fold == f
            Xtr, ytr = X[tr], y[tr]
            if aug:
                Xtr, ytr = np.vstack([Xtr, Xm[tr]]), np.concatenate([ytr, 1 - ytr])
            pred = fit(Xtr, ytr, lam)
            o += (pred(X[te]) == y[te]).sum()
            m += (pred(Xm[te]) == 1 - y[te]).sum()
            pt = te[p_idx]
            p += (pred(Xp[pt]) == y[p_idx[pt]]).sum()
        n = len(y)
        print(f"  {name:10s} 元の指示 {o / n * 100:5.1f}%   反転で予測も反転 {m / n * 100:5.1f}%   "
              f"言い換え {p / len(P) * 100:5.1f}%")
