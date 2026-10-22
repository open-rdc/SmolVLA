"""評価案4(第1段階): 学習済み OmniVLA-edge の FiLM の係数(γ, β)で left/right の差が拡大されているかを測る。

FiLM の係数 p = film_param_generator(CLIP のベクトル) は指示文だけで決まる(画像は使わない)。
CLIP のベクトルは学習時と同じく OpenAI clip の ViT-B/32 encode_text(正規化しない)。
指標(ユークリッド距離):
  ratio = 中央値|x(s) - x(入れ替えた s)| / 平均|x(a) - x(b)|  (a, b は無関係な指示の組)
  amp   = FiLM 係数での ratio / CLIP での ratio。1 より大きければ学習で left/right の差が拡大されている
基準として同じ形のランダムな線形層でも測る。
使い方: python film_lr_test.py <名前=重み.pth> ...
"""
import glob
import random
import re
import sys

import clip
import torch

exec(open(__import__("pathlib").Path(__file__).resolve().parent / "encoder_lr_test_v2.py")
     .read().split("\ndef mean_pool")[0].split("import torch\n", 1)[1])  # mirror / load_sets / split_lr を流用

torch.manual_seed(0)
model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()
cache = {}


def enc(ts):
    todo = [t for t in dict.fromkeys(ts) if t not in cache]
    with torch.no_grad():
        for i in range(0, len(todo), 64):
            v = model.encode_text(clip.tokenize(todo[i:i + 64], truncate=True)).float()
            cache.update(zip(todo[i:i + 64], v))
    return torch.stack([cache[t] for t in ts])


def ratio(x_s, x_m, x_a, x_b):
    return ((x_s - x_m).norm(dim=1).median() / (x_a - x_b).norm(dim=1).mean()).item()


sets = load_sets()
groups = {}
for cam, prompts in sets.items():
    rng = random.Random(0)
    pairs = [tuple(rng.sample(prompts, 2)) for _ in range(2000)]
    for kind, g in split_lr(prompts).items():
        groups[f"{cam}_{kind}"] = (g, pairs)

layers = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    layers[name] = (sd["film_model.film_param_generator.weight"].float(), sd["film_model.film_param_generator.bias"].float())
W0 = next(iter(layers.values()))[0]
layers["random"] = (torch.randn_like(W0) * W0.std(), torch.zeros(W0.shape[0]))

print("group\tn\tclip_ratio\t" + "\t".join(f"{k}_ratio\t{k}_amp" for k in layers))
for gname, (g, pairs) in groups.items():
    e_s, e_m = enc(g), enc([mirror(s) for s in g])
    e_a, e_b = enc([a for a, _ in pairs]), enc([b for _, b in pairs])
    r0 = ratio(e_s, e_m, e_a, e_b)
    row = [gname, str(len(g)), f"{r0:.3f}"]
    for k, (W, b) in layers.items():
        f = lambda e: e @ W.T + b
        r = ratio(f(e_s), f(e_m), f(e_a), f(e_b))
        row += [f"{r:.3f}", f"{r / r0:.2f}"]
    print("\t".join(row), flush=True)
