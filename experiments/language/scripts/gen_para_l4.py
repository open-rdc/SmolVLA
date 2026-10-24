# -*- coding: utf-8 -*-
"""未知語だけで作る言い換え L4 の生成。

L3 は gen_para2.py の OOV 辞書で作ったが，言い換え拡張の学習用データ(paraphrases_train.json)も
同じ OOV 辞書から作られていたため，拡張ありのモデルにとって L3 の単語は学習中に見た単語だった
（2026-10-11 判明．L3 の 1,026 本すべてで，元の語彙に無い語が学習用の言い換えに出現する）．
L4 は，元の語彙・学習用の言い換え・評価用の L1〜L3 のどれにも出てこない語だけで置換する．

手順は gen_para2.gen_L3 と同じ（置換 2 箇所以上，旋回動詞も置換，left/right の並びを保存）．
置換候補のうち禁止語を含むものは自動で外し，外した候補を表示する．

使い方:
    WORK=data python scripts/gen_para_l4.py data/paraphrases_train.json   # → data/paraphrases_l4.json
"""
import json, re, random, collections, os, sys

SP = os.environ.get('WORK', '.').rstrip('/') + '/'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_para2 import OOV, TURNPAT, VERBKEYS, DC, VN, norm, dirs  # noqa: E402

TRAIN_PARA = sys.argv[1] if len(sys.argv) > 1 else SP + 'paraphrases_train.json'
words_of = lambda s: re.findall(r'[a-z]+', s.lower())

def flat(v):
    if isinstance(v, str): return [v]
    if isinstance(v, list): return [x for y in v for x in flat(y)]
    if isinstance(v, dict): return [x for y in v.values() for x in flat(y)]
    return []

# 禁止語 = 元の語彙 ∪ 学習用の言い換え ∪ 評価用の L1〜L3
train_words = set(json.load(open(SP + 'train_words.json')))
vocab_words = {w for s in json.load(open(SP + 'train_vocab_full.json')) for w in words_of(s)}
aug_words = {w for s in flat(json.load(open(TRAIN_PARA))) for w in words_of(s)}
ev = json.load(open(SP + 'paraphrases_v3.json'))
ev_words = {w for v in ev.values() for lv in ('L1', 'L2', 'L3') for s in v[lv] for w in words_of(s)}
BANNED = train_words | vocab_words | aug_words | ev_words

# 置換候補（OOV と同じキー）．禁止語を含む候補は下で自動的に外す
CAND = {
 'avenue': ['parkway', 'causeway'], 'street': ['carriageway', 'byway'], 'road': ['carriageway', 'byway', 'motorway'],
 'path': ['footpath', 'trail', 'pathway', 'track'], 'walkway': ['footway', 'boardwalk', 'pavement'],
 'promenade': ['boardwalk', 'mall'], 'plaza': ['piazza', 'forecourt', 'courtyard'], 'square': ['piazza', 'courtyard'],
 'area': ['zone', 'district', 'section'], 'lot': ['carpark', 'compound'],
 'hedge': ['hedgerow', 'shrubbery', 'bushes'], 'hedges': ['hedgerows', 'bushes'], 'trees': ['woods', 'treeline', 'canopy'],
 'tree': ['oak', 'elm'], 'tree-lined': ['leafy', 'shaded'], 'lined': ['bordered', 'fringed'],
 'lawn': ['sward', 'yard'], 'grass': ['sward', 'pasture'], 'grassy': ['verdant', 'leafy'], 'green': ['verdant', 'emerald'],
 'building': ['structure', 'hall', 'premises'], 'buildings': ['structures', 'halls'], 'glass': ['glassy', 'mirrored', 'windowed'],
 'white': ['ivory', 'snowy', 'pale'], 'red': ['crimson', 'scarlet', 'maroon'], 'brick': ['masonry', 'stone'],
 'bicycle': ['bike'], 'racks': ['stands', 'holders'], 'parking': ['carpark', 'garage'],
 'corridor': ['hallway', 'passageway'], 'gate': ['gateway', 'archway', 'entryway'],
 'statue': ['sculpture', 'bust'], 'colonnade': ['pillars', 'columns', 'cloister'], 'park': ['gardens', 'grounds'],
 'pylon': ['cone', 'post', 'pole'], 'corner': ['junction', 'turning'], 'front': ['facade'],
 'open': ['spacious', 'broad'], 'paved': ['surfaced', 'asphalt', 'concrete'], 'tiled': ['slabbed', 'mosaic'],
 'entrance': ['entryway', 'gateway', 'access'], 'apartment': ['housing', 'residential'], 'towers': ['highrises', 'skyscrapers'],
 'trash can': ['rubbish bin', 'dustbin', 'litter bin'], 'covered': ['roofed', 'sheltered'],
 'both sides': ['each side', 'two sides'], 'ahead': ['forward', 'onward'],
 'continue': ['keep going', 'go on'], 'head': ['travel', 'make your way'], 'proceed': ['advance', 'travel'],
 'follow': ['track', 'stick to', 'shadow'], 'following': ['tracking', 'shadowing'], 'move': ['advance', 'travel'],
 'edge': ['border', 'margin', 'fringe'], 'curves': ['arcs', 'sweeps', 'winds'], 'curving': ['arcing', 'sweeping', 'winding'],
 'across': ['traversing', 'spanning'], 'along': ['parallel to'], 'beside': ['adjacent to', 'next to'],
 'toward': ['in the direction of', 'facing'], 'between': ['amid', 'amidst'], 'past': ['after'],
 'through': ['via', 'throughout'], 'around': ['circling', 'about'], 'far': ['distant', 'furthest'],
 'bear': ['drift', 'lean'],
}
TURN_CAND = {'L': ['wheel left', 'steer left', 'arc left', 'angle left', 'slant left', 'drift left'],
             'R': ['wheel right', 'steer right', 'arc right', 'angle right', 'slant right', 'drift right']}

ok = lambda p: not [w for w in words_of(p) if w in BANNED and w not in ('left', 'right')]
dropped = sorted({p for v in list(CAND.values()) + list(TURN_CAND.values()) for p in v if not ok(p)})
OOV4 = {k: [p for p in v if ok(p)] for k, v in CAND.items()}
OOV4 = {k: v for k, v in OOV4.items() if v}
TURNV4 = {c: [p for p in v if ok(p)] for c, v in TURN_CAND.items()}
missing = sorted(set(OOV) - set(OOV4))


def sub_map(s, M, rng):
    t = s; used = 0; vdone = False
    for k in sorted(M, key=len, reverse=True):
        pat = re.compile(r'\b' + re.escape(k) + r'\b', re.I)
        if vdone and k in VERBKEYS: continue
        if pat.search(t):
            t = pat.sub(rng.choice(M[k]), t, count=1); used += 1
            if k in VERBKEYS: vdone = True
    return t, used


def gen_L4(s, cls, rng, k=6):
    out = set()
    for _ in range(120):
        t, u = sub_map(s, OOV4, rng)
        if cls in 'LR' and TURNPAT.search(t):
            t = TURNPAT.sub(rng.choice(TURNV4[cls]), t, count=1); u += 1
        if u >= 2: out.add(re.sub(r'\s+', ' ', t).strip())
        if len(out) >= k: break
    return list(out)


if __name__ == '__main__':
    print(f"禁止語 {len(BANNED)} 語（元の語彙 {len(train_words | vocab_words)} / 学習用の言い換え {len(aug_words)} / L1〜L3 {len(ev_words)}）")
    print(f"禁止語を含むため外した候補 {len(dropped)}: {dropped}")
    print(f"置換先が残らなかったキー {len(missing)}: {missing}")
    rng = random.Random(11); NEW = {}; st = collections.Counter(); seen = set()
    for orig, v in ev.items():
        cls = DC[orig]['data_cls']; d0 = dirs(orig); keep = []
        for c in gen_L4(orig, cls, rng):
            if norm(c) in VN: st['leak'] += 1; continue
            if dirs(c) != d0: st['dir'] += 1; continue
            novel = [w for w in words_of(c) if w not in (train_words | vocab_words)]
            if not novel: st['nooov'] += 1; continue
            assert not [w for w in novel if w in aug_words | ev_words], c
            if norm(c) in seen: st['dup'] += 1; continue
            seen.add(norm(c)); keep.append(c)
        NEW[orig] = {'class': v['class'], 'L4': keep[:5]}
    json.dump(NEW, open(SP + 'paraphrases_l4.json', 'w'), ensure_ascii=False, indent=1)
    n = sum(len(v['L4']) for v in NEW.values())
    print(f"L4 {n} 本 / 指示 {sum(1 for v in NEW.values() if v['L4'])} 種（全 {len(NEW)} 種）  除外 {dict(st)}")
    for c in 'LRS':
        print(f"  {c}: {sum(len(v['L4']) for v in NEW.values() if v['class'] == c)} 本")
