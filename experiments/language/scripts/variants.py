"""指示文を「目印だけ」「方向だけ」に作り替える。
目印だけ: 方向の語(left/right/straight/ahead/around/back/U-turn)と、動きを表す動詞(turn, go, head, move ...)を消す。
方向だけ: 文に出てくる方向の語だけで作った短い指示(turn around / turn left / turn right / go straight)。
"""
import re

DIR = r"(left|right|straight|ahead|around|back|u-turn)"
VERB = r"(turn|turns|turning|go|goes|head|heading|move|continue|follow|make|retrace|travel|proceed|enter|curving|curves|circling|bear|veer|keep)"


def landmark_only(s):
    t = re.sub(r"\b(on|to|toward|towards|into) (your|the) " + DIR + r"\b", " ", s, flags=re.I)
    t = re.sub(r"\bmake a u-turn\b", " ", t, flags=re.I)
    t = re.sub(r"\b" + DIR + r"\b", " ", t, flags=re.I)
    t = re.sub(r"\b" + VERB + r"\b", " ", t, flags=re.I)
    t = re.sub(r"\b(a|by|then|and|as it|it)\b(?=\s*(,|\.|$))", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    return t if re.search(r"[a-z]{3,}", t, re.I) and len(t.split()) >= 2 else ""


def direction_only(s):
    low = s.lower()
    if re.search(r"\bturn(ing)? around\b|\bu-turn\b|\bcircling\b", low):
        return "turn around"
    has = [w for w in ("left", "right") if re.search(r"\b" + w + r"\b", low)]
    if len(has) == 1:
        return f"turn {has[0]}"
    if not has and re.search(r"\b(straight|ahead)\b", low):
        return "go straight"
    return ""
