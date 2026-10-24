#!/usr/bin/env python3
"""言語汎化評価の解析。issue #6 に載せた数値をすべてこのスクリプトで再現する。

使い方:
    python analyze.py --work <結果ディレクトリ>

必要なファイル（このリポジトリの experiments/language/data/ に同梱）:
    theta.json / v3_frames.json / paraphrases_v3.json / meta_v3.json / v3_results.json（.gz 可）
（初回版の p1_results.json / p1b_results.json があれば 111本版の数値も出す）
"""
import json, os, argparse, collections
from math import erfc, sqrt
import numpy as np

def load(w, n):
    """<work>/<n> を読む。無ければ <work>/<n>.gz を試す（v3_results.json は .gz で配布）。"""
    p = os.path.join(w, n)
    if os.path.exists(p):
        return json.load(open(p))
    if os.path.exists(p + '.gz'):
        import gzip
        with gzip.open(p + '.gz', 'rt', encoding='utf-8') as f:
            return json.load(f)
    return None

def wilcoxon(a, b):
    """対応のある Wilcoxon 符号順位検定（正規近似）。戻り値 (n, z, p, どちらが小さいか)"""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]; n = len(d)
    if n < 10: return n, float('nan'), 1.0, '判定不能'
    r = np.argsort(np.argsort(np.abs(d))) + 1
    Wp, Wm = r[d > 0].sum(), r[d < 0].sum()
    W = min(Wp, Wm)
    z = (W - n*(n+1)/4) / sqrt(n*(n+1)*(2*n+1)/24)
    p = erfc(abs(z)/sqrt(2))
    return n, z, p, ('同等' if p > 0.05 else ('B' if Wp > Wm else 'A'))

def sign_test(ok_a, ok_b):
    """対応のある符号検定（正誤の入れ替わりだけを数える）"""
    from math import comb
    win = sum(1 for x, y in zip(ok_a, ok_b) if x and not y)
    lose = sum(1 for x, y in zip(ok_a, ok_b) if y and not x)
    n = win + lose
    if n == 0: return win, lose, 1.0
    p = min(1.0, sum(comb(n, i) for i in range(min(win, lose)+1)) / 2**n * 2)
    return win, lose, p


# ============================ クラスタブートストラップ ============================
# データは入れ子構造（1つの指示から複数の言い換え、複数フレームが同じ指示を共有）。
# 標本を独立扱いすると検出力を過大評価するので、指示を単位にリサンプリングする。

def cluster_bootstrap_ratio(num_by_cluster, den_by_cluster, B=4000, seed=0):
    """クラスタごとに (差の和, 標本数) を渡す。平均差をブートストラップする。
    純Pythonのループを使わずベクトル化しているのでこの規模でも一瞬で終わる。"""
    num = np.asarray(num_by_cluster, dtype=np.float64)
    den = np.asarray(den_by_cluster, dtype=np.float64)
    point = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(B, len(num)))
    vals = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(np.mean(vals > 0))


def verdict(lo, hi):
    if lo > 0:  return '有意'
    if hi < 0:  return '有意(逆)'
    return '非有意'


def cluster_bootstrap_corr_diff(pred_a, pred_b, true, cluster, B=4000, seed=0):
    """相関 r の差 (a - b) を指示単位のクラスタブートストラップで区間推定する。
    r は cluster_bootstrap_ratio の「和の比」の形に書けないので、
    クラスタを再標本化して毎回 r を引き直す。"""
    a = np.asarray(pred_a, dtype=np.float64); b = np.asarray(pred_b, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64); cl = np.asarray(cluster)
    groups = [np.where(cl == c)[0] for c in sorted(set(cl.tolist()))]
    r = lambda x, i: np.corrcoef(x[i], t[i])[0, 1]
    allidx = np.arange(len(t))
    point = r(a, allidx) - r(b, allidx)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for j in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        vals[j] = r(a, idx) - r(b, idx)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)

def bootstrap_section(R, frames, TRUE_PSI, TRUE_CLS, B=4000):
    key = lambda r: (r['gidx'], r['seed'], r['level'], r['vid'])
    print("\n" + "=" * 78)
    print("指示単位クラスタブートストラップ（95%%信頼区間, B=%d）" % B)
    print("正の差 = 検索置換の方が誤差が大きい = 直接投入が良い。CIが0を挟まなければ有意")
    print("=" * 78)

    def agg(tasks, vals):
        acc = collections.defaultdict(lambda: [0.0, 0])
        for t, v in zip(tasks, vals):
            r = acc[t]; r[0] += float(v); r[1] += 1
        cl = sorted(acc)
        return (np.array([acc[c][0] for c in cl]), np.array([acc[c][1] for c in cl]), len(cl))

    for lv in ['L1', 'L2', 'L3']:
        D = {c: {key(r): r for r in R if r['cond'] == '%s_%s' % (c, lv)} for c in ['U','Remb','Rtfidf']}
        ks = [k for k in D['U'] if k in D['Remb'] and k in D['Rtfidf']]
        if not ks: continue
        for turn_only, tag in [(False, '全体'), (True, '旋回のみ')]:
            sel = [k for k in ks if (not turn_only) or TRUE_CLS(D['U'][k]) in 'LR']
            if not sel: continue
            tasks = [D['U'][k]['task'] for k in sel]
            tru = np.array([TRUE_PSI(D['U'][k]) for k in sel])
            err = {c: np.abs(np.array([D[c][k]['pred_psi'] for k in sel]) - tru) for c in D}
            print("\n[%s / %s] クラスタ %d指示 / 標本 %d" % (lv, tag, len(set(tasks)), len(sel)))
            print("  Ψ MAE  直接投入 %.4f  検索置換(埋め込み) %.4f  検索置換(TF-IDF) %.4f"
                  % (err['U'].mean(), err['Remb'].mean(), err['Rtfidf'].mean()))
            for c, lab, sd in [('Remb','検索置換(埋め込み)',0), ('Rtfidf','検索置換(TF-IDF)',1)]:
                num, den, _ = agg(tasks, err[c] - err['U'])
                pt, lo, hi, pr = cluster_bootstrap_ratio(num, den, B, sd)
                print("  %s − 直接投入 = %+.4f  95%%CI [%+.4f, %+.4f]  → %s" % (lab, pt, lo, hi, verdict(lo,hi)))
        tasks = [D['U'][k]['task'] for k in ks]
        ok = {c: np.array([1.0 if D[c][k]['pred_cls'] == TRUE_CLS(D[c][k]) else 0.0 for k in ks]) for c in D}
        print("  [行動クラス正答率] 直接投入 %.1f%%  埋め込み %.1f%%  TF-IDF %.1f%%"
              % (ok['U'].mean()*100, ok['Remb'].mean()*100, ok['Rtfidf'].mean()*100))
        for c, lab, sd in [('Remb','埋め込み',2), ('Rtfidf','TF-IDF',3)]:
            num, den, _ = agg(tasks, ok['U'] - ok[c])
            pt, lo, hi, pr = cluster_bootstrap_ratio(num, den, B, sd)
            print("    直接投入 − %s = %+.1fpt  95%%CI [%+.1f, %+.1f]pt  → %s" % (lab, pt*100, lo*100, hi*100, verdict(lo,hi)))
        if lv == 'L3':
            tru = np.array([TRUE_PSI(D['U'][k]) for k in ks])
            eu = np.abs(np.array([D['U'][k]['pred_psi'] for k in ks]) - tru)
            ee = np.abs(np.array([D['Remb'][k]['pred_psi'] for k in ks]) - tru)
            print("\n  [平均と中位数の比較 / L3 n=%d]" % len(ks))
            print("    MAE(平均)   直接投入 %.4f  埋め込み %.4f  → 優位 %s"
                  % (eu.mean(), ee.mean(), '直接投入' if eu.mean()<ee.mean() else '埋め込み'))
            print("    中位数       直接投入 %.4f  埋め込み %.4f  → 優位 %s"
                  % (np.median(eu), np.median(ee), '直接投入' if np.median(eu)<np.median(ee) else '埋め込み'))
            print("    直接投入が勝つ標本の割合 %.1f%%" % (np.mean(eu<ee)*100))
            print("    90パーセンタイル 直接投入 %.4f  埋め込み %.4f" % (np.percentile(eu,90), np.percentile(ee,90)))

    Du = {(r['gidx'], r['seed']): r for r in R if r['cond']=='U_L3' and r['vid']==0}
    De = {(r['gidx'], r['seed']): r for r in R if r['cond']=='empty'}
    ks = [k for k in Du if k in De]
    print("\n[指示なし（下界）との比較 / L3の第1変種]")
    for turn_only, tag, sd in [(False,'全体',4), (True,'旋回のみ',5)]:
        sel = [k for k in ks if (not turn_only) or TRUE_CLS(Du[k]) in 'LR']
        if not sel: continue
        tasks = [Du[k]['task'] for k in sel]
        au = np.array([1.0 if Du[k]['pred_cls']==TRUE_CLS(Du[k]) else 0.0 for k in sel])
        ae = np.array([1.0 if De[k]['pred_cls']==TRUE_CLS(De[k]) else 0.0 for k in sel])
        num, den, _ = agg(tasks, au - ae)
        pt, lo, hi, pr = cluster_bootstrap_ratio(num, den, B, sd)
        print("  %s: 直接投入 %.1f%% vs 指示なし %.1f%%  差 %+.1fpt  95%%CI [%+.1f, %+.1f]pt  → %s"
              % (tag, au.mean()*100, ae.mean()*100, pt*100, lo*100, hi*100, verdict(lo,hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='.')
    ap.add_argument('--boot', type=int, default=4000, help='ブートストラップ反復数。0で省略')
    ap.add_argument('--results', default=None, help='結果JSONのファイル名（既定は v2_results.json → v3_results.json を順に探す）')
    a = ap.parse_args(); W = a.work
    th = load(W, 'theta.json'); N, TH = th['N'], th['theta']
    fj = load(W, 'v3_frames.json') or load(W, 'p1_frames.json')
    frames = {f['gidx']: f for f in fj}
    # gidx はデータセットごとの通し番号なので衝突しうる。レコードが持つ真値を優先する。
    def TRUE_PSI(r): return r.get('true_psi', frames[r['gidx']]['psi'])
    def TRUE_CLS(r): return r.get('true_cls', frames[r['gidx']]['cls'])
    print(f"  （真値はレコード内の true_psi/true_cls を使用。gidxの衝突: {len(fj)-len(frames)}件）")
    print(f"設定: N={N} ({N*0.2:.1f}秒) / θ={TH:.3f} rad ({np.degrees(TH):.1f}°) / 評価フレーム {len(frames)}")

    # ---------- 拡張版（327本） ----------
    R = load(W, a.results) if a.results else None
    if R is None: R = load(W, 'v3_results.json') or load(W, 'v2_results.json')
    M = load(W, 'meta_v3.json') or load(W, 'meta_v2.json')
    if R is None:
        print("結果JSONが無い（v3_results.json / v2_results.json / --results）"); return
    V = load(W, 'paraphrases_v3.json') or load(W, 'paraphrases_v2.json')
    n_para = sum(len(V[o][lv]) for o in V for lv in ['L1','L2','L3'])
    print(f"言い換え {n_para}本 / レコード {len(R)}\n")

    acc = lambda rs: np.mean([r['pred_cls'] == r['true_cls'] for r in rs]) if rs else float('nan')
    def percls(rs, k):
        s = [r['pred_cls'] == r['true_cls'] for r in rs if r['true_cls'] == k]
        return np.mean(s) if s else float('nan')
    sub = lambda c, lv=None: [r for r in R if r['cond'] == c and (lv is None or r['level'] == lv)]

    print("=== 条件別 行動クラス正答率 ===")
    print(f"{'条件':<18} {'n':>6} {'全体':>7} {'L':>7} {'R':>7} {'S':>7}")
    def line(lab, rs):
        if not rs: return
        print(f"{lab:<18} {len(rs):>6} {acc(rs)*100:>6.1f}% " +
              " ".join(f"{percls(rs,k)*100:>6.1f}%" if percls(rs,k)==percls(rs,k) else f"{'-':>7}" for k in 'LRS'))
    line('元指示(上界)', sub('orig'))
    for lv in ['L1','L2','L3']:
        line(f'U {lv}', sub(f'U_{lv}', lv))
        line(f'  R-emb {lv}', sub(f'Remb_{lv}', lv))
        line(f'  R-tfidf {lv}', sub(f'Rtfidf_{lv}', lv))
    line('指示なし(下界)', sub('empty')); line('方向反転', sub('mirror'))

    # 方向反転の指示追従
    mir = sub('mirror'); og = {(r['gidx'], r['seed']): r for r in sub('orig')}
    if mir:
        flip = {'L':'R','R':'L'}
        f_rate = np.mean([r['pred_cls'] == flip[r['true_cls']] for r in mir])
        sgn = np.mean([np.sign(r['pred_psi']) != np.sign(og[(r['gidx'],r['seed'])]['pred_psi']) for r in mir])
        dp = [r['pred_psi'] - og[(r['gidx'],r['seed'])]['pred_psi'] for r in mir]
        print(f"\n=== 方向語反転 (n={len(mir)}) ===")
        print(f"  反転側クラスに切替 {f_rate*100:.1f}% / 元のまま {np.mean([r['pred_cls']==r['true_cls'] for r in mir])*100:.1f}%"
              f" / 直進化 {np.mean([r['pred_cls']=='S' for r in mir])*100:.1f}%")
        print(f"  予測Ψの符号が反転 {sgn*100:.1f}% / ΔΨ 平均 {np.mean(dp):+.4f} rad")

    print("\n=== 並進 dx は指示に依存しないか（言語が旋回だけをゲートしている証拠）===")
    for c, lv in [('U_L1','L1'),('U_L2','L2'),('U_L3','L3'),('empty',None),('mirror',None)]:
        rs = sub(c, lv)
        if not rs: continue
        d  = [abs(r['pred_dx']  - og[(r['gidx'],r['seed'])]['pred_dx'])  for r in rs]
        dp = [abs(r['pred_psi'] - og[(r['gidx'],r['seed'])]['pred_psi']) for r in rs]
        print(f"  {c:<10} |Δdx| {np.mean(d):.4f} m   |ΔΨ| {np.mean(dp):.4f} rad   比 {np.mean(dp)/np.mean(d):.2f}")

    # ---------- 同一土俵の比較 ----------
    key = lambda r: (r['gidx'], r['seed'], r['level'], r['vid'])
    for lv in ['L1','L2','L3']:
        U  = {key(r): r for r in sub(f'U_{lv}', lv)}
        Ee = {key(r): r for r in sub(f'Remb_{lv}', lv)}
        Tt = {key(r): r for r in sub(f'Rtfidf_{lv}', lv)}
        ks = sorted(set(U) & set(Ee) & set(Tt))
        if not ks: continue
        true = np.array([TRUE_PSI(U[k]) for k in ks])
        turn = np.array([TRUE_CLS(U[k]) in 'LR' for k in ks])
        print(f"\n=== {lv}: 同一土俵の比較 (n={len(ks)}) ===")
        print(f"{'条件':<12} {'MAE':>8} {'r':>7} | {'旋回MAE':>9} {'旋回r':>7}")
        E = {}; X = {}
        for lab, d in [('U', U), ('R-emb', Ee), ('R-tfidf', Tt)]:
            x = np.array([d[k]['pred_psi'] for k in ks]); e = np.abs(x - true); E[lab] = e; X[lab] = x
            print(f"{lab:<12} {e.mean():>8.4f} {np.corrcoef(x,true)[0,1]:>7.3f} | "
                  f"{e[turn].mean():>9.4f} {np.corrcoef(x[turn],true[turn])[0,1]:>7.3f}")
        if a.boot:
            task = np.array([U[k]['task'] for k in ks])
            print(f"  [相関 r の差 / 指示単位クラスタブートストラップ B={a.boot}]")
            for tag, m, sd in [('全体', np.ones(len(ks), bool), 6), ('旋回のみ', turn, 7)]:
                for bl, jp in [('R-emb', '埋め込み'), ('R-tfidf', 'TF-IDF')]:
                    pt, lo, hi = cluster_bootstrap_corr_diff(
                        X['U'][m], X[bl][m], true[m], task[m], a.boot, sd)
                    print(f"    {tag:<8} 直接投入 − 検索置換({jp}): {pt:+.3f} "
                          f"CI [{lo:+.3f}, {hi:+.3f}] {verdict(lo, hi)}")
        for A, B in [('U','R-emb'), ('U','R-tfidf')]:
            n, z, p, w = wilcoxon(E[A], E[B])
            print(f"  [Wilcoxon] {A} vs {B}: n={n} z={z:.2f} p={p:.2e} → "
                  f"{'同等' if w=='同等' else '誤差が小さいのは '+(A if w=='A' else B)}")
        n, z, p, w = wilcoxon(E['U'][turn], E['R-emb'][turn])
        print(f"  [Wilcoxon] U(旋回) vs R-emb(旋回): n={n} z={z:.2f} p={p:.2e} → "
              f"{'同等' if w=='同等' else '誤差が小さいのは '+('U' if w=='A' else 'R-emb')}")
        oa = [U[k]['pred_cls']==TRUE_CLS(U[k]) for k in ks]
        ob = [Ee[k]['pred_cls']==TRUE_CLS(Ee[k]) for k in ks]
        win, lose, p = sign_test(oa, ob)
        print(f"  [符号検定/3クラス] U勝ち {win} R-emb勝ち {lose} p={p:.2e}")

    # ---------- 距離ごと ----------
    mk = lambda r: f"{r['task']}||{r['level']}||{r['vid']}"
    rows = []
    Uall  = {key(r): r for r in R if r['cond'].startswith('U_')}
    Eall  = {key(r): r for r in R if r['cond'].startswith('Remb_')}
    for k, r in Uall.items():
        if k not in Eall: continue
        m = M[mk(r)]
        rows.append(dict(d=m['d_emb'], lv=r['level'], turn=TRUE_CLS(r) in 'LR',
                         true=TRUE_PSI(r), u=r['pred_psi'], e=Eall[k]['pred_psi'],
                         retcls=m['ret_emb_cls'], intended=m['intended']))
    print("\n=== 埋め込み距離の分布 ===")
    for lv in ['L1','L2','L3']:
        dd = np.array([x['d'] for x in rows if x['lv']==lv])
        print(f"  {lv}: n={len(dd):5d} 中央値 {np.median(dd):.4f} 10-90%tile {np.percentile(dd,10):.4f}-{np.percentile(dd,90):.4f}")
    tr = [x for x in rows if x['turn']]
    dt = np.array([x['d'] for x in tr]); q = np.quantile(dt, np.linspace(0,1,6))
    print("\n=== 距離5分位ごとのΨ誤差（旋回フレームのみ）===")
    print(f"{'距離帯':<16} {'n':>6} {'U':>8} {'R-emb':>8} {'差':>8} {'比':>6}")
    for i in range(5):
        lo, hi = q[i], q[i+1]
        g = [x for x in tr if x['d'] >= lo and (x['d'] < hi or i == 4)]
        if not g: continue
        ue = np.mean([abs(x['u']-x['true']) for x in g]); ee = np.mean([abs(x['e']-x['true']) for x in g])
        print(f"{lo:.3f}-{hi:.3f}    {len(g):>6} {ue:>8.4f} {ee:>8.4f} {ee-ue:>+8.4f} {ee/ue:>5.2f}x")
    print("\n=== 検索のクラス誤り率（引いた既知指示のクラスが正解と違う割合）===")
    for lv in ['L1','L2','L3']:
        g = [x for x in rows if x['lv']==lv]
        print(f"  {lv}: {np.mean([x['retcls']!=x['intended'] for x in g])*100:.1f}%")
    if a.boot:
        bootstrap_section(R, frames, TRUE_PSI, TRUE_CLS, a.boot)
    print("\n[注意] 埋め込み距離は指示の種類と相関する（旋回指示は短く距離が大きい）。"
          "\n       全フレームで距離5分位に切ると第5分位が旋回フレームに偏るため、"
          "\n       距離を独立変数として扱う際は上の旋回限定の集計を使う。")

if __name__ == '__main__':
    main()
