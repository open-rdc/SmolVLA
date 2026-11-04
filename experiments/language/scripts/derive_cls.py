"""指示のクラスを文字列推定ではなく実データのΨから決める。
各指示の全フレームのΨ分布を見て、明確なクラスを持つものだけ採用する。"""
import pyarrow.parquet as pq,glob,numpy as np,json,re,collections,os
D=os.environ.get('SMOLVLA_DATA','../../training/data').rstrip('/')+'/'   # LeRobotデータセット置き場
SP=os.environ.get('WORK','.').rstrip('/')+'/'          # 入出力ディレクトリ。既定はカレント
th=json.load(open(SP+'theta.json')); N=th['N']; TH=th['theta']
tcs=['orne_nav_tc','orne_random_tc','orne_junction_tc','orne_linestop_tc','orne_recovery_tc','orne_recovery2_tc']
psi=collections.defaultdict(list); frames=collections.defaultdict(list)
for d in tcs:
    tasks=[str(x) for x in pq.read_table(f'{D}{d}/meta/tasks.parquet').to_pydict()['task']]
    for f in sorted(glob.glob(f'{D}{d}/data/**/*.parquet',recursive=True)):
        t=pq.read_table(f).to_pydict()
        a=np.array(t['action'],dtype=np.float32);ep=np.array(t['episode_index']);ti=np.array(t['task_index'])
        fidx=np.array(t['frame_index'])
        for e in np.unique(ep):
            m=np.where(ep==e)[0]; dy=a[m,1]; L=len(dy)
            cs=np.concatenate([[0.0],np.cumsum(dy)])
            Ps=cs[np.minimum(np.arange(L)+N,L)]-cs[np.arange(L)]
            for k in range(L):
                task=tasks[ti[m[k]]]
                psi[task].append(float(Ps[k]))
                frames[task].append(dict(ds=d,ep=int(e),frame=int(fidx[m[k]]),psi=float(Ps[k])))
# データ由来のクラス: フレームの多数決（θで3値化）＋十分な優勢を要求
out={}
for task,ps in psi.items():
    p=np.array(ps)
    lab=np.where(p>TH,'L',np.where(p<-TH,'R','S'))
    cnt=collections.Counter(lab); tot=len(lab)
    top,ntop=cnt.most_common(1)[0]
    out[task]=dict(n=tot,frac={k:cnt[k]/tot for k in 'LRS'},data_cls=top,purity=ntop/tot,
                   mean=float(p.mean()),median=float(np.median(p)))
json.dump(out,open(SP+'data_cls.json','w'),ensure_ascii=False)
# 文字列推定との比較
LMK=re.compile(r'\b(?:on|to|off to)\s+(?:the\s+|your\s+)?(left|right)\b'); DT=re.compile(r'\b(left|right)\b')
def tcls(s):
    t=s.lower()
    if re.search(r'turn around|u-turn|double back',t): return 'U'
    d=set(DT.findall(LMK.sub(' LM ',t)))
    return 'L' if d=={'left'} else 'R' if d=={'right'} else 'S' if not d else '?'
dis=[(t,tcls(t),v['data_cls'],v['purity'],v['mean'],v['n']) for t,v in out.items() if tcls(t)!=v['data_cls']]
print(f"指示 {len(out)}種 / 文字列推定とデータ由来クラスの不一致: {len(dis)}種 ({len(dis)/len(out)*100:.0f}%)")
print("\n--- 不一致の例（純度の高い＝データが明確なもの上位10）---")
for t,tc,dc,pu,mn,n in sorted(dis,key=lambda x:-x[3])[:10]:
    print(f"  文字列={tc} データ={dc} 純度{pu:.2f} 平均Ψ{mn:+.3f} n={n}  {t[:60]}")
print("\n--- 純度の分布 ---")
pu=np.array([v['purity'] for v in out.values()])
for th_ in [0.5,0.6,0.7,0.8,0.9]:
    ok=[t for t,v in out.items() if v['purity']>=th_ and v['n']>=8]
    cc=collections.Counter(out[t]['data_cls'] for t in ok)
    print(f"  純度>={th_} かつ n>=8: {len(ok):3d}種 (直進 {cc['S']:3d} / 左折 {cc['L']:3d} / 右折 {cc['R']:3d})")
