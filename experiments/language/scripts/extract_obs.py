#!/usr/bin/env python3
"""GPU無しのPCで実行する前処理。
ローカルのtcデータセットからフレームを選抜し、観測（画像3枚＋状態）だけを1ファイルに固める。
出力を持っていけば、GPU機はデータセット無しで推論できる。
"""
import json,glob,re,collections,numpy as np,warnings,time,sys
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq
D=os.environ.get('SMOLVLA_DATA','../../training/data').rstrip('/')+'/'   # LeRobotデータセット置き場
SP=os.environ.get('WORK','.').rstrip('/')+'/'          # 入出力ディレクトリ。既定はカレント
PURITY=0.80; PER_INST=int(sys.argv[1]) if len(sys.argv)>1 else 4
V=json.load(open(SP+'paraphrases_v3.json'))
th=json.load(open(SP+'theta.json')); N=th['N']; TH=th['theta']
pc=lambda p:'L' if p>TH else 'R' if p<-TH else 'S'
tcs=['orne_nav_tc','orne_random_tc','orne_junction_tc','orne_linestop_tc','orne_recovery_tc','orne_recovery2_tc']
# --- フレーム選抜（指示のデータ由来クラスと一致するフレームのみ） ---
psi=collections.defaultdict(list); rows=collections.defaultdict(list)
for d in tcs:
    tasks=[str(x) for x in pq.read_table(f'{D}{d}/meta/tasks.parquet').to_pydict()['task']]
    for f in sorted(glob.glob(f'{D}{d}/data/**/*.parquet',recursive=True)):
        t=pq.read_table(f).to_pydict()
        a=np.array(t['action'],dtype=np.float32);ep=np.array(t['episode_index']);ti=np.array(t['task_index'])
        gi=np.array(t['index']); fi=np.array(t['frame_index'])
        for e in np.unique(ep):
            m=np.where(ep==e)[0]; dy=a[m,1]; L=len(dy)
            cs=np.concatenate([[0.0],np.cumsum(dy)])
            Ps=cs[np.minimum(np.arange(L)+N,L)]-cs[np.arange(L)]
            for k in range(L):
                task=tasks[ti[m[k]]]
                if task not in V: continue
                psi[task].append(float(Ps[k]))
                rows[task].append(dict(ds=d,gidx=int(gi[m[k]]),ep=int(e),frame=int(fi[m[k]]),psi=float(Ps[k])))
rng=np.random.default_rng(0); sel=[]; drop=[]
for task in sorted(V):
    ps=psi.get(task,[])
    if not ps: drop.append([task,'フレーム無し']); continue
    cnt=collections.Counter(pc(p) for p in ps); top,nt=cnt.most_common(1)[0]; pur=nt/len(ps)
    if pur<PURITY: drop.append([task,f'純度{pur:.2f}']); continue
    if top!=V[task]['class']: drop.append([task,f'クラス不一致 {V[task]["class"]}→{top}']); continue
    cand=[r for r in rows[task] if pc(r['psi'])==top]
    for i in rng.permutation(len(cand))[:PER_INST]:
        sel.append(dict(task=task,cls=top,purity=float(pur),**cand[i]))
print(f"選抜 {len(sel)}フレーム / 指示 {len(set(s['task'] for s in sel))}種",flush=True)
print("クラス別:",dict(collections.Counter(s['cls'] for s in sel)),flush=True)
print(f"除外 {len(drop)}種:",dict(collections.Counter(d[1].split()[0] for d in drop)),flush=True)
json.dump(sel,open(SP+'v3_frames_local.json','w'),ensure_ascii=False)
json.dump(drop,open(SP+'v3_dropped_local.json','w'),ensure_ascii=False)
# --- 観測の切り出し（動画デコード） ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
DS={}
CAMS=['observation.images.camera1','observation.images.camera2','observation.images.camera3']
imgs=np.zeros((len(sel),3,224,224,3),dtype=np.uint8); states=np.zeros((len(sel),2),dtype=np.float32)
t0=time.time()
for i,s in enumerate(sel):
    if s['ds'] not in DS:
        DS[s['ds']]=LeRobotDataset(repo_id=f"local/{s['ds']}",root=D+s['ds'],video_backend='pyav')
    it=DS[s['ds']][s['gidx']]
    for c,k in enumerate(CAMS):
        x=it[k].numpy()                      # (3,224,224) float32 [0,1]
        imgs[i,c]=np.clip(np.transpose(x,(1,2,0))*255.0+0.5,0,255).astype(np.uint8)
    states[i]=it['observation.state'].numpy()
    if i%100==0:
        el=time.time()-t0
        print(f"  {i+1}/{len(sel)} {el:.0f}s ({el/(i+1):.2f}s/f) 残り約{el/(i+1)*(len(sel)-i-1)/60:.0f}分",flush=True)
np.savez_compressed(SP+'v3_obs.npz',images=imgs,states=states,
                    tasks=np.array([s['task'] for s in sel]),
                    cls=np.array([s['cls'] for s in sel]),
                    true_psi=np.array([s['psi'] for s in sel],dtype=np.float32),
                    gidx=np.array([s['gidx'] for s in sel]),
                    ds=np.array([s['ds'] for s in sel]))
import os
print(f"\n保存: v3_obs.npz  {os.path.getsize(SP+'v3_obs.npz')/1e6:.0f} MB",flush=True)
