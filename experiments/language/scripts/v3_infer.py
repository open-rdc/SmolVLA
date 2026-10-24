#!/usr/bin/env python3
"""GPU機で実行する推論。データセット不要（v3_obs.npz だけ読む）。

使い方:
    python v3_infer.py --bench 5          # まず5フレームで速度実測（総時間の見積りが出る）
    python v3_infer.py                    # 本番。途中保存するので中断しても再開できる
    python v3_infer.py --resume           # 中断した続きから

必要ファイル（同じディレクトリに置く）:
    v3_obs.npz  paraphrases_v3.json  train_vocab_full.json  theta.json
重みは --ckpt で指定（既定は候補パスを自動探索）
"""
import json,re,time,os,glob,argparse,collections,numpy as np,torch,warnings
warnings.filterwarnings('ignore')

CKPT_CANDIDATES=[
 os.path.expanduser('~/ros2_ws/src/SmolVLA/outputs/train/smolvla_rec5_v2/checkpoints/last/pretrained_model'),
 os.path.expanduser('~/ros2_ws/src/SmolVLA/deployment/weights/smolvla_rec5_v2'),
 os.path.expanduser('~/smolvla_rec5_v2/pretrained_model'),
 './pretrained_model',
]
ap=argparse.ArgumentParser()
ap.add_argument('--work',default='.')
ap.add_argument('--ckpt',default=None)
ap.add_argument('--bench',type=int,default=0,help='この数のフレームだけ回して速度を測る')
ap.add_argument('--resume',action='store_true')
ap.add_argument('--seeds',type=int,default=3)
ap.add_argument('--bs',type=int,default=8)
ap.add_argument('--num-steps',type=int,default=4,
    help='ODEソルバのステップ数。既定4=実機運用と同じHeun N=4(8評価)。'
         '記憶の実測でEuler N=10より時間20%減かつ精度1.6〜1.8倍と確定している')
a=ap.parse_args(); W=a.work.rstrip('/')+'/'

ck=a.ckpt
if ck is None:
    for c in CKPT_CANDIDATES:
        if os.path.exists(os.path.join(c,'config.json')): ck=c; break
if ck is None or not os.path.exists(os.path.join(ck,'config.json')):
    raise SystemExit(f"重みが見つからない。--ckpt で指定してください。探した場所:\n  "+"\n  ".join(CKPT_CANDIDATES))
print("checkpoint:",ck,flush=True)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from transformers import AutoTokenizer
dev='cuda' if torch.cuda.is_available() else 'cpu'
if dev=='cuda':
    print("GPU:",torch.cuda.get_device_name(0),f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB",flush=True)
else:
    print("★警告: CUDAが使えない。CPUだと現実的な時間で終わらない",flush=True)

Z=np.load(W+'v3_obs.npz',allow_pickle=False)
imgs=Z['images']; states=Z['states']; tasks=[str(x) for x in Z['tasks']]
cls=[str(x) for x in Z['cls']]; true_psi=Z['true_psi']; gidx=Z['gidx']
V=json.load(open(W+'paraphrases_v3.json')); vocab=json.load(open(W+'train_vocab_full.json'))
th=json.load(open(W+'theta.json')); N=th['N']; TH=th['theta']
pc=lambda p:'L' if p>TH else 'R' if p<-TH else 'S'
NF=len(tasks); print(f"フレーム {NF} / 指示 {len(set(tasks))} / 言い換え {sum(len(V[o][l]) for o in V for l in ['L1','L2','L3'])}本",flush=True)

pol=SmolVLAPolicy.from_pretrained(ck).to(dev).eval()
_orig_ns=pol.config.num_steps
if a.num_steps and a.num_steps!=_orig_ns:
    pol.config.num_steps=a.num_steps
    print(f"num_steps を {_orig_ns} → {a.num_steps} に変更（実機運用と同じ設定）",flush=True)
pre,post=make_pre_post_processors(pol.config,pretrained_path=ck,
        preprocessor_overrides={"device_processor":{"device":dev}})
CH=pol.config.chunk_size; AD=pol.config.max_action_dim
CAMS=['observation.images.camera1','observation.images.camera2','observation.images.camera3']

# ---- retrieval 写像（テキストエンコーダ + TF-IDF） ----
MP=W+'meta_v3.json'
if os.path.exists(MP):
    META=json.load(open(MP)); print("retrieval写像: 既存を再利用",flush=True)
else:
    tm=pol.model.vlm_with_expert.vlm.model.text_model
    tok=AutoTokenizer.from_pretrained('HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    LMK=re.compile(r'\b(?:on|to|off to)\s+(?:the\s+|your\s+)?(left|right)\b'); DT=re.compile(r'\b(left|right)\b')
    def tcls(s):
        t=s.lower()
        if re.search(r'turn around|u-turn|double back',t): return 'U'
        d=set(DT.findall(LMK.sub(' LM ',t)))
        return 'L' if d=={'left'} else 'R' if d=={'right'} else 'S' if not d else '?'
    vcls=[tcls(s) for s in vocab]
    Q=[(o,lv,i,p) for o in sorted(V) for lv in ['L1','L2','L3'] for i,p in enumerate(V[o][lv])]
    @torch.no_grad()
    def emb(ts,bs=32):
        out=[]
        for i in range(0,len(ts),bs):
            b=tok(ts[i:i+bs],return_tensors='pt',padding=True,truncation=True,max_length=64).to(dev)
            h=tm(**b).last_hidden_state; m=b['attention_mask'].unsqueeze(-1).float()
            out.append(((h*m).sum(1)/m.sum(1)).float().cpu())
        return torch.cat(out).numpy()
    E=emb(list(vocab)+[q[3] for q in Q]); E/=np.linalg.norm(E,axis=1,keepdims=True)
    S=E[len(vocab):]@E[:len(vocab)].T; nn_e=S.argmax(1); sim_e=S.max(1)
    def tfidf(docs,an):
        fs=[collections.Counter(an(d)) for d in docs]; df=collections.Counter(k for f in fs for k in f)
        idf={k:np.log(len(docs)/(1+v))+1 for k,v in df.items()}; ks={k:i for i,k in enumerate(sorted(df))}
        M=np.zeros((len(docs),len(ks)),dtype=np.float32)
        for i,f in enumerate(fs):
            for k,c in f.items(): M[i,ks[k]]=(1+np.log(c))*idf[k]
        n=np.linalg.norm(M,axis=1,keepdims=True); n[n==0]=1; return M/n
    Mt=tfidf(list(vocab)+[q[3] for q in Q],lambda s:re.findall(r'[a-z]+',s.lower()))
    St=Mt[len(vocab):]@Mt[:len(vocab)].T; nn_t=St.argmax(1); sim_t=St.max(1)
    META={}
    for i,(o,lv,k,p) in enumerate(Q):
        META[f"{o}||{lv}||{k}"]=dict(text=p,lv=lv,intended=V[o]['class'],
            ret_emb=vocab[nn_e[i]],ret_emb_cls=vcls[nn_e[i]],d_emb=float(1-sim_e[i]),
            ret_tfidf=vocab[nn_t[i]],ret_tfidf_cls=vcls[nn_t[i]],d_tfidf=float(1-sim_t[i]))
    json.dump(META,open(MP,'w'),ensure_ascii=False,indent=1)
    print("retrieval写像 作成完了",flush=True)

mirror=lambda s:re.sub(r'\b(left|right)\b',lambda m:'right' if m.group(1)=='left' else 'left',s,flags=re.I)
def conds_for(i):
    t=tasks[i]; c=[('orig','orig',-1,t),('empty','empty',-1,'')]
    if cls[i] in 'LR': c.append(('mirror','mirror',-1,mirror(t)))
    for lv in ['L1','L2','L3']:
        for k,p in enumerate(V[t][lv]):
            m=META[f"{t}||{lv}||{k}"]
            c += [(f'U_{lv}',lv,k,p),(f'Remb_{lv}',lv,k,m['ret_emb']),(f'Rtfidf_{lv}',lv,k,m['ret_tfidf'])]
    return c

RES=W+'v3_results.json'; PROG=W+'v3_progress.json'
res=[]; done=set()
if a.resume and os.path.exists(RES):
    res=json.load(open(RES)); done={r['gidx'] for r in res}
    print(f"再開: 既存 {len(res)}レコード / 済みフレーム {len(done)}",flush=True)

if a.bench:
    step=max(1,NF//a.bench); order=list(range(0,NF,step))[:a.bench]   # 等間隔サンプル
    print(f"ベンチ: 全体から等間隔に{len(order)}フレーム抽出（条件数の偏りを避ける）",flush=True)
else:
    order=list(range(NF))
todo=[i for i in order if gidx[i] not in done]
print(f"処理対象 {len(todo)}フレーム / ノイズ{a.seeds}種 / batch={a.bs}",flush=True)
t0=time.time(); ninf=0
for n,i in enumerate(todo):
    cc_all=conds_for(i)
    im=torch.from_numpy(imgs[i]).permute(0,3,1,2).float()/255.0   # (3,3,224,224)
    st=torch.from_numpy(states[i])
    # 画像埋め込みキャッシュ: 同一フレームの全条件・全ノイズで画像は同じなので
    # SigLIP を1回だけ走らせて再利用する。キャッシュはバッチ形状に依存するので
    # バッチサイズを固定し、端数は複製で埋めて結果を捨てる。
    fid=int(gidx[i])*3
    cache={}; ids=[fid,fid+1,fid+2]
    for sd in range(a.seeds):
        g=torch.Generator(device='cpu').manual_seed(sd*100003+int(gidx[i]))
        base=torch.normal(0.,1.,size=(1,CH,AD),generator=g)
        for s0 in range(0,len(cc_all),a.bs):
            cc=cc_all[s0:s0+a.bs]; real=len(cc)
            pad=cc+[cc[-1]]*(a.bs-real)          # 端数をパディングして形状を一定に保つ
            B=a.bs
            batch={CAMS[c]:im[c].unsqueeze(0).repeat(B,1,1,1) for c in range(3)}
            batch['observation.state']=st.unsqueeze(0).repeat(B,1)
            batch['task']=[x[3] for x in pad]
            with torch.no_grad():
                ch=post(pol.predict_action_chunk(pre(batch),noise=base.repeat(B,1,1).to(dev),
                                                 image_ids=ids,image_embed_cache=cache))
            arr=ch.detach().float().cpu().numpy(); ninf+=real
            for j,(cond,lv,k,text) in enumerate(cc):     # パディング分は捨てる
                p_=float(arr[j,:N,1].sum())
                res.append(dict(gidx=int(gidx[i]),task=tasks[i],true_cls=cls[i],true_psi=float(true_psi[i]),
                    cond=cond,level=lv,vid=k,seed=sd,pred_psi=p_,pred_dx=float(arr[j,:N,0].sum()),pred_cls=pc(p_)))
    cache.clear()
    if (n+1)%10==0 or n+1==len(todo):
        el=time.time()-t0; per=el/(n+1)
        print(f"{n+1}/{len(todo)} {el:.0f}s ({per:.2f}s/f, {ninf/el:.1f}推論/s) "
              f"残り約{per*(len(todo)-n-1)/60:.0f}分 rec={len(res)}",flush=True)
        json.dump(res,open(RES,'w'),ensure_ascii=False)
        json.dump({'done':sorted({int(r['gidx']) for r in res})},open(PROG,'w'))
json.dump(res,open(RES,'w'),ensure_ascii=False)
el=time.time()-t0
if a.bench:
    print(f"\n=== 速度実測 ===")
    print(f"  {a.bench}フレームで {el:.0f}秒 → {el/a.bench:.2f} s/フレーム, {ninf/el:.1f} 推論/秒")
    print(f"  ★全{NF}フレームの見込み: {el/a.bench*NF/3600:.1f} 時間")
else:
    open(W+'DONE_v3','w').write(f"{len(res)} records / {el:.0f}s\n")
    print(f"\nDONE {len(res)}レコード / {el/3600:.2f}時間",flush=True)
