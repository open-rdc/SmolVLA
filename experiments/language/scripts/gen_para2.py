# -*- coding: utf-8 -*-
"""言い換えの機械生成（拡張版）。
・正解ラベルは実データのΨ由来（data_cls.json）を使う
・方向語の保存は「元の指示に現れる left/right トークンの集合が一致するか」という語彙的検査で行う
  （文字列からクラスを推定する方式は27%誤るため使わない）
"""
import json,re,itertools,random,collections,os
SP=os.environ.get('WORK','.').rstrip('/')+'/'          # 入出力ディレクトリ。既定はカレント
vocab=set(json.load(open(SP+'train_vocab_full.json')))
words=set(json.load(open(SP+'train_words.json')).keys())
DC=json.load(open(SP+'data_cls.json'))
norm=lambda s:re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]','',s.lower())).strip()
VN={norm(s) for s in vocab}
DIR=re.compile(r'\b(left|right)\b',re.I)
def dirs(s): return collections.Counter(m.lower() for m in DIR.findall(s))

MILD={'avenue':['street','road'],'street':['road','avenue'],'road':['street','roadway'],
 'path':['trail','walkway'],'walkway':['path','footway'],'promenade':['walkway'],
 'plaza':['square','open area'],'square':['plaza'],'area':['zone'],'lot':['yard'],
 'hedge':['bushes','hedgerow'],'hedges':['bushes'],'trees':['trees'],'tree':['tree'],
 'tree-lined':['tree-bordered'],'lined':['bordered'],'lawn':['grass'],'grass':['lawn'],
 'grassy':['grass-covered'],'green':['lawn'],'building':['structure'],'buildings':['structures'],
 'glass':['glass-walled'],'white':['pale'],'red':['reddish'],'brick':['brickwork'],
 'bicycle':['bike'],'racks':['stands'],'parking':['parking area'],'corridor':['passage'],
 'gate':['gateway'],'statue':['monument'],'colonnade':['columns'],'park':['park grounds'],
 'pylon':['cone'],'corner':['bend'],'front':['ahead'],'open':['unenclosed'],
 'straight':['straight ahead'],'continue':['keep going'],'head':['go'],'proceed':['continue on'],
 'follow':['trace'],'following':['tracing'],'move':['travel'],'bear':['keep'],
 'both sides':['either side'],'ahead':['in front'],'toward':['towards'],'across':['over'],
 'along':['beside'],'beside':['alongside'],'down':['along'],'between':['amid'],'past':['beyond'],
 'through':['across'],'around':['about'],'entrance':['doorway'],'apartment':['residential'],
 'towers':['blocks'],'tiled':['tile-paved'],'paved':['surfaced'],'trash can':['bin'],
 'covered':['roofed'],'curves':['bends'],'curving':['bending'],'edge':['side'],'far':['distant']}
OOV={'avenue':['boulevard','esplanade'],'street':['boulevard'],'road':['lane','thoroughfare'],
 'path':['lane','aisle'],'walkway':['arcade','aisle'],'promenade':['esplanade'],
 'plaza':['quad','esplanade'],'square':['quad','flagstone court'],'area':['precinct'],'lot':['yard'],
 'hedge':['shrubs','thicket'],'hedges':['shrubs','thickets'],'trees':['foliage','greenery'],
 'tree':['sapling'],'tree-lined':['foliage-lined'],'lined':['rimmed','flanked'],
 'lawn':['turf','meadow'],'grass':['turf','meadow'],'grassy':['turf-covered'],'green':['turf'],
 'building':['block','edifice'],'buildings':['blocks','edifices'],'glass':['glazed'],
 'white':['chalky'],'red':['ochre'],'brick':['cobbled'],'bicycle':['cycle'],'racks':['stalls','rails'],
 'parking':['stalls'],'corridor':['aisle','arcade'],'gate':['doorway','portal'],
 'statue':['monument','effigy'],'colonnade':['portico','arcade'],'park':['commons'],
 'pylon':['bollard'],'corner':['bend'],'front':['fore'],'open':['unwalled'],
 'paved':['flagstone','cobbled'],'tiled':['flagstone'],'entrance':['doorway','portal'],
 'apartment':['tenement'],'towers':['spires'],'trash can':['waste bin'],'covered':['canopied'],
 'both sides':['each flank','either hand'],'ahead':['up ahead','beyond'],
 'continue':['carry on','press on'],'head':['keep on','press'],'proceed':['carry on'],
 'follow':['hug','trace'],'following':['hugging','tracing'],'move':['proceed onward'],
 'edge':['verge','rim'],'curves':['bends'],'curving':['bending'],'across':['over'],
 'along':['down'],'beside':['abutting'],'toward':['for'],'between':['dividing'],'past':['beyond'],
 'through':['athwart'],'around':['round'],'far':['outermost'],'bear':['veer']}
TURNV={'L':['swing left','veer left','hook left','pivot left','cut left'],
       'R':['swing right','veer right','hook right','pivot right','cut right']}
TURNPAT=re.compile(r'\b(turn|bear|curve|veer)\s+(left|right)\b',re.I)
VERBKEYS={'continue','head','proceed','follow','move','following'}

def clauses(s): return [c.strip() for c in s.split(',') if c.strip()]
PREP=r'(?:along|toward|towards|past|across|down|through|beside|between|onto|into|around|over|under|beneath|near|by|at|on|in|for|with)'
def split_pp(s):
    """命令部 + 前置詞句の列 に分解する。語は一切変えない。"""
    t=s.strip().rstrip('.')
    ms=list(re.finditer(r'\b'+PREP+r'\b',t,re.I))
    # 先頭が前置詞の場合は分解しない（既に前置されている）
    ms=[m for m in ms if m.start()>0]
    if not ms: return t,[]
    heads=[t[:ms[0].start()].strip()]
    parts=[]
    for i,m in enumerate(ms):
        end=ms[i+1].start() if i+1<len(ms) else len(t)
        parts.append(t[m.start():end].strip())
    return heads[0],parts
def _case(first_upper,txt):
    if not txt: return txt
    return (txt[0].upper()+txt[1:]) if first_upper else (txt[0].lower()+txt[1:])
def gen_L1(s,k=6):
    """L1 = 語順（および大文字小文字）のみの変更。語の追加・削除・置換はしない。"""
    out=[]
    up=s[:1].isupper()
    # (a) カンマ節の並べ替え
    cs=clauses(s.rstrip('.'))
    if len(cs)>=2:
        for p in itertools.permutations(range(len(cs))):
            if list(p)==list(range(len(cs))): continue
            out.append(_case(up,', '.join(cs[i] for i in p)))
            if len(out)>=k: break
    # (b) 前置詞句の前置
    head,pps=split_pp(s)
    if head and pps:
        for i in range(len(pps)):
            cand=' '.join([pps[i]]+[p for j,p in enumerate(pps) if j!=i]+[head])
            out.append(_case(up,cand))
        if len(pps)>=2:
            cand=' '.join(list(reversed(pps))+[head])
            out.append(_case(up,cand))
    # 元と同一のものを除去し重複を潰す
    seen=set(); res=[]
    for c in out:
        c=re.sub(r'\s+',' ',c).strip()
        kk=norm(c)
        if kk==norm(s) or kk in seen: continue
        seen.add(kk); res.append(c)
    return res[:k]
def sub_map(s,M,rng):
    t=s; used=0; vdone=False
    for k in sorted(M,key=len,reverse=True):
        if k=='straight' and vdone: continue
        pat=re.compile(r'\b'+re.escape(k)+r'\b',re.I)
        if pat.search(t):
            t=pat.sub(rng.choice(M[k]),t,count=1); used+=1
            if k in VERBKEYS: vdone=True
    return t,used
def gen_sub(s,M,rng,k=6,need=2):
    out=set()
    for _ in range(60):
        t,u=sub_map(s,M,rng)
        if u>=need and t.lower()!=s.lower(): out.add(re.sub(r'\s+',' ',t).strip())
        if len(out)>=k: break
    return list(out)
def gen_L3(s,cls,rng,k=6):
    out=set()
    for _ in range(120):
        t,u=sub_map(s,OOV,rng)
        if cls in 'LR' and TURNPAT.search(t):
            t=TURNPAT.sub(rng.choice(TURNV[cls]),t,count=1)
            u+=1                      # 旋回動詞の置換も1置換として数える（語彙外動詞が入る）
        if u>=2: out.add(re.sub(r'\s+',' ',t).strip())
        if len(out)>=k: break
    return list(out)

def build(sel,seed=7):
    rng=random.Random(seed); NEW={}; st=collections.Counter()
    for orig in sel:
        cls=DC[orig]['data_cls']; d={'class':cls,'purity':DC[orig]['purity'],'n_frames':DC[orig]['n']}
        d0=dirs(orig)
        for lv,gen in [('L1',lambda:gen_L1(orig)),
                       ('L2',lambda:gen_sub(orig,MILD,rng)),
                       ('L3',lambda:gen_L3(orig,cls,rng))]:
            keep=[]
            for c in gen():
                if norm(c) in VN: st[lv+'_leak']+=1; continue
                if dirs(c)!=d0: st[lv+'_dir']+=1; continue      # ★方向語トークンの集合が元と一致すること
                if lv=='L3' and not [w for w in re.findall(r'[a-z]+',c.lower()) if w not in words]:
                    st['L3_nooov']+=1; continue
                if norm(c) in {norm(x) for x in keep}: continue
                keep.append(c)
            d[lv]=keep[:5]; st[lv]+=len(d[lv])
        NEW[orig]=d
    return NEW,st

def dedup(NEW):
    """階層内で重複した文字列は1本だけ残す。残す先は変種が少ない指示（均衡のため）。"""
    idx=collections.defaultdict(list)
    for o,v in NEW.items():
        for lv in ['L1','L2','L3']:
            for p in v[lv]: idx[(lv,norm(p))].append((o,p))
    removed=0
    for (lv,_),lst in idx.items():
        if len(lst)<2: continue
        lst.sort(key=lambda x:len(NEW[x[0]][lv]))     # 変種が少ない指示を優先して残す
        for o,p in lst[1:]:
            if p in NEW[o][lv]: NEW[o][lv].remove(p); removed+=1
    return removed

if __name__=='__main__':
    sel=json.load(open(SP+'sel_instructions_v3.json'))
    NEW,st=build(sel)
    rm=dedup(NEW); print(f"階層内重複を1本に統合: {rm}本除去")
    json.dump(NEW,open(SP+'paraphrases_v3.json','w'),ensure_ascii=False,indent=1)
    tot=sum(len(v[lv]) for v in NEW.values() for lv in ['L1','L2','L3'])
    print("生成本数:",{k:v for k,v in st.items() if not k.endswith(('_leak','_dir','_nooov'))})
    print("自動除外:",{k:v for k,v in st.items() if k.endswith(('_leak','_dir','_nooov'))})
    print(f"合計 {tot}本 / 指示 {len(NEW)}種")
