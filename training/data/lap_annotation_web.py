#!/usr/bin/env python3
"""周回エピソード(bag_to_lap_episode.py の出力)の言語指示を確認・修正する Web ツール。

NavVLA/tools/lang_anotation_tool.py(Tk)の置き換え。1 周を 1 本の長いエピソードのまま開き、
指示が同じ連続区間ごとに確認していく。確認が終わったら split_lap_by_prompt.py で区切り直す。

画面:
  - 画像(640x480 魚眼をそのまま大きく)と，周回の軌跡(今の区間を強調)
  - タイムライン: 指示ごとに色分けした区間。赤 = 未割当(No language instruction)，
    下の緑の線 = 確認済み。クリックでそのフレームへ，ドラッグで範囲を選ぶ
  - 今の区間の指示の編集欄(旧トポマップと既存の指示から候補を出す)
  - 区間の実際の旋回(4.8 秒先までの向き変化 Ψ から求めた左折/右折/直進)と，指示の方向語が
    食い違うときは警告を出す

キー: ←/→ 1 フレーム，Ctrl+←/→ 10 フレーム，Shift+←/→ 範囲を伸ばす(押し始めが始点)，↑/↓ 前/次の区間，
      Space 再生/停止，[ / ] 範囲の始点/終点を今のフレームに，Enter 指示を範囲に適用して確認済みにし次の未確認区間へ，
      N / Shift+N 次/前の停止へ，Delete 範囲に DELETE(学習から除く)を付ける，
      H 今のフレームで HOLD_S 秒止まる印(書き出しでそのフレームを複製する。白線の前で待つ場面などを作る)，Ctrl+S 保存
保存: <lap>/episode01/traj_prompt.txt(初回保存時に traj_prompt.auto.txt として元を残す)と
      <lap>/episode01/review_state.json(フレームごとの確認済みフラグ)，holds.json(止まる印のフレーム番号)

    python training/data/lap_annotation_web.py ~/tsudanuma_data/laps \\
        --instructions ~/tsudanuma_data/old_topomap_instructions.txt --port 8010
    → ブラウザで http://localhost:8010
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np

DUMMY = "No language instruction"
H = 24          # Ψ の先読み(4.8 s)。eval_monitor.py と同じ
THETA = 0.090   # 行動クラスのしきい値 [rad]。eval_monitor.py と同じ

ARGS = None


def lap_dirs():
    return sorted(p for p in ARGS.laps_root.iterdir() if (p / "episode01" / "traj_prompt.txt").exists())


def ep_dir(name: str) -> Path:
    p = (ARGS.laps_root / name / "episode01").resolve()
    if ARGS.laps_root.resolve() not in p.parents:
        raise ValueError(name)
    return p


def read_prompts(d: Path) -> list[str]:
    return (d / "traj_prompt.txt").read_text(encoding="utf-8").split("\n")


def read_reviewed(d: Path, n: int) -> list[int]:
    f = d / "review_state.json"
    if f.exists():
        r = json.loads(f.read_text())["reviewed"]
        if len(r) == n:
            return r
    return [0] * n


def read_holds(d: Path, n: int) -> list[int]:
    f = d / "holds.json"
    return sorted(h for h in json.loads(f.read_text())["holds"] if 0 <= h < n) if f.exists() else []


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def translations() -> dict[str, str]:
    """指示の日本語訳 {英語: 日本語}。--translations が無ければ空。"""
    if ARGS.translations and ARGS.translations.exists():
        return json.loads(ARGS.translations.read_text(encoding="utf-8"))
    return {}


def instruction_classes() -> dict[str, str]:
    """指示の動作分類 {英語: L/R/S/U}。--classes が無ければ空(画面側で語から推定する)。"""
    if ARGS.classes and ARGS.classes.exists():
        return json.loads(ARGS.classes.read_text(encoding="utf-8"))
    return {}


def suggestions() -> list[str]:
    """候補 = 旧トポマップの指示 + 各周回で使われている指示(使用回数の多い順)。"""
    cnt = collections.Counter()
    for p in lap_dirs():
        for s in read_prompts(p / "episode01"):
            if s and s != DUMMY:
                cnt[s] += 1
    out = [s for s, _ in cnt.most_common()]
    if ARGS.instructions and ARGS.instructions.exists():
        for line in ARGS.instructions.read_text(encoding="utf-8").splitlines():
            s = line.split("\t", 1)[-1].strip()
            if s and s not in cnt:
                out.append(s)
    return out


def lap_summary(p: Path) -> dict:
    d = p / "episode01"
    pr = read_prompts(d)
    rv = read_reviewed(d, len(pr))
    return {"name": p.name, "n": len(pr), "dummy": sum(s == DUMMY for s in pr), "reviewed": sum(rv)}


def lap_detail(name: str) -> dict:
    d = ep_dir(name)
    pr = read_prompts(d)
    n = len(pr)
    traj = pickle.load(open(d / "traj_data.pkl", "rb"))
    pos = np.asarray(traj["position"], dtype=float)[:n]
    yaw = np.unwrap(np.asarray(traj["yaw"], dtype=float)[:n])
    psi = np.array([yaw[min(t + H, n - 1)] - yaw[t] for t in range(n)])
    cls = np.where(psi > THETA, "L", np.where(psi < -THETA, "R", "S"))
    return {"name": name, "n": n, "prompts": pr, "reviewed": read_reviewed(d, n),
            "pos": np.round(pos, 3).tolist(), "yaw": np.round(yaw, 4).tolist(),
            "cls": "".join(cls.tolist()), "psi": np.round(psi, 3).tolist(),
            # 画像 URL に付ける版。フレームを削って番号を詰め直したら変わり、ブラウザの古いキャッシュを使わない
            "ver": (d / "timestamps.npy").stat().st_mtime_ns, "holds": read_holds(d, n)}


def save(name: str, body: dict) -> dict:
    d = ep_dir(name)
    old = read_prompts(d)
    pr, rv = body["prompts"], body["reviewed"]
    if len(pr) != len(old) or len(rv) != len(old):
        raise ValueError("フレーム数が合わない")
    if any("\n" in s for s in pr):
        raise ValueError("指示に改行がある")
    auto = d / "traj_prompt.auto.txt"
    if not auto.exists():
        auto.write_text("\n".join(old), encoding="utf-8")
    write_atomic(d / "traj_prompt.txt", "\n".join(s.strip() or DUMMY for s in pr))
    write_atomic(d / "review_state.json", json.dumps({"reviewed": [int(bool(x)) for x in rv]}))
    if "holds" in body:
        write_atomic(d / "holds.json", json.dumps({"holds": sorted({int(h) for h in body["holds"] if 0 <= int(h) < len(old)})}))
    return lap_summary(ARGS.laps_root / name)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        try:
            if path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/laps":
                self.send_json({"laps": [lap_summary(p) for p in lap_dirs()], "suggestions": suggestions(),
                                "ja": translations(), "icls": instruction_classes()})
            elif m := re.fullmatch(r"/api/lap/([^/]+)", path):
                self.send_json(lap_detail(m.group(1)))
            elif m := re.fullmatch(r"/img/([^/]+)/(\d+)\.jpg", path):
                f = ep_dir(m.group(1)) / f"{int(m.group(2))}.jpg"
                self.send(200, f.read_bytes(), "image/jpeg", cache=True)
            else:
                self.send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001
            self.send_json({"error": str(e)}, 500)

    def do_POST(self):
        path = unquote(self.path)
        try:
            m = re.fullmatch(r"/api/lap/([^/]+)/save", path)
            if not m:
                return self.send(404, b"not found", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_json(save(m.group(1), body))
        except Exception as e:  # noqa: BLE001
            self.send_json({"error": str(e)}, 500)


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>周回の言語指示</title>
<style>
:root{--bg:#16181d;--panel:#20232a;--line:#343842;--text:#e6e6e6;--muted:#9aa0aa;--accent:#4ea1ff;--ok:#3fb950;--bad:#f85149;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Noto Sans JP",sans-serif}
header{display:flex;gap:12px;align-items:center;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
header select{background:var(--bg);color:var(--text);border:1px solid var(--line);padding:4px 6px;font-size:14px}
#status{margin-left:auto;color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;padding:12px 14px}
#imgwrap{position:relative;background:#000;border-radius:6px;overflow:hidden}
#img{display:block;width:100%;height:calc(100vh - 215px);min-height:360px;object-fit:contain;margin:0 auto}
#frameinfo{position:absolute;left:8px;top:6px;background:#000a;padding:2px 8px;border-radius:4px;font-variant-numeric:tabular-nums}
aside{display:flex;flex-direction:column;gap:10px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}
#map{width:100%;height:200px;display:block}
.k{color:var(--muted);font-size:12px}
.cls{font-weight:700;font-size:18px}
#warn{color:var(--warn);min-height:1.5em}
#timeline{width:100%;height:56px;display:block;cursor:crosshair;margin-top:10px;border-radius:4px}
#edit{display:flex;gap:8px;align-items:center;margin-top:10px}
#prompt{flex:1;font-size:17px;padding:8px 10px;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:4px}
button{background:#2b3038;color:var(--text);border:1px solid var(--line);border-radius:4px;padding:7px 12px;font-size:14px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
#range{font-variant-numeric:tabular-nums}
.help{color:var(--muted);font-size:12px;margin-top:6px}
.ja{color:var(--muted);font-size:14px;margin-top:2px}
#promptja{min-height:1.5em;padding-left:2px}
#tabs{display:flex;gap:4px;margin:6px 0}
#tabs button{flex:1;padding:5px 4px}
#tabs button.on{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
#chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
#chips button{padding:4px 8px}
#chips button.on{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
#sfilter{width:100%;padding:5px 8px;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:4px}
#slist{max-height:calc(100vh - 540px);min-height:200px;overflow-y:auto;margin-top:6px}
.si{padding:5px 6px;border-bottom:1px solid var(--line);cursor:pointer}
.si:hover{background:#2b3038}
.si.cur{outline:1px solid var(--accent)}
.si .ja{font-size:12px;margin:0}
</style></head><body>
<header>
  <strong>周回の言語指示</strong>
  <select id="lapsel"></select>
  <span id="lapprog" class="k"></span>
  <span id="status">読み込み中</span>
</header>
<main>
  <section>
    <div id="imgwrap"><img id="img" alt=""><div id="frameinfo"></div></div>
    <canvas id="timeline"></canvas>
    <div id="edit">
      <span id="range"></span>
      <input id="prompt" list="sugg" placeholder="指示（候補から選ぶか入力）" autocomplete="off">
      <datalist id="sugg"></datalist>
      <button id="apply" class="primary" title="Enter">適用して確認済み → 次へ</button>
      <button id="unrev" title="確認済みを外す">未確認に戻す</button>
    </div>
    <div id="promptja" class="ja"></div>
    <div class="help">←/→ 1フレーム　Ctrl+←/→ 10フレーム　Shift+←/→ 範囲を伸ばす（押し始めが始点，Ctrl+Shift で 10 フレームずつ）　↑/↓ 前/次の区間　Space 再生　[ / ] 範囲の始点/終点　N / Shift+N 次/前の停止　Delete 範囲を学習から除く　H ここで 5 秒止まる（複製）　Enter 適用して次へ　Ctrl+S 保存　タイムライン: クリックで移動、ドラッグで範囲選択</div>
  </section>
  <aside>
    <div class="box"><canvas id="map"></canvas></div>
    <div class="box">
      <div class="k">今のフレームの実際の動き（4.8 秒先までの向き変化）</div>
      <div><span id="cls" class="cls"></span> <span id="psi" class="k"></span></div>
      <div class="k" style="margin-top:6px">今の区間</div>
      <div id="seginfo"></div>
      <div id="warn"></div>
    </div>
    <div class="box"><div class="k">今のフレームの指示</div><div id="cur"></div><div id="curja" class="ja"></div></div>
    <div class="box">
      <div class="k">指示の候補（クリックで入力欄へ。タブは今のフレームの実際の動きに合わせて切り替わる）</div>
      <div id="tabs"></div>
      <div id="chips"></div>
      <input id="sfilter" placeholder="絞り込み（英語・日本語）" autocomplete="off">
      <div id="slist"></div>
    </div>
  </aside>
</main>
<script>
const DUMMY="No language instruction";
// 学習から除くフレームの印(書き出しで除く。センサ確認などで理由なく止まった所)
const DEL="DELETE";
// H で付ける「ここで止まる」印の長さ(書き出しの export_tsukuba_train.py の HOLD_S と同じ)
const HOLD_S=5;
let JA={},ICLS={},SUGG=[],tab="S",tabLocked=false,lastSeg="";const ja=s=>JA[(s||"").trim()]||"";
const TABS=[["L","左折"],["R","右折"],["S","直進"],["U","転回"]];
const icls=s=>ICLS[s]||dirWords(s)||"S";
let laps=[],L=null,f=0,selA=0,selB=0,anchor=null,dirty=false,playing=null,drag=null;
const $=id=>document.getElementById(id);
const CLSNAME={L:"左折",R:"右折",S:"直進"};

function hashColor(s){if(s===DUMMY)return"#f85149";if(s===DEL)return"#555";let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))>>>0;return`hsl(${h%360} 55% 52%)`}
function segAt(i){const p=L.prompts[i];let a=i,b=i;while(a>0&&L.prompts[a-1]===p)a--;while(b<L.n-1&&L.prompts[b+1]===p)b++;return[a,b]}
function setStatus(t){$("status").textContent=t}

async function loadLaps(keep){
  const r=await (await fetch("/api/laps")).json();laps=r.laps;JA=r.ja||{};ICLS=r.icls||{};SUGG=r.suggestions;
  const q=t=>t.replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;");
  $("sugg").innerHTML=r.suggestions.map(s=>`<option value="${q(s)}" label="${q(ja(s))}">`).join("");
  const sel=$("lapsel"),cur=keep||sel.value;
  sel.innerHTML=laps.map(l=>`<option value="${l.name}">${l.name}　確認 ${Math.round(100*l.reviewed/l.n)}%　未割当 ${l.dummy}</option>`).join("");
  if(cur)sel.value=cur;
}
async function openLap(name){
  if(dirty&&!confirm("保存していない変更があります。破棄して開きますか？"))return;
  setStatus("読み込み中");L=await (await fetch("/api/lap/"+encodeURIComponent(name))).json();dirty=false;L.stops=findStops();
  const first=L.reviewed.indexOf(0);go(first<0?0:first,true);setStatus("保存済み");
}
function go(i,resetSel){
  f=Math.max(0,Math.min(L.n-1,i));
  if(resetSel){[selA,selB]=segAt(f);anchor=null}
  $("img").src=`/img/${L.name}/${f}.jpg?v=${L.ver}`;
  for(let k=1;k<=6;k++){if(f+k<L.n)(new Image).src=`/img/${L.name}/${f+k}.jpg?v=${L.ver}`}
  render();
}
// 停止の区間 [[a,b],...]: 前後 1 秒の平均で 速度<0.05 m/s かつ 角速度<0.05 rad/s(finalize_tsukuba.py と同じ)
function findStops(){
  const n=L.n,h=5,st=[];
  for(let i=0;i<n;i++){const lo=Math.max(0,i-h),hi=Math.min(n-1,i+h),dt=Math.max(hi-lo,1)/5;
    const v=Math.hypot(L.pos[hi][0]-L.pos[lo][0],L.pos[hi][1]-L.pos[lo][1])/dt,w=Math.abs(L.yaw[hi]-L.yaw[lo])/dt;st.push(v<0.05&&w<0.05)}
  const out=[];let i=0;while(i<n){if(st[i]){let j=i;while(j<n&&st[j])j++;if(j-i>=3)out.push([i,j-1]);i=j}else i++}
  return out;
}
function goStop(dir){
  const s=dir>0?L.stops.find(([a])=>a>f):[...L.stops].reverse().find(([,b])=>b<f);
  if(!s){setStatus(dir>0?"この後に停止はありません":"この前に停止はありません");return}
  anchor=null;go(s[0],false);[selA,selB]=s;render();
  setStatus(`停止 ${L.stops.indexOf(s)+1}/${L.stops.length}（${((s[1]-s[0]+1)/5).toFixed(1)} 秒）`);
}
function markDelete(){
  const a=Math.min(selA,selB),b=Math.max(selA,selB);
  for(let i=a;i<=b;i++){L.prompts[i]=DEL;L.reviewed[i]=1}
  dirty=true;save();go(b,false);goStop(1);
}
function dirWords(s){
  // 「右手に建物」のような目印の表現を除いてから，旋回の向きを表す語を探す
  const t=s.toLowerCase().replace(/\b(on|to|at|along|of|from|by) (your|the|its) (left|right)\b/g,"").replace(/on both sides/g,"");
  const l=/\bleft\b/.test(t),r=/\bright\b/.test(t);return l&&!r?"L":r&&!l?"R":null;
}
function render(){
  const p=L.prompts[f],[a,b]=segAt(f);
  $("frameinfo").textContent=`${f} / ${L.n-1}`+(L.holds.includes(f)?`　（ここで ${HOLD_S} 秒止まる）`:"");
  $("cls").textContent=CLSNAME[L.cls[f]];$("psi").textContent=`Ψ = ${L.psi[f].toFixed(2)} rad`;
  const cnt={L:0,R:0,S:0};for(let i=a;i<=b;i++)cnt[L.cls[i]]++;
  const dom=Object.entries(cnt).sort((x,y)=>y[1]-x[1])[0][0];
  $("seginfo").textContent=`${a}–${b}（${b-a+1} フレーム, ${((b-a+1)/5).toFixed(1)} 秒）　左折 ${cnt.L} / 右折 ${cnt.R} / 直進 ${cnt.S}`;
  const d=dirWords(p);let w="";
  if(p===DUMMY)w="未割当の区間です";
  else if(p===DEL)w="学習から除く区間です";
  else if(d&&cnt[d]===0)w=`指示は${CLSNAME[d]}ですが、区間内に${CLSNAME[d]}のフレームがありません`;
  else if(!d&&/\bturn\b/i.test(p))w="旋回の指示ですが向き（left/right）が読み取れません";
  $("warn").textContent=w;
  $("cur").textContent=p;$("curja").textContent=ja(p);
  if(document.activeElement!==$("prompt"))$("prompt").value=L.prompts[selA]===DUMMY?"":L.prompts[selA];
  $("promptja").textContent=ja($("prompt").value);
  $("range").textContent=`範囲 ${selA}–${selB}`;
  const rv=L.reviewed.slice(0).reduce((s,x)=>s+x,0);
  $("lapprog").textContent=`確認 ${rv}/${L.n}（${Math.round(100*rv/L.n)}%）　未割当 ${L.prompts.filter(x=>x===DUMMY).length}`;
  const seg=`${L.name}:${a}`;if(seg!==lastSeg){lastSeg=seg;tabLocked=false}
  if(!tabLocked)tab=L.cls[f];
  renderSugg();
  drawTimeline();drawMap(a,b);
}
function renderSugg(){
  const q=$("sfilter").value.trim().toLowerCase(),cur=$("prompt").value.trim();
  const hit=s=>!q||s.toLowerCase().includes(q)||ja(s).includes(q);
  const cnt={L:0,R:0,S:0,U:0};for(const s of SUGG)if(hit(s))cnt[icls(s)]++;
  $("tabs").innerHTML=TABS.map(([c,n])=>`<button data-c="${c}" class="${c===tab?"on":""}">${n} ${cnt[c]}</button>`).join("");
  const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;");
  $("slist").innerHTML=SUGG.filter(s=>icls(s)===tab&&hit(s)).map(s=>
    `<div class="si${s===cur?" cur":""}" data-s="${esc(s).replace(/"/g,"&quot;")}">${esc(s)}<div class="ja">${esc(ja(s))}</div></div>`).join("");
}
function sizeCanvas(c){const r=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=r.width*dpr;c.height=r.height*dpr;const g=c.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);return[g,r.width,r.height]}
function drawTimeline(){
  const[g,W,Hh]=sizeCanvas($("timeline"));g.fillStyle="#0d0f12";g.fillRect(0,0,W,Hh);
  const x=i=>i/L.n*W;let i=0;
  while(i<L.n){const[a,b]=segAt(i);g.fillStyle=hashColor(L.prompts[i]);g.fillRect(x(a),6,Math.max(1,x(b+1)-x(a)-0.5),30);i=b+1}
  for(i=0;i<L.n;i++)if(L.reviewed[i]){let j=i;while(j<L.n&&L.reviewed[j])j++;g.fillStyle="#3fb950";g.fillRect(x(i),40,x(j)-x(i),6);i=j}
  g.strokeStyle="#fff";g.lineWidth=2;g.strokeRect(x(selA),3,x(selB+1)-x(selA),36);
  g.fillStyle="#e3b341";for(const h of L.holds){g.beginPath();g.moveTo(x(h),0);g.lineTo(x(h)-5,-0+8);g.lineTo(x(h)+5,8);g.closePath();g.fill();g.fillRect(x(h)-1,0,2,Hh)}
  g.fillStyle="#fff";g.fillRect(x(f)-1,0,2,Hh);
}
function drawMap(a,b){
  const[g,W,Hh]=sizeCanvas($("map"));g.fillStyle="#0d0f12";g.fillRect(0,0,W,Hh);
  const xs=L.pos.map(p=>p[0]),ys=L.pos.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const s=Math.min((W-20)/(x1-x0||1),(Hh-20)/(y1-y0||1));
  const P=p=>[10+(p[0]-x0)*s+((W-20)-(x1-x0)*s)/2,Hh-10-(p[1]-y0)*s-((Hh-20)-(y1-y0)*s)/2];
  g.lineWidth=1.5;g.strokeStyle="#3a3f48";g.beginPath();L.pos.forEach((p,i)=>{const[u,v]=P(p);i?g.lineTo(u,v):g.moveTo(u,v)});g.stroke();
  g.lineWidth=4;g.strokeStyle=hashColor(L.prompts[f]);g.beginPath();for(let i=a;i<=b;i++){const[u,v]=P(L.pos[i]);i>a?g.lineTo(u,v):g.moveTo(u,v)}g.stroke();
  const[u,v]=P(L.pos[f]),th=L.yaw[f];
  g.fillStyle="#fff";g.beginPath();g.moveTo(u+10*Math.cos(th),v-10*Math.sin(th));g.lineTo(u+5*Math.cos(th+2.5),v-5*Math.sin(th+2.5));g.lineTo(u+5*Math.cos(th-2.5),v-5*Math.sin(th-2.5));g.closePath();g.fill();
}
function applyAndNext(){
  const t=$("prompt").value.trim();if(!t){setStatus("指示が空です");return}
  if(JP.test(t)){setStatus("指示に日本語が混ざっています");return}
  const a=Math.min(selA,selB),b=Math.max(selA,selB);
  for(let i=a;i<=b;i++){L.prompts[i]=t;L.reviewed[i]=1}
  dirty=true;save();
  let n=L.reviewed.indexOf(0,b+1);if(n<0)n=L.reviewed.indexOf(0);
  $("prompt").blur();if(n<0){go(b,true);setStatus("この周回はすべて確認済み");return}go(n,true);
}
async function save(){
  if(!L)return;setStatus("保存中");
  const r=await (await fetch(`/api/lap/${encodeURIComponent(L.name)}/save`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompts:L.prompts,reviewed:L.reviewed,holds:L.holds})})).json();
  if(r.error){setStatus("保存に失敗: "+r.error);return}
  dirty=false;setStatus("保存済み");loadLaps(L.name);
}
function togglePlay(){
  if(playing){clearInterval(playing);playing=null;return}
  playing=setInterval(()=>{if(f>=L.n-1){togglePlay();return}go(f+1,false)},200);
}
document.addEventListener("keydown",e=>{
  if(!L)return;
  if((e.ctrlKey||e.metaKey)&&e.key==="s"){e.preventDefault();save();return}
  if(e.key==="Enter"){e.preventDefault();applyAndNext();return}
  if(document.activeElement===$("prompt")||document.activeElement===$("sfilter")){if(e.key==="Escape")document.activeElement.blur();return}
  const k=e.key;
  if(k==="ArrowRight"||k==="ArrowLeft"){
    // Shift で範囲を伸ばす(押し始めのフレームが始点)。Ctrl は 10 フレームずつ
    const step=(e.ctrlKey||e.metaKey?10:1)*(k==="ArrowRight"?1:-1);
    if(e.shiftKey){if(anchor===null)anchor=f;go(f+step,false);selA=Math.min(anchor,f);selB=Math.max(anchor,f);render()}
    else{anchor=null;go(f+step,false)}
    e.preventDefault()}
  else if(k==="ArrowDown"){const[,b]=segAt(f);go(b+1,true);e.preventDefault()}
  else if(k==="ArrowUp"){const[a]=segAt(f);go(a>0?segAt(a-1)[0]:0,true);e.preventDefault()}
  else if(k===" "){togglePlay();e.preventDefault()}
  else if(k==="n"||k==="N"){goStop(e.shiftKey?-1:1);e.preventDefault()}
  else if(k==="Delete"){markDelete();e.preventDefault()}
  else if(k==="h"||k==="H"){
    const i=L.holds.indexOf(f);if(i<0)L.holds.push(f);else L.holds.splice(i,1);L.holds.sort((a,b)=>a-b);
    dirty=true;save();render();setStatus(i<0?`${f} で ${HOLD_S} 秒止まる印を付けた（書き出しで複製）`:`${f} の止まる印を外した`);e.preventDefault()}
  else if(k==="["){selA=f;if(selB<selA)selB=selA;anchor=null;render()}
  else if(k==="]"){selB=f;if(selA>selB)selA=selB;anchor=null;render()}
});
const tl=$("timeline");
const tlIdx=e=>{const r=tl.getBoundingClientRect();return Math.max(0,Math.min(L.n-1,Math.floor((e.clientX-r.left)/r.width*L.n)))};
tl.addEventListener("mousedown",e=>{if(!L)return;drag=tlIdx(e);go(drag,true)});
tl.addEventListener("mousemove",e=>{if(drag===null)return;const i=tlIdx(e);if(Math.abs(i-drag)>1){selA=Math.min(drag,i);selB=Math.max(drag,i);go(i,false)}});
addEventListener("mouseup",()=>{drag=null});
$("apply").onclick=applyAndNext;
// 日本語入力の変換中に datalist の候補を選ぶと，確定した日本語が候補の後ろに付くので候補だけに戻す
const JP=/[　-ヿ一-鿿＀-￯]/;
function stripIme(e){if(e.isComposing)return;const v=$("prompt").value;if(JP.test(v)){const s=SUGG.find(s=>v.startsWith(s));if(s)$("prompt").value=s}$("promptja").textContent=ja($("prompt").value)}
$("prompt").oninput=stripIme;$("prompt").addEventListener("compositionend",stripIme);
$("tabs").onclick=e=>{const c=e.target.closest("button")?.dataset.c;if(c){tab=c;tabLocked=true;renderSugg()}};
$("slist").onclick=e=>{const el=e.target.closest(".si");if(!el)return;$("prompt").value=el.dataset.s;$("promptja").textContent=ja(el.dataset.s);renderSugg()};
// よく使う検索語。クリックで絞り込み欄に入れる(もう一度押すと消す)
const CHIPS=["並木","生け垣","建物","駐輪","ラック","芝","公園","白","赤"];
function renderChips(){const q=$("sfilter").value.trim();$("chips").innerHTML=CHIPS.map(w=>`<button data-w="${w}" class="${w===q?"on":""}">${w}</button>`).join("")}
$("chips").onclick=e=>{const b=e.target.closest("button");if(!b)return;$("sfilter").value=$("sfilter").value.trim()===b.dataset.w?"":b.dataset.w;renderChips();renderSugg()};
renderChips();
$("sfilter").oninput=()=>{renderChips();renderSugg()};
$("unrev").onclick=()=>{const a=Math.min(selA,selB),b=Math.max(selA,selB);for(let i=a;i<=b;i++)L.reviewed[i]=0;dirty=true;save()};
$("lapsel").onchange=e=>openLap(e.target.value);
addEventListener("resize",()=>L&&render());
addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue=""}});
loadLaps().then(()=>laps.length&&openLap($("lapsel").value));
</script></body></html>
"""


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("laps_root", type=Path, help="bag_to_lap_episode.py の --out-root(<bag名>/episode01 が並ぶ)")
    ap.add_argument("--instructions", type=Path, default=None, help="候補に出す指示の一覧(1 行 1 指示，タブ区切りなら最後の列)")
    ap.add_argument("--translations", type=Path, default=None, help="指示の日本語訳 JSON {英語: 日本語}(画面に併記する)")
    ap.add_argument("--classes", type=Path, default=None, help="指示の動作分類 JSON {英語: L/R/S/U}(候補をタブで分ける)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ARGS = ap.parse_args()
    print(f"http://localhost:{ARGS.port}  ({len(lap_dirs())} 周回)", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
