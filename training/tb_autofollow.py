#!/usr/bin/env python
"""クラスタの学習ログ(.out)を取り込み、resume で分割されたログを自動で --segment に
繋ぎ直して TensorBoard に常時追従させる常駐スクリプト。

背景: lerobot-train は 24h の壁で TIMEOUT するたび resume ジョブが新しい
smolvla_<name>_<jobid>.out に書き始めるので、loss_to_tb.py --follow を手で起動して
おくと古いファイルを見張ったまま配信が止まる(実際に 2026-08-02 に 12h 分の欠損)。
このスクリプトは新しい .out の出現を検知して --segment を組み直し、
loss_to_tb.py と TensorBoard を自動で再起動する。

resume 時の offset は「直前ファイルの最終 step を save_freq で切り捨てた値」で求まる
(resume は checkpoints/last = save_freq の倍数から再開するため)。実績3件
(13339->13348, 13348->13412, 13380->13413) で一致を確認済み。

注意: loss_to_tb.py --follow はイベントファイルを作り直すので、TensorBoard は
必ずその「後に」起動する(逆にすると配信データが古いまま固まる)。

使い方:
  ~/.venvs/smolvla/bin/python training/tb_autofollow.py
  (Ctrl-C で follow / TensorBoard までまとめて停止)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_to_tb import extract_train  # noqa: E402  行数カウントを本体と揃えるため再利用

REPO = Path(__file__).resolve().parent.parent
REMOTE = "V23C1036@10.246.10.30"
REMOTE_GLOB = "/home/share/V23C1036/SmolVLA/smolvla_orne_tc*.out"

# 追従する run: (ログの glob, loss_to_tb の出力先)
# glob は jobid 部分だけが数字になるよう書く。"smolvla_orne_tc_[0-9]*" は
# "smolvla_orne_tc_ms_13380.out" にマッチしない(直後が 'm' なので)。
RUNS = [
    ("smolvla_orne_tc_[0-9]*.out", "tb_orne_tc"),
    ("smolvla_orne_tc_ms_[0-9]*.out", "tb_orne_tc_ms"),
]
# TensorBoard に渡す親ディレクトリ(配下の各シンボリックリンクが別 run として認識される)
TB_LOGDIR = "tb_compare_speed"

JOBID = re.compile(r"_(\d+)\.out$")
RESUME = re.compile(r"Resuming data order at epoch (\d+), sample (\d+)")
FRAMES = re.compile(r"dataset\.num_frames=(\d+)")
BATCH = re.compile(r"Effective batch size:.*=\s*(\d+)")


def resume_step(path: Path, save_freq: int) -> int | None:
    """ログ自身の "Resuming data order at epoch E, sample S" から再開 step を復元する。
    step = (E*num_frames + S)/batch。resume は checkpoints/last = save_freq の倍数から
    始まるので丸めて返す。resume でない(初回)ログは None。"""
    epoch = sample = frames = batch = None
    with path.open(errors="ignore") as fh:
        for line in fh:
            if "ot_train.py:596" in line:
                break  # 学習ログに入った = 以降に必要な行は無い
            if m := RESUME.search(line):
                epoch, sample = int(m[1]), int(m[2])
            elif m := FRAMES.search(line):
                frames = int(m[1])
            elif m := BATCH.search(line):
                batch = int(m[1])
    if epoch is None or frames is None or batch is None:
        return None
    return round((epoch * frames + sample) / batch / save_freq) * save_freq


def segments(pattern: str, log_freq: int, save_freq: int) -> list[tuple[Path, int]]:
    """jobid 昇順の (ログ, offset) 列。offset は各ログ自身の resume 位置から復元し、
    取れないログだけ「直前ファイルの最終 step の save_freq 切り捨て」で補う。
    (TIMEOUT 時は末尾のログ行が欠けることがあり、行数からの推定だけだと後続 segment が
     手前にずれて前 segment を切り落としてしまうため、自己申告の方を優先する。)"""
    files = sorted(REPO.glob(pattern), key=lambda p: int(JOBID.search(p.name)[1]))
    segs, offset = [], 0
    for p in files:
        rows = extract_train(p)
        if not rows:
            continue  # loss 行ゼロのログを segment にすると前 segment をそこで切ってしまう
        own = resume_step(p, save_freq)
        if own is not None:
            offset = own
        segs.append((p, offset))
        offset = (offset + log_freq * len(rows)) // save_freq * save_freq
    return segs


def wait_for_events(logdirs: list[str], timeout: float = 90.0) -> None:
    """各 logdir にイベントファイルが書かれるまで待つ(TB を先に上げないため)。"""
    def ready(d: str) -> bool:
        p = REPO / d
        return p.is_dir() and any(f.stat().st_size > 0 for f in p.glob("events.out.tfevents.*"))

    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(ready(d) for d in logdirs):
            return
        time.sleep(1)
    print("警告: イベントファイルの生成待ちがタイムアウトしました", flush=True)


def spawn(cmd: list[str], log_name: str) -> subprocess.Popen:
    out = open(REPO / log_name, "a", buffering=1)
    return subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
    procs.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=6008)
    ap.add_argument("--interval", type=int, default=25, help="rsync とログ確認の間隔(秒)")
    ap.add_argument("--log-freq", type=int, default=100, help="学習時の log_freq")
    ap.add_argument("--save-freq", type=int, default=10000, help="学習時の save_freq(offset 導出用)")
    a = ap.parse_args()

    py = sys.executable
    tb_bin = str(Path(py).with_name("tensorboard"))
    follows: list[subprocess.Popen] = []
    tb: list[subprocess.Popen] = []
    sig = None

    try:
        while True:
            subprocess.run(["rsync", "-az", f"{REMOTE}:{REMOTE_GLOB}", str(REPO)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=180, check=False)

            segs = {pat: segments(pat, a.log_freq, a.save_freq) for pat, _ in RUNS}
            new_sig = {pat: [(p.name, off) for p, off in s] for pat, s in segs.items()}
            died = [p for p in follows if p.poll() is not None]

            if new_sig != sig or died:
                why = "follow が落ちた" if died and new_sig == sig else "segment 構成が変化"
                print(f"[{time.strftime('%F %T')}] {why} -> 再構築", flush=True)
                stop(follows)
                stop(tb)
                for pat, logdir in RUNS:
                    cmd = [py, "training/loss_to_tb.py"]
                    for p, off in segs[pat]:
                        cmd += ["--segment", f"{p.name}:{off}"]
                    cmd += ["--logdir", logdir, "--follow"]
                    chain = " ".join(f"{p.name}:{off}" for p, off in segs[pat])
                    print(f"    {logdir} <- {chain}", flush=True)
                    follows.append(spawn(cmd, f"{logdir}.follow.log"))
                wait_for_events([d for _, d in RUNS])
                tb.append(spawn([tb_bin, "--logdir", TB_LOGDIR, "--host", "0.0.0.0",
                                 "--port", str(a.port), "--load_fast=false"],
                                "tb_autofollow.tensorboard.log"))
                print(f"    TensorBoard 再起動 (port {a.port}, 起動に数分かかることあり)", flush=True)
                sig = new_sig
            elif tb and tb[0].poll() is not None:
                # ポート衝突やクラッシュで TB だけ死んだ場合は TB のみ上げ直す
                print(f"[{time.strftime('%F %T')}] TensorBoard が落ちていたので再起動", flush=True)
                stop(tb)
                tb.append(spawn([tb_bin, "--logdir", TB_LOGDIR, "--host", "0.0.0.0",
                                 "--port", str(a.port), "--load_fast=false"],
                                "tb_autofollow.tensorboard.log"))

            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    finally:
        stop(follows)
        stop(tb)


if __name__ == "__main__":
    main()
