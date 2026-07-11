#!/usr/bin/env python3
"""
lerobot-train の .out ログ（train の loss:/grdn:/lr:/epch: と、有効時の eval_loss=）を
TensorBoard event に変換する。

lerobot は TensorBoard を持たない（wandb か .out のみ）ので、.out を後処理して
TB で loss 曲線を見られるようにする。lerobot 本体は非改造。

出力する scalar:
  train/loss, train/grad_norm, train/lr, train/epoch   … 毎 log_freq step
  val/loss                                              … eval_steps>0 のとき eval_steps ごと

使い方（1本の .out、ローカル WSL で表示する想定）:
  rsync -az <user>@<host>:'~/SmolVLA/smolvla_<jobid>.out' .
  ~/.venvs/smolvla/bin/python training/loss_to_tb.py --segment smolvla_<jobid>.out:0 --logdir tb_nav201
  ~/.venvs/smolvla/bin/tensorboard --logdir tb_nav201 --host 0.0.0.0 --port 6007

--resume で学習が複数の .out に分かれた場合（例: 24h打ち切り→resume再開）:
  各 .out の先頭行は「実際は何 step 目からか」が異なる（新規なら0、resumeならcheckpoint step）。
  時系列順に `--segment FILE:OFFSET` を複数指定すると、前の segment は次の segment の
  開始 step より前の区間だけを使い（resumeで re-compute された重複区間は新しい方を優先）、
  重複や逆戻りの無い1本の連続した曲線にする。最後の segment だけ --follow で追記監視できる。

  例: 1本目 smolvla_12768.out(offset=0, step0で開始) が11:00に中断、
      2本目 smolvla_12775.out(offset=200000, checkpoint step200000から再開)で継続:
  python training/loss_to_tb.py \\
      --segment smolvla_12768.out:0 --segment smolvla_12775.out:200000 \\
      --logdir tb_all --follow

注意: train の step 表示は後半 "20K" 等に丸められるため step 値は当てにせず、
      loss 行を log_freq 間隔で数えて正確な step を復元する（--log-freq, 既定100）。
      eval_loss 行は生 step（"step 1000:"）なのでそのまま使う。
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# train の metrics 行（checkpoint 等の INFO 行は loss:/grdn:/lr: を持たないので除外）
TRAIN = re.compile(r"loss:(?P<loss>[-\d.eE+]+).*?grdn:(?P<grdn>[-\d.eE+]+).*?lr:(?P<lr>[-\d.eE+]+)")
EPCH = re.compile(r"epch:(?P<epch>[-\d.eE+]+)")
# 検証 loss 行:  "step 1000: eval_loss=0.4123"（生 step）
EVAL = re.compile(r"step (?P<step>\d+): eval_loss=(?P<v>[-\d.eE+]+)")


def extract_train(path: Path):
    """(loss, grad_norm, lr, epoch) を出現順に。"""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if "eval_loss=" in line:
            continue
        m = TRAIN.search(line)
        if not m:
            continue
        e = EPCH.search(line)
        rows.append((float(m["loss"]), float(m["grdn"]), float(m["lr"]),
                     float(e["epch"]) if e else None))
    return rows


def extract_eval(path: Path):
    """(step, val_loss) を出現順に（生 step）。"""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = EVAL.search(line)
        if m:
            rows.append((int(m["step"]), float(m["v"])))
    return rows


def rows_to_steps(rows, log_freq, step_offset):
    """行インデックス -> 復元した絶対step。"""
    return [step_offset + log_freq * (i + 1) for i in range(len(rows))]


def write_train_steps(rows, steps, writer, epoch_offset=0.0):
    for (loss, grdn, lr, epch), step in zip(rows, steps):
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/grad_norm", grdn, step)
        writer.add_scalar("train/lr", lr, step)
        if epch is not None:
            writer.add_scalar("train/epoch", epch + epoch_offset, step)
    writer.flush()


def write_eval(rows, writer, seen):
    for step, val in rows:
        if step in seen:
            continue
        seen.add(step)
        writer.add_scalar("val/loss", val, step)
    writer.flush()


def parse_segment(s: str) -> tuple[Path, int]:
    file_s, _, offset_s = s.rpartition(":")
    if not file_s:
        raise argparse.ArgumentTypeError(f"--segment は FILE:OFFSET 形式で指定 (got {s!r})")
    return Path(file_s), int(offset_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=parse_segment, action="append", required=True,
                     help="FILE:OFFSET を時系列順に指定(複数可)。OFFSET=そのファイル先頭行の実際のstep数"
                          "(新規学習なら0、resumeならcheckpoint step)")
    ap.add_argument("--logdir", type=Path, default=Path("tb"))
    ap.add_argument("--log-freq", type=int, default=100, help="学習時の log_freq（train step 復元用）")
    ap.add_argument("--follow", action="store_true", help="最後のsegmentの追記を待ち受けてライブ更新")
    a = ap.parse_args()

    a.logdir.mkdir(parents=True, exist_ok=True)
    for old in a.logdir.glob("events.out.tfevents.*"):
        old.unlink()
    writer = SummaryWriter(str(a.logdir))

    eval_seen: set[int] = set()
    n_last = 0  # 最後のsegmentで書き込み済みの行数(--follow継続用)
    epoch_base = 0.0         # 直前までの累積epoch(新規run検知用)
    last_epoch_offset = 0.0  # 最後のsegmentに適用するepoch offset(--follow継続用に確定させる)

    for idx, (path, offset) in enumerate(a.segment):
        is_last = idx == len(a.segment) - 1
        rows = extract_train(path)
        steps = rows_to_steps(rows, a.log_freq, offset)
        if not is_last:
            # 次segmentの開始stepより前だけ採用(resumeで再計算された重複区間は次を優先)
            next_offset = a.segment[idx + 1][1]
            keep = [(r, s) for r, s in zip(rows, steps) if s < next_offset]
            rows = [r for r, _ in keep]
            steps = [s for _, s in keep]

        # --resumeでの継続runはepochが連続するが、--policy.pathでの新規runは
        # lerobot内部のepochカウンタが0から数え直される。前segmentの最終epochより
        # 小さい値から始まっていたら「リセットされた」とみなし、continueするようoffsetを足す。
        raw_epochs = [r[3] for r in rows if r[3] is not None]
        epoch_offset = epoch_base if raw_epochs and raw_epochs[0] < epoch_base else 0.0

        write_train_steps(rows, steps, writer, epoch_offset)
        write_eval(extract_eval(path), writer, eval_seen)
        print(f"[{path}] wrote {len(rows)} train points (step {steps[0] if steps else '-'}..{steps[-1] if steps else '-'}, epoch_offset={epoch_offset:.2f})")
        if raw_epochs:
            epoch_base = epoch_offset + raw_epochs[-1]
        if is_last:
            n_last = len(extract_train(path))  # フィルタ前の実行数(follow差分用)
            last_epoch_offset = epoch_offset

    last_path, last_offset = a.segment[-1]
    print(f"total -> {a.logdir}")

    if a.follow:
        print("following (Ctrl-C で終了)...")
        try:
            while True:
                time.sleep(10)
                rows = extract_train(last_path)
                if len(rows) > n_last:
                    added_rows = rows[n_last:]
                    added_steps = [last_offset + a.log_freq * (i + 1) for i in range(n_last, len(rows))]
                    write_train_steps(added_rows, added_steps, writer, last_epoch_offset)
                    n_last = len(rows)
                    print(f"+{len(added_rows)} train points (total {n_last})")
                before = len(eval_seen)
                write_eval(extract_eval(last_path), writer, eval_seen)
                if len(eval_seen) > before:
                    print(f"+{len(eval_seen) - before} val points (total {len(eval_seen)})")
        except KeyboardInterrupt:
            pass
    writer.close()


if __name__ == "__main__":
    main()
