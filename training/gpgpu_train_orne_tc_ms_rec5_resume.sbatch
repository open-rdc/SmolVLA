#!/bin/bash
# gpgpu_train_orne_tc_ms_rec5.sbatch の続きから再開する版。
# 完走済み smolvla_orne_tc_ms(MEAN_STD, 1,150,000step) からファインチューニングする。
#
# 設計メモ:
#  - 復帰データ単体でのファインチューニングはしない。理由は2つ:
#    (1) lerobot_train.py は --policy.path 指定時、正規化統計を新データセットの
#        ものに上書きする(normalizer_processor overrides)。復帰データは旋回が
#        主役で dyaw の std がベースの約3倍あり、単体だと正規化が壊れる。
#    (2) 4560frame 対 300699frame なので破滅的忘却が起きる。
#  - 5倍複製(混合比6.6%)なら dyaw の std のズレは 1.23倍に収まる。
#  - 学習率はピーク 1e-5(ベース学習時の1/10)。継続学習なので低めにする。
#  - 180,000step は実測 4.3steps/s で約11.6時間。24h上限に収まるので resume 不要。
#
# 使い方: `sbatch training/gpgpu_train_orne_tc_ms_rec5.sbatch`
# Classmode で requeue された場合は held を scancel して
# `training/gpgpu_train_orne_tc_ms_rec5_resume.sbatch` を投げること。
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH -J smolvla_rec5_re
#SBATCH -o smolvla_orne_tc_ms_rec5_%j.out

set -euo pipefail

source ~/smolvla_venv/bin/activate
cd ~/SmolVLA

CKPT_DIR=~/SmolVLA/outputs/train/smolvla_orne_tc_ms_rec5/checkpoints
KEEP=3

# set -e 下で cd の失敗がバックグラウンドサブシェルごと殺すのを防ぐため `|| true`。
prune_checkpoints() {
  while true; do
    sleep 300
    (cd "$CKPT_DIR" 2>/dev/null && ls -1 | grep -E '^[0-9]+$' | sort -n | head -n -"$KEEP" | xargs -r rm -rf --) || true
  done
}
prune_checkpoints &
PRUNE_PID=$!
trap 'kill "$PRUNE_PID" 2>/dev/null' EXIT

lerobot-train \
  --resume=true \
  --config_path=$HOME/SmolVLA/outputs/train/smolvla_orne_tc_ms_rec5/checkpoints/last/pretrained_model/train_config.json
