#!/bin/bash
# orne-box_dataset の cosmos系5サブセットを時系列3フレーム形式(_tc)に変換する。
# GPGPUクラスタ側(32コア/~1TB RAM)で実行する版。ローカル(RAM7GB)の
# convert_all_tc.sh と違い、5つ全部を同時並列で回せる。
set -uo pipefail
cd ~/SmolVLA
source ~/smolvla_venv/bin/activate
RAW=training/data/raw/orne-box_dataset
LOGDIR=training/data/logs
mkdir -p "$LOGDIR"

pairs=(
  "navvla_tsudanuma_cosmos_junction orne_cosmos_junction_tc"
  "navvla_tsudanuma_cosmos_linestop orne_cosmos_linestop_tc"
  "navvla_tsudanuma_cosmos_nav orne_cosmos_nav_tc"
  "navvla_tsudanuma_cosmos_obstacle orne_cosmos_obstacle_tc"
  "navvla_tsudanuma_cosmos_random orne_cosmos_random_tc"
)

for pair in "${pairs[@]}"; do
  read -r raw out <<< "$pair"
  (
    echo "=== convert open-rdc/${out} start $(date) ==="
    python training/data/lerobot_dataset.py \
      --input "$RAW/$raw" \
      --repo-id "open-rdc/${out}" \
      --root "training/data/${out}"
    echo "=== convert open-rdc/${out} done $(date) exit=$? ==="
  ) > "$LOGDIR/convert_${out}.log" 2>&1 &
done
wait
echo "ALL COSMOS CONVERSIONS DONE $(date)" | tee "$LOGDIR/convert_cosmos_tc.done"
