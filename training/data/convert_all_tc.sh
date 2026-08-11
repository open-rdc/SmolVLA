#!/bin/bash
# orne-box_dataset の9サブセットを時系列3フレーム形式(_tc)に変換する。
# RAM が7GBしかない環境向けに、同時実行数を制限して1件ずつ空きが出たら次を投入する。
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> ~/ros2_ws/src/SmolVLA
PY=~/.venvs/smolvla/bin/python
RAW=training/data/raw/orne-box_dataset
LOGDIR=training/data/logs
mkdir -p "$LOGDIR"
MAXJOBS=2

pairs=(
  "navvla_tsudanuma_junction orne_junction_tc"
  "navvla_tsudanuma_linestop orne_linestop_tc"
  "navvla_tsudanuma_nav orne_nav_tc"
  "navvla_tsudanuma_random orne_random_tc"
  "navvla_tsudanuma_cosmos_junction orne_cosmos_junction_tc"
  "navvla_tsudanuma_cosmos_linestop orne_cosmos_linestop_tc"
  "navvla_tsudanuma_cosmos_nav orne_cosmos_nav_tc"
  "navvla_tsudanuma_cosmos_obstacle orne_cosmos_obstacle_tc"
  "navvla_tsudanuma_cosmos_random orne_cosmos_random_tc"
)

running=0
for pair in "${pairs[@]}"; do
  read -r raw out <<< "$pair"
  (
    echo "=== convert open-rdc/${out} start $(date) ==="
    "$PY" training/data/lerobot_dataset.py \
      --input "$RAW/$raw" \
      --repo-id "open-rdc/${out}" \
      --root "training/data/${out}"
    echo "=== convert open-rdc/${out} done $(date) exit=$? ==="
  ) > "$LOGDIR/convert_${out}.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$MAXJOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "ALL CONVERSIONS DONE $(date)" | tee "$LOGDIR/convert_all_tc.done"
