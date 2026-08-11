#!/bin/bash
# 時系列3フレーム版9サブセットを1つの LeRobotDataset (open-rdc/orne_box_all_tc) にマージする。
# 各サブセットの episodes/frames は事前に info.json で検証済み(合計3515ep/300699frame、
# 生データのn-1合計と一致を確認済み)。
#
# 使い方: `sbatch training/data/merge_orne_box_all_tc.sbatch`
#SBATCH -p research
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J merge_orne_box_all_tc
#SBATCH -o merge_orne_box_all_tc_%j.out

set -euo pipefail
cd ~/SmolVLA
source ~/smolvla_venv/bin/activate

lerobot-edit-dataset \
  --new_repo_id open-rdc/orne_box_all_tc \
  --new_root "$HOME/SmolVLA/training/data/orne_box_all_tc" \
  --operation.type merge \
  --operation.repo_ids "['open-rdc/orne_junction_tc','open-rdc/orne_linestop_tc','open-rdc/orne_nav_tc','open-rdc/orne_random_tc','open-rdc/orne_cosmos_junction_tc','open-rdc/orne_cosmos_linestop_tc','open-rdc/orne_cosmos_nav_tc','open-rdc/orne_cosmos_obstacle_tc','open-rdc/orne_cosmos_random_tc']" \
  --operation.roots "['$HOME/SmolVLA/training/data/orne_junction_tc','$HOME/SmolVLA/training/data/orne_linestop_tc','$HOME/SmolVLA/training/data/orne_nav_tc','$HOME/SmolVLA/training/data/orne_random_tc','$HOME/SmolVLA/training/data/orne_cosmos_junction_tc','$HOME/SmolVLA/training/data/orne_cosmos_linestop_tc','$HOME/SmolVLA/training/data/orne_cosmos_nav_tc','$HOME/SmolVLA/training/data/orne_cosmos_obstacle_tc','$HOME/SmolVLA/training/data/orne_cosmos_random_tc']"

echo "MERGE_DONE $(date) exit=$?"
