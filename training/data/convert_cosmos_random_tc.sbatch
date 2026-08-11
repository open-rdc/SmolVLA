#!/bin/bash
# orne-box_dataset の cosmos_random サブセット単体の時系列3フレーム(_tc)変換。
# convert_cosmos_tc.sbatch(5並列, -c 10)が2h時間切れでcosmos_randomだけ
# 1118/1198(93%)で中断→単体-t 1hで再実行するも664/1198(55%)で再度時間切れ。
# 実測ペース(~11ep/min)だと1198話フルには約110分かかるため、余裕を見て3hに設定。
#
# 使い方: `sbatch training/data/convert_cosmos_random_tc.sbatch`
#SBATCH -p research
#SBATCH -t 03:00:00
#SBATCH -c 16
#SBATCH -J convert_cosmos_random_tc
#SBATCH -o convert_cosmos_random_tc_%j.out

set -uo pipefail
cd ~/SmolVLA
source ~/smolvla_venv/bin/activate
python training/data/lerobot_dataset.py \
  --input training/data/raw/orne-box_dataset/navvla_tsudanuma_cosmos_random \
  --repo-id open-rdc/orne_cosmos_random_tc \
  --root training/data/orne_cosmos_random_tc
echo "COSMOS_RANDOM_TC_DONE $(date) exit=$?"
