#!/bin/bash
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -J eval_ftv2
#SBATCH -o eval_monitor_fisheye_tsudanuma_v2_%j.out
# 【H100(津田沼 / gpumng2.cle.it-chiba.ac.jp)用】
# 魚眼の津田沼だけで学習した SmolVLA / OmniVLA-edge の毎エポックの重みを、検証用の
# tsudanuma_20261010_134655 から固定で選んだ 1,000 フレームで評価して TensorBoard に書く(eval_monitor.py)。
# OmniVLA-edge は事前学習済みの重みも epoch 0 として評価する。
# 出力: ~/SmolVLA/tb_eval_fisheye_tsudanuma_v2/{fisheye_tsudanuma_v2, fisheye_tsudanuma_v2_omnivla}/
#
# 使い方:
#   常駐(10 分おきに新しいチェックポイントを評価): sbatch --dependency=afterok:<変換ジョブ> <この sbatch>
#   学習が全部終わった後の取りこぼし分:          sbatch --dependency=afterany:<学習2本> --export=ALL,ONCE=1 <この sbatch>

set -uo pipefail
cd $HOME/SmolVLA
[ -n "${ONCE:-}" ] && sleep 180   # 書き込みから 120 秒経っていないチェックポイントは読まないので待つ

$HOME/navvla_venv/bin/python training/eval_monitor.py \
  --smolvla-run fisheye_tsudanuma_v2 \
  --omnivla-run fisheye_tsudanuma_v2_omnivla --omnivla-pretrained \
  --val-root $HOME/smolvla_data/fisheye_tsudanuma_v2_test \
  --val-repo-id open-rdc/fisheye_tsudanuma_v2_test \
  --map $HOME/SmolVLA/training/data/splits/fisheye_tsudanuma_v2_episode_map.json \
  --raw-pkl $HOME/navvla_data/fisheye_tsudanuma_v2 \
  --omni-config fisheye_tsudanuma_v2 --omni-dataset fisheye_tsudanuma_v2 \
  --tb-root $HOME/SmolVLA/tb_eval_fisheye_tsudanuma_v2 \
  --train-frames 129667 --n-cosmos 0 \
  ${ONCE:+--once}
