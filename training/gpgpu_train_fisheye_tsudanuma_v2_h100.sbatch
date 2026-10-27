#!/bin/bash
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 3
#SBATCH --mem=64G
#SBATCH -J smol_ftv2
#SBATCH -o smolvla_fisheye_tsudanuma_v2_%j.out
# 【H100(津田沼 / gpumng2.cle.it-chiba.ac.jp)用】
# 魚眼の津田沼だけ(注釈は 2026-10-20 に見直しが終わったもの)の学習用 fisheye_tsudanuma_v2_train
# (15 周 / 129,667 frame)で、一から 30 エポック学習する。OmniVLA-edge の fisheye_tsudanuma_v2 と同じ分割。
#
# ■ レシピは gpgpu_train_rec3x_ho2_h100.sbatch と同一(batch 8 / lr 1e-4 / cosine)。違いはデータと step 数
#   30エポック = 129,667 x 30 / 8 = 486,252 step(H100 で約 13 時間)
#   save_freq = 16,209 step (1エポック) で、チェックポイントを全部残す。
#   毎エポックの重みを検証用で評価して最適なエポックを選ぶため(training/eval_monitor.py)。
#
# 使い方: `sbatch --dependency=afterok:<変換ジョブ> training/gpgpu_train_fisheye_tsudanuma_v2_h100.sbatch`

set -euo pipefail
source $HOME/smolvla_venv/bin/activate
cd $HOME/SmolVLA
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=486252 \
  --dataset.repo_id=open-rdc/fisheye_tsudanuma_v2_train \
  --dataset.root=$HOME/smolvla_data/fisheye_tsudanuma_v2_train \
  --dataset.video_backend=pyav \
  --batch_size=8 --steps=486252 --num_workers=3 \
  --save_freq=16209 --log_freq=100 \
  --output_dir=outputs/train/smolvla_fisheye_tsudanuma_v2 \
  --job_name=smolvla_fisheye_tsudanuma_v2 \
  --wandb.enable=false
