#!/bin/bash
# SmolVLA fine-tune on GPGPU (Slurm) — orne_box_all_tc / ACTION=MEAN_STD 版。
#
# gpgpu_train_orne_tc.sbatch (JOBID 13339/13348) と「正規化以外は完全に同一」の対照実験。
#
# ■ 何を変えたか
#   ACTION の正規化を IDENTITY → MEAN_STD に戻す(下の --policy.normalization_mapping)。
#
# ■ なぜ必要か
#   configuration_smolvla.py の既定は ACTION=IDENTITY だが、これは waypoint 形式
#   (action=[x,y,cos_yaw,sin_yaw] の4D絶対姿勢chunk)専用の設定である。4D のときは
#   WaypointRebaseProcessorStep が独自スケール(WAYPOINT_POS_SCALE=0.2)で割ってくれる
#   ので IDENTITY で辻褄が合う。
#   しかし orne_box_all_tc の action は 2D [dx_body, dyaw] で、
#   processor_smolvla.py:76 の
#       use_waypoint_actions = config.action_feature.shape[0] == 4
#   が False になるため rebase は無効。結果、ACTION はどちらの経路からも正規化されず
#   生値のまま学習されていた:
#       dx_body  mean 0.187  std 0.060
#       dyaw     mean 0.001  std 0.022
#   MSE への寄与は分散に比例するので dx:dyaw = 0.060^2 : 0.022^2 ≒ 7.3 : 1 となり、
#   旋回チャンネルが前進の約13%の重みしか持てない(=直進バイアス)。
#   MEAN_STD にすると両チャンネルとも平均0・分散1に揃い 1:1 になる。
#   nav201 など過去の body-delta run はすべて ACTION=MEAN_STD だったので、
#   学習曲線が再びそれらと比較可能になるという副次効果もある。
#
# ■ なぜコードを編集せず CLI override なのか
#   configuration_smolvla.py の既定値を書き換えると waypoint(4D)側が壊れる
#   (生の絶対座標の mean/std が適用され、docstring の言う「学習が破綻する」状態)。
#   normalization_mapping は draccus で dict ごと上書きできる(検証済み)ので、
#   この実験専用の設定として sbatch 側に閉じ込める。lerobot/ には一切触らない。
#
# ■ それ以外は gpgpu_train_orne_tc.sbatch と同一
#   データセット orne_box_all_tc (3,515ep / 300,699frame / 観測は現在・1秒前・2秒前の
#   3枚を camera1/2/3 に割り当て)、action expert はランダム初期化(--policy.path 未指定)、
#   VLM(SmolVLM2)のみ事前学習維持、state は全フレーム0(stateless)、batch_size=8。
#   steps も 1,150,000 (=30.6epoch) のまま揃える(エポック軸で直接重ねられるように)。
#
#   ※ 実測 ~4.4 steps/s (3カメラ分 vision encoder を通すため単一フレーム版の約半分)。
#      1,150,000 steps は約72h かかり 24h 上限に3回ぶつかる。時間切れごとに
#      gpgpu_train_orne_tc_ms_resume.sbatch を投入すること。
#
# 使い方: `sbatch training/gpgpu_train_orne_tc_ms.sbatch`
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -J smolvla_orne_tc_ms
#SBATCH -o smolvla_orne_tc_ms_%j.out

set -euo pipefail

source ~/smolvla_venv/bin/activate
cd ~/SmolVLA

CKPT_DIR=~/SmolVLA/outputs/train/smolvla_orne_tc_ms/checkpoints
KEEP=3

# 2026-07-11 の smolvla_all_ep40 事故の教訓: set -e 下では while ループ内で
# cd(チェックポイント未作成時に失敗) がそのまま非ゼロ終了として伝播し、
# このバックグラウンドループ自体が最初のイテレーションで死んで以後一切
# 間引きが効かなくなる（29個のckptが残り36GB消費→disk quota逼迫）。
# `|| true` で一周分の失敗を握りつぶし、ループを継続させる。
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
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"}' \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=1150000 \
  --dataset.repo_id=open-rdc/orne_box_all_tc \
  --dataset.root=$HOME/smolvla_data/orne_box_all_tc \
  --dataset.video_backend=pyav \
  --batch_size=8 --steps=1150000 --num_workers=4 \
  --save_freq=10000 --log_freq=100 \
  --output_dir=outputs/train/smolvla_orne_tc_ms \
  --job_name=smolvla_orne_tc_ms \
  --wandb.enable=false
