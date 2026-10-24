#!/bin/bash
#SBATCH -p research
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -J conv_ftv2
#SBATCH -o convert_fisheye_tsudanuma_v2_%j.out
# 【H100(津田沼 / gpumng2.cle.it-chiba.ac.jp)用】
# 魚眼の津田沼 16 周(~/navvla_data/fisheye_tsudanuma_v2, OmniVLA-edge と同じ周回データ)を
# SmolVLA 用の LeRobotDataset(時系列 3 フレーム, action=[dx_body, dyaw], 画像は 640x480 を切り出さず 224x224 に縮小)にする。
# 分割は OmniVLA-edge と同じ: 学習用 15 周 / 検証用 tsudanuma_20261010_134655。
#   ~/smolvla_data/fisheye_tsudanuma_v2_train / fisheye_tsudanuma_v2_test
# CPU だけ使う。2 つを並列に変換する。
#
# 使い方: `sbatch training/data/convert_fisheye_tsudanuma_v2.sbatch`
set -euo pipefail
source $HOME/smolvla_venv/bin/activate
cd $HOME/SmolVLA
SRC=$HOME/navvla_data/fisheye_tsudanuma_v2
TEST=tsudanuma_20261010_134655
TMP=$HOME/smolvla_data/_ftv2_input
rm -rf "$TMP"
mkdir -p "$TMP/train" "$TMP/test"
for d in "$SRC"/tsudanuma_*; do
  n=$(basename "$d")
  if [ "$n" = "$TEST" ]; then ln -s "$d" "$TMP/test/$n"; else ln -s "$d" "$TMP/train/$n"; fi
done
echo "train $(ls $TMP/train | wc -l) / test $(ls $TMP/test | wc -l)"
for s in train test; do
  python training/data/lerobot_dataset.py --input "$TMP/$s" \
    --repo-id open-rdc/fisheye_tsudanuma_v2_$s --root $HOME/smolvla_data/fisheye_tsudanuma_v2_$s \
    > convert_fisheye_tsudanuma_v2_$s.log 2>&1 &
done
wait
tail -n 3 convert_fisheye_tsudanuma_v2_*.log
for s in train test; do
  python -c "import json;i=json.load(open('$HOME/smolvla_data/fisheye_tsudanuma_v2_$s/meta/info.json'));print('$s', i['total_episodes'], i['total_frames'], i['total_tasks'])"
done
echo "CONV_FTV2_DONE $(date)"
