#!/bin/bash
# orne_box_all_tc(ベース 3515ep/300699frame) に復帰データ(158ep/4560frame)を
# 5倍複製して混ぜた open-rdc/orne_box_all_tc_rec5 を作る。
#
# 5倍にする理由: ファインチューニング時、正規化統計は新データセットのもので
# 上書きされる(lerobot_train.py の normalizer_processor overrides)。復帰データは
# 旋回が主役で dyaw の std がベースの約3倍あるため、混合比を上げすぎると
# 学習済みの行動ヘッドに対して正規化がずれる。5倍(混合比6.6%)なら dyaw の
# std のズレは 1.23倍に収まる(10倍だと 1.40倍)。
#
# 期待値: 158*5+3515 = 4305 episodes / 4560*5+300699 = 323499 frames
#
# 使い方: `sbatch training/data/merge_orne_box_all_tc_recovery.sbatch`
#SBATCH -p research
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH -J merge_rec5
#SBATCH -o merge_rec5_%j.out

set -euo pipefail
cd ~/SmolVLA
source ~/smolvla_venv/bin/activate

D="$HOME/smolvla_data"

lerobot-edit-dataset \
  --new_repo_id open-rdc/orne_box_all_tc_rec5 \
  --new_root "$D/orne_box_all_tc_rec5" \
  --operation.type merge \
  --operation.repo_ids "['open-rdc/orne_box_all_tc','open-rdc/orne_recovery_tc','open-rdc/orne_recovery_tc_c2','open-rdc/orne_recovery_tc_c3','open-rdc/orne_recovery_tc_c4','open-rdc/orne_recovery_tc_c5']" \
  --operation.roots "['$D/orne_box_all_tc','$D/orne_recovery_tc','$D/orne_recovery_tc_c2','$D/orne_recovery_tc_c3','$D/orne_recovery_tc_c4','$D/orne_recovery_tc_c5']"

echo "MERGE_DONE $(date) exit=$?"
python - <<'PY'
import json
d = json.load(open("/home/share/V23C1036/smolvla_data/orne_box_all_tc_rec5/meta/info.json"))
ep, fr = d["total_episodes"], d["total_frames"]
print(f"RESULT episodes={ep} frames={fr}")
assert ep == 4305 and fr == 323499, f"期待値 4305ep/323499frame と不一致"
print("OK: 期待値と一致")
PY
