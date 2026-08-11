#!/bin/bash
# orne-box_dataset cosmos系5サブセットの時系列3フレーム(_tc)変換をGPGPUクラスタで実行。
# CPUのみ使用（GPU不要）。ローカル(RAM7GB)だと2並列が限界で数時間かかるため、
# 32コア/~1TB RAMのノードで5並列フル同時実行して短縮する。
#
# 使い方: `sbatch training/data/convert_cosmos_tc.sbatch`
#SBATCH -p research
#SBATCH -t 02:00:00
#SBATCH -c 10
#SBATCH -J convert_cosmos_tc
#SBATCH -o convert_cosmos_tc_%j.out

set -uo pipefail
bash ~/SmolVLA/training/data/convert_cosmos_tc_cluster.sh
