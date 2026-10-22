#!/bin/bash
#SBATCH -p research
#SBATCH -t 2:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -J probe_lay
#SBATCH -o /home/share/V23C1036/SmolVLA/probe_layers_%j.out
cd $HOME/SmolVLA
for w in e22=$HOME/NavVLA/training/runs/fisheye_tsudanuma_v2_omnivla/model_epoch_022.pth pre=$HOME/NavVLA/training/weights/omnivla-edge.pth; do
  echo "### ${w%%=*}"
  $HOME/navvla_venv/bin/python probe/probe_omni_layers.py --branch probe/branch_134655_ref.npz --weights ${w#*=} --out probe/layers_${w%%=*}.npz
done
