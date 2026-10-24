#!/bin/bash
#SBATCH -p research
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -J pvar_omni
#SBATCH -o /home/share/V23C1036/SmolVLA/probe_variants_omni_%j.out
cd $HOME/SmolVLA
$HOME/navvla_venv/bin/python probe/probe_branch_variants.py --model omnivla --weights $HOME/NavVLA/training/runs/fisheye_tsudanuma_v2_omnivla/model_epoch_022.pth --out probe/variants_omni_e22.npz
