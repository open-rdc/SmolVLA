#!/bin/bash
#SBATCH -p research
#SBATCH -t 4:00:00
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -J pvar_smol
#SBATCH -o /home/share/V23C1036/SmolVLA/probe_variants_smol_%j.out
cd $HOME/SmolVLA
$HOME/smolvla_venv/bin/python probe/probe_branch_variants.py --model smolvla --weights $HOME/SmolVLA/outputs/train/smolvla_fisheye_tsudanuma_v2/checkpoints/226926/pretrained_model --out probe/variants_smol_e14.npz
