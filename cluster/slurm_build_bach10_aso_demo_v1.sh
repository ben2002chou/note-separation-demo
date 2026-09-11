#!/bin/bash -l
#SBATCH --job-name=bach10-aso-demo
#SBATCH --account=yunglu
#SBATCH --partition=a30,a10
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gilbreth/chou150/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/gilbreth/chou150/slurm_logs/%x_%j.err

source /etc/profile.d/modules.sh
module --force purge
set -euo pipefail

task=/scratch/gilbreth/chou150/datasets/bach10_official_eval_20260902
runs=/scratch/gilbreth/chou150/mt3_rt2_note_resynthesis/runs
snapshot=/scratch/gilbreth/chou150/mt3_rt2_note_resynthesis/source_snapshots/uncapped_acoustic_relative_aso_scns_eval_v2_v1b_20260910
python=/depot/yunglu/data/ben/.conda/envs/dna-model-polytune/bin/python
script=/scratch/gilbreth/chou150/note_separation_demo/build_bach10_aso_demo_v1.py
output=/scratch/gilbreth/chou150/note_separation_demo/bach10_aso_07_herrgott_v1

test ! -e "$output"
"$python" -u "$script" \
  --dna-root "$snapshot" \
  --dataset-root "$task/staged_v1/data/Bach10_v1.1-2a53cdc6495bb82f03f943b77ca3b11ddf7f5a31" \
  --piece 07-HerrGott \
  --best-checkpoint "$runs/occ_dual_hpss_input_only_symmetric_alignment_5000_sym30k_robust_v2_plus5k_11501464/checkpoints/best_monitor_spectral.pt" \
  --gated-checkpoint "$runs/occ_dual_hpss_below_f0_input_only_percussive_w2_30000_target_full_v1/checkpoints/step-000030000.pt" \
  --aso-authority /scratch/gilbreth/chou150/mt3_rt2_note_resynthesis/evidence/uncapped_acoustic_relative_aso_selection_v1_20260910/AUTHORITY.json \
  --nmf-root "$task/staged_v1/results/Bach10scoreinformedISMIR2017/audio/audioNMF" \
  --output-dir "$output"

sha256sum "$script" "$output/result.json" > "$output/run_sha256.txt"
date --iso-8601=seconds > "$output/COMPLETE"
