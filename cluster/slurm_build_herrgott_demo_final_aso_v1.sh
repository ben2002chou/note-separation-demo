#!/usr/bin/env bash
#SBATCH --job-name=herrgott-demo-final-aso
#SBATCH --account=yunglu
#SBATCH --partition=a100-80gb,a100-40gb,a30,a10
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gilbreth/chou150/scns_v3/logs/%x-%j.out
#SBATCH --error=/scratch/gilbreth/chou150/scns_v3/logs/%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module --force purge
module load rcac cuda/12.1.1
set -u

readonly SNAPSHOT=/scratch/gilbreth/chou150/scns_v3/authorities/code_snapshots/identity5_mature_symmetric_selective_30k_aso_v4_20260911
readonly PYTHON=/depot/yunglu/data/ben/.conda/envs/dna-model-polytune/bin/python
readonly DATASET=/scratch/gilbreth/chou150/datasets/bach10_official_eval_20260902/staged_v1/data/Bach10_v1.1-2a53cdc6495bb82f03f943b77ca3b11ddf7f5a31
readonly NMF=/scratch/gilbreth/chou150/datasets/bach10_official_eval_20260902/staged_v1/results/Bach10scoreinformedISMIR2017/audio/audioNMF
readonly SEPARATOR=/scratch/gilbreth/chou150/scns_v3/runs/identity5_2x2_30k_v2_20260910/symmetric_selective/checkpoints/step-000030000.pt
readonly ASO=/scratch/gilbreth/chou150/scns_v3/runs/identity5_mature_symmetric_selective_30k_uncapped_temporal_aso_v4_20260911/acoustic_multiplicative/checkpoint-step-029058.pt
readonly SCRIPT=/scratch/gilbreth/chou150/scns_v3/demos/build_bach10_final_aso_demo_v2.py
readonly OUTPUT=/scratch/gilbreth/chou150/scns_v3/demos/herrgott_selected_final_aso_v1_20260912

test ! -e "$OUTPUT"
exec "$PYTHON" -u "$SCRIPT" \
  --dna-root "$SNAPSHOT" \
  --dataset-root "$DATASET" \
  --piece 07-HerrGott \
  --event 176 --event 180 --event 182 --event 183 \
  --event 184 --event 185 --event 186 --event 187 \
  --separator-checkpoint "$SEPARATOR" \
  --aso-checkpoint "$ASO" \
  --nmf-root "$NMF" \
  --output-dir "$OUTPUT"
