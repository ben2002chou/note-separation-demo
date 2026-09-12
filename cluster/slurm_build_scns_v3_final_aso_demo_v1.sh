#!/usr/bin/env bash
#SBATCH --job-name=scv3-final-aso-demo
#SBATCH --account=yunglu
#SBATCH --qos=standby
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/gilbreth/chou150/scns_v3/logs/%x-%j.out
#SBATCH --error=/scratch/gilbreth/chou150/scns_v3/logs/%x-%j.err
set -euo pipefail
module --force purge
module load rcac cuda/12.1.1

readonly SNAPSHOT=/scratch/gilbreth/chou150/scns_v3/authorities/code_snapshots/identity5_mature_symmetric_selective_30k_aso_v4_20260911
readonly PYTHON=/depot/yunglu/data/ben/.conda/envs/dna-model-polytune/bin/python
readonly MANIFEST=/scratch/gilbreth/chou150/scns_v3/authorities/scns_eval_v3_factorial_manifest_v1_20260909/manifest.json
readonly SEPARATOR=/scratch/gilbreth/chou150/scns_v3/runs/identity5_2x2_30k_v2_20260910/symmetric_selective/checkpoints/step-000030000.pt
readonly ASO=/scratch/gilbreth/chou150/scns_v3/runs/identity5_mature_symmetric_selective_30k_uncapped_temporal_aso_v4_20260911/acoustic_multiplicative/checkpoint-step-029058.pt
readonly RESULT=/scratch/gilbreth/chou150/scns_v3/evals/identity5_mature_symmetric_selective_30k_uncapped_temporal_aso_v4_eval_v3_full_v1_20260911/result.json
readonly METRICS=/scratch/gilbreth/chou150/mt3_rt2_note_resynthesis/benchmarks/goodsounds_fullpiece_v1/model_comparison_v1/control/evaluate_models_144.py
readonly OUTPUT=/scratch/gilbreth/chou150/scns_v3/demos/final_temporal_aso_v4_curated_v1_20260911

exec "$PYTHON" "$SCNS_DEMO_SCRIPT" \
  --snapshot "$SNAPSHOT" \
  --manifest "$MANIFEST" --manifest-sha256 bdedc67d0402283d9162674abaff45c7aa9a700dce3fc0a3569d6fe6c4c8625a \
  --frozen-metrics "$METRICS" \
  --separator-checkpoint "$SEPARATOR" --separator-sha256 a3535f3ad99a297967dd18a4c4b8a9a8f7d47f3c6b05f72c0e2449d8c4be2d68 \
  --aso-checkpoint "$ASO" --aso-sha256 ff89599cef65e5eb98ff061372d8362e0752d2060d167aaf553b24c29577ae2a \
  --aggregate-result "$RESULT" --aggregate-sha256 "$SCNS_RESULT_SHA256" \
  --piece scns_eval_v3_101 --piece scns_eval_v3_003 --piece scns_eval_v3_028 \
  --output-root "$OUTPUT"
