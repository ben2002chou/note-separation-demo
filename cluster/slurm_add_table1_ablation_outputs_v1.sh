#!/usr/bin/env bash
#SBATCH --job-name=scv3-table1-demo
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
readonly METRICS=/scratch/gilbreth/chou150/mt3_rt2_note_resynthesis/benchmarks/goodsounds_fullpiece_v1/model_comparison_v1/control/evaluate_models_144.py
readonly SOURCE=/scratch/gilbreth/chou150/scns_v3/demos/final_temporal_aso_v4_curated_v1_20260911
readonly OUTPUT=/scratch/gilbreth/chou150/scns_v3/demos/final_temporal_aso_v4_table1_curated_v1_20260911
readonly RUN=/scratch/gilbreth/chou150/scns_v3/runs/identity5_2x2_30k_v2_20260910

exec "$PYTHON" "$SCNS_DEMO_SCRIPT" \
  --snapshot "$SNAPSHOT" \
  --manifest "$MANIFEST" --manifest-sha256 bdedc67d0402283d9162674abaff45c7aa9a700dce3fc0a3569d6fe6c4c8625a \
  --frozen-metrics "$METRICS" \
  --source-root "$SOURCE" --output-root "$OUTPUT" \
  --piece scns_eval_v3_101 --piece scns_eval_v3_003 --piece scns_eval_v3_028 \
  --arm-checkpoint independent_ungated "$RUN/independent_ungated/checkpoints/step-000030000.pt" f25a2c801cc52ad3f79deb22c025327937e16074540e927abd68bb74c987e726 \
  --arm-checkpoint independent_selective "$RUN/independent_selective/checkpoints/step-000030000.pt" f1c92be5b78345102e84c9b3126301460149ac4989571c53de2dd289de6377b7 \
  --arm-checkpoint symmetric_ungated "$RUN/symmetric_ungated/checkpoints/step-000030000.pt" ccea982e6694c757708cec92939f41bebedab3f315ed8ffbee3503c9ed8398bf \
  --arm-checkpoint symmetric_selective "$RUN/symmetric_selective/checkpoints/step-000030000.pt" a3535f3ad99a297967dd18a4c4b8a9a8f7d47f3c6b05f72c0e2449d8c4be2d68
