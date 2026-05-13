#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

WHAT=${WHAT:-all}
RUN_TAG=${RUN_TAG:-hard10}
PRINT_ONLY=${PRINT_ONLY:-0}

DATASET_JSON=${DATASET_JSON:-"data/real_benchmarks/upir/UPIR_open_bigbind_forward.json"}
PY=${PY:-"venv/bin/python"}

CPU_PARTITION=${CPU_PARTITION:-"i64m512ue"}
GPU_PARTITION=${GPU_PARTITION:-"i64m1tga800ue"}
QOS=${QOS:-}
ACCOUNT=${ACCOUNT:-}
INHERIT_SBATCH_QOS=${INHERIT_SBATCH_QOS:-0}

BASELINE_FOLDS=${BASELINE_FOLDS:-"5"}
LEARNED_FOLDS=${LEARNED_FOLDS:-"0,1"}
LEARNED_MODELS=${LEARNED_MODELS:-"B1,M1"}
LEARNED_SAVE_RANKINGS=${LEARNED_SAVE_RANKINGS:-"0"}
INCLUDE_SCAFFOLD_BASELINES=${INCLUDE_SCAFFOLD_BASELINES:-"0"}
INCLUDE_HARD_SLICE_ANALYSIS=${INCLUDE_HARD_SLICE_ANALYSIS:-"0"}

BASELINE_MIN_KNOWN_POS=${BASELINE_MIN_KNOWN_POS:-"1"}
BASELINE_MIN_KNOWN_NEG=${BASELINE_MIN_KNOWN_NEG:-"10"}
LEARNED_MIN_KNOWN_POS=${LEARNED_MIN_KNOWN_POS:-"1"}
LEARNED_MIN_KNOWN_NEG=${LEARNED_MIN_KNOWN_NEG:-"10"}

if [[ "${WHAT}" == "all" ]]; then
  echo "Submitting BigBind hard-slice campaign: baselines + learned on target_rec_cluster with >=10 known negatives/query."
elif [[ "${WHAT}" == "baselines" ]]; then
  echo "Submitting BigBind hard-slice baselines on target_rec_cluster with >=10 known negatives/query."
elif [[ "${WHAT}" == "learned" ]]; then
  echo "Submitting BigBind hard-slice learned runs on target_rec_cluster with >=10 known negatives/query."
else
  echo "Delegating WHAT=${WHAT} to submit_upir_top_tier_benchmark.sh"
fi

exec env \
  WHAT="${WHAT}" \
  RUN_TAG="${RUN_TAG}" \
  PRINT_ONLY="${PRINT_ONLY}" \
  DATASET_JSON="${DATASET_JSON}" \
  PY="${PY}" \
  CPU_PARTITION="${CPU_PARTITION}" \
  GPU_PARTITION="${GPU_PARTITION}" \
  QOS="${QOS}" \
  ACCOUNT="${ACCOUNT}" \
  INHERIT_SBATCH_QOS="${INHERIT_SBATCH_QOS}" \
  BASELINE_FOLDS="${BASELINE_FOLDS}" \
  LEARNED_FOLDS="${LEARNED_FOLDS}" \
  LEARNED_MODELS="${LEARNED_MODELS}" \
  LEARNED_SAVE_RANKINGS="${LEARNED_SAVE_RANKINGS}" \
  INCLUDE_SCAFFOLD_BASELINES="${INCLUDE_SCAFFOLD_BASELINES}" \
  INCLUDE_HARD_SLICE_ANALYSIS="${INCLUDE_HARD_SLICE_ANALYSIS}" \
  BASELINE_MIN_KNOWN_POS="${BASELINE_MIN_KNOWN_POS}" \
  BASELINE_MIN_KNOWN_NEG="${BASELINE_MIN_KNOWN_NEG}" \
  LEARNED_MIN_KNOWN_POS="${LEARNED_MIN_KNOWN_POS}" \
  LEARNED_MIN_KNOWN_NEG="${LEARNED_MIN_KNOWN_NEG}" \
  bash scripts/slurm/submit_upir_top_tier_benchmark.sh
