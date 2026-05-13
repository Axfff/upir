#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

mkdir -p logs/slurm

WHAT=${WHAT:-all}
RUN_TAG=${RUN_TAG:-rev1}
PRINT_ONLY=${PRINT_ONLY:-0}

DATASET_JSON=${DATASET_JSON:-"data/real_benchmarks/upir/UPIR_strict_reverse.json"}
PY=${PY:-"venv/bin/python"}

QOS=${QOS:-}
ACCOUNT=${ACCOUNT:-}
INHERIT_SBATCH_QOS=${INHERIT_SBATCH_QOS:-0}

CPU_PARTITION=${CPU_PARTITION:-"i64m512ue"}

BASELINE_TIME=${BASELINE_TIME:-"04:00:00"}
BASELINE_CPUS=${BASELINE_CPUS:-"4"}
BASELINE_MEM=${BASELINE_MEM:-"16G"}

LEARNED_TIME=${LEARNED_TIME:-"08:00:00"}
LEARNED_CPUS=${LEARNED_CPUS:-"4"}
LEARNED_MEM=${LEARNED_MEM:-"24G"}

SPLIT=${SPLIT:-"standard"}
FOLDS=${FOLDS:-"5"}
BASELINE_MODELS=${BASELINE_MODELS:-"RANDOM,POPULARITY"}
LEARNED_MODELS=${LEARNED_MODELS:-"B1"}
TOPK=${TOPK:-"1,5,10"}
LEARNED_MAX_TRAIN_PAIRS=${LEARNED_MAX_TRAIN_PAIRS:-"60000"}
LEARNED_MAX_EVAL_QUERIES=${LEARNED_MAX_EVAL_QUERIES:-"0"}
BASELINE_MAX_EVAL_QUERIES=${BASELINE_MAX_EVAL_QUERIES:-"0"}
MIN_KNOWN_POS=${MIN_KNOWN_POS:-"1"}
MIN_KNOWN_NEG=${MIN_KNOWN_NEG:-"1"}

submit_sbatch() {
  local -a submit_opts=()
  if [[ -n "${QOS}" ]]; then
    submit_opts+=(--qos="${QOS}")
  fi
  if [[ -n "${ACCOUNT}" ]]; then
    submit_opts+=(--account="${ACCOUNT}")
  fi

  if [[ "${INHERIT_SBATCH_QOS}" == "1" ]]; then
    sbatch "${submit_opts[@]}" "$@"
  else
    env -u SBATCH_QOS -u SLURM_QOS sbatch "${submit_opts[@]}" "$@"
  fi
}

submit_wrap() {
  local job_name="$1"
  local partition="$2"
  local time_limit="$3"
  local cpus="$4"
  local mem="$5"
  local cmd="$6"

  local -a args=(
    --job-name="${job_name}"
    --output="logs/slurm/%x_%j.out"
    --error="logs/slurm/%x_%j.err"
    --chdir="${ROOT_DIR}"
    --partition="${partition}"
    --time="${time_limit}"
    --cpus-per-task="${cpus}"
    --mem="${mem}"
  )

  if [[ "${PRINT_ONLY}" == "1" ]]; then
    echo "[PRINT_ONLY] sbatch ${args[*]} --wrap ${cmd}"
    return 0
  fi
  submit_sbatch "${args[@]}" --wrap "${cmd}"
}

emit_baseline_cmd() {
  local out_dir="$1"
  local extra=""
  if [[ "${BASELINE_MAX_EVAL_QUERIES}" != "0" ]]; then
    extra="${extra} --max-eval-queries ${BASELINE_MAX_EVAL_QUERIES}"
  fi
  echo "${PY} scripts/run_upir_reverse_official_baselines.py --dataset-path ${DATASET_JSON} --split ${SPLIT} --folds ${FOLDS} --models ${BASELINE_MODELS} --out-dir ${out_dir} --topk ${TOPK} --min-known-pos-override ${MIN_KNOWN_POS} --min-known-neg-override ${MIN_KNOWN_NEG}${extra}"
}

emit_learned_cmd() {
  local model="$1"
  local model_lc
  model_lc=$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')
  local run_id="upir_strict_reverse_${SPLIT}_${model_lc}_${RUN_TAG}"
  local extra="--max-train-pairs ${LEARNED_MAX_TRAIN_PAIRS} --min-known-pos-override ${MIN_KNOWN_POS} --min-known-neg-override ${MIN_KNOWN_NEG}"
  if [[ "${LEARNED_MAX_EVAL_QUERIES}" != "0" ]]; then
    extra="${extra} --max-eval-queries ${LEARNED_MAX_EVAL_QUERIES}"
  fi
  echo "${PY} scripts/run_bridge_experiment_reverse.py --run-id ${run_id} --dataset-path ${DATASET_JSON} --split ${SPLIT} --models ${model} --seed 0 --topk ${TOPK} ${extra}"
}

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Dataset JSON not found: ${DATASET_JSON}"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1 || ! sbatch --version >/dev/null 2>&1; then
  echo "sbatch unavailable. Printing equivalent local commands:"
  if [[ "${WHAT}" == "all" || "${WHAT}" == "baselines" ]]; then
    emit_baseline_cmd "results/upir_strict_reverse_official_baselines_${RUN_TAG}"
  fi
  if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
    IFS=',' read -r -a model_arr <<< "${LEARNED_MODELS}"
    for model in "${model_arr[@]}"; do
      emit_learned_cmd "${model}"
    done
  fi
  exit 0
fi

case "${WHAT}" in
  all|baselines|learned)
    ;;
  *)
    echo "Unsupported WHAT=${WHAT}. Use one of: all, baselines, learned"
    exit 2
    ;;
esac

if [[ "${WHAT}" == "all" || "${WHAT}" == "baselines" ]]; then
  out_dir="results/upir_strict_reverse_official_baselines_${RUN_TAG}"
  cmd=$(emit_baseline_cmd "${out_dir}")
  echo "Submitting upir_strict_reverse_baselines_${RUN_TAG}"
  submit_wrap "upir_rev_base_${RUN_TAG}" "${CPU_PARTITION}" "${BASELINE_TIME}" "${BASELINE_CPUS}" "${BASELINE_MEM}" "${cmd}"
fi

if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
  IFS=',' read -r -a model_arr <<< "${LEARNED_MODELS}"
  for model in "${model_arr[@]}"; do
    model_lc=$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')
    echo "Submitting upir_strict_reverse_${SPLIT}_${model_lc}_${RUN_TAG}"
    submit_wrap \
      "upir_rev_${model_lc}_${RUN_TAG}" \
      "${CPU_PARTITION}" \
      "${LEARNED_TIME}" \
      "${LEARNED_CPUS}" \
      "${LEARNED_MEM}" \
      "$(emit_learned_cmd "${model}")"
  done
fi
