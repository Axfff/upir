#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

mkdir -p logs/slurm

WHAT=${WHAT:-all}
RUN_TAG=${RUN_TAG:-b1cg2}
PRINT_ONLY=${PRINT_ONLY:-0}

DATASET_JSON=${DATASET_JSON:-"data/real_benchmarks/upir/UPIR_open_bigbind_forward.json"}
PY=${PY:-"venv/bin/python"}

QOS=${QOS:-}
ACCOUNT=${ACCOUNT:-}
INHERIT_SBATCH_QOS=${INHERIT_SBATCH_QOS:-0}

GPU_PARTITION=${GPU_PARTITION:-"i64m1tga800ue"}
CPU_PARTITION=${CPU_PARTITION:-"i64m512ue"}

LEARNED_TIME=${LEARNED_TIME:-"12:00:00"}
LEARNED_CPUS=${LEARNED_CPUS:-"4"}
LEARNED_MEM=${LEARNED_MEM:-"32G"}
LEARNED_GRES=${LEARNED_GRES:-"gpu:1"}
LEARNED_SCORE_BATCH_SIZE=${LEARNED_SCORE_BATCH_SIZE:-"4096"}
LEARNED_CACHE_DIR=${LEARNED_CACHE_DIR:-"results/encoding_cache"}
LEARNED_MAX_TRAIN_PAIRS=${LEARNED_MAX_TRAIN_PAIRS:-"60000"}
LEARNED_STREAM_EVAL=${LEARNED_STREAM_EVAL:-"1"}
LEARNED_SAVE_RANKINGS=${LEARNED_SAVE_RANKINGS:-"0"}
LEARNED_FORCE_REENCODE=${LEARNED_FORCE_REENCODE:-"0"}

AGG_TIME=${AGG_TIME:-"00:30:00"}
AGG_CPUS=${AGG_CPUS:-"2"}
AGG_MEM=${AGG_MEM:-"8G"}

FOLDS=${FOLDS:-"0,1,2,3,4"}
MODEL=${MODEL:-"B1"}
SPLIT=${SPLIT:-"target_rec_cluster"}
TOPK=${TOPK:-"10,50"}
EF_PERCENTS=${EF_PERCENTS:-"1,5"}
MIN_KNOWN_POS=${MIN_KNOWN_POS:-"1"}
MIN_KNOWN_NEG=${MIN_KNOWN_NEG:-"0"}

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
  local gres="$6"
  local cmd="$7"

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
  if [[ -n "${gres}" ]]; then
    args+=(--gres="${gres}")
  fi

  if [[ "${PRINT_ONLY}" == "1" ]]; then
    echo "[PRINT_ONLY] sbatch ${args[*]} --wrap ${cmd}"
    return 0
  fi

  submit_sbatch "${args[@]}" --wrap "${cmd}"
}

emit_learned_cmd() {
  local fold="$1"
  local model="$2"
  local model_lc
  model_lc=$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')
  local run_id="upir_open_bigbind_${SPLIT}_seed${fold}_${model_lc}_${RUN_TAG}"
  local extra_args="--dataset-path ${DATASET_JSON} --topk ${TOPK} --ef-percents ${EF_PERCENTS} --score-batch-size ${LEARNED_SCORE_BATCH_SIZE} --cache-dir ${LEARNED_CACHE_DIR} --max-train-pairs ${LEARNED_MAX_TRAIN_PAIRS} --min-known-pos-override ${MIN_KNOWN_POS} --min-known-neg-override ${MIN_KNOWN_NEG}"
  if [[ "${LEARNED_STREAM_EVAL}" == "1" ]]; then
    extra_args="${extra_args} --stream-eval"
  fi
  if [[ "${LEARNED_SAVE_RANKINGS}" == "1" ]]; then
    extra_args="${extra_args} --save-rankings"
  fi
  if [[ "${LEARNED_FORCE_REENCODE}" == "1" ]]; then
    extra_args="${extra_args} --force-reencode"
  fi
  echo "${PY} scripts/run_bridge_experiment.py --run-id ${run_id} --dataset upir --split ${SPLIT} --models ${model} --seed ${fold} ${extra_args}"
}

emit_aggregate_cmd() {
  echo "${PY} scripts/aggregate_upir_learned_candidate_generation.py --run-tag ${RUN_TAG} --folds ${FOLDS} --model ${MODEL}"
}

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Dataset JSON not found: ${DATASET_JSON}"
  exit 2
fi

case "${WHAT}" in
  all|learned|aggregate)
    ;;
  *)
    echo "Unsupported WHAT=${WHAT}. Use one of: all, learned, aggregate"
    exit 2
    ;;
esac

IFS=',' read -r -a fold_arr <<< "${FOLDS}"

if ! command -v sbatch >/dev/null 2>&1 || ! sbatch --version >/dev/null 2>&1; then
  echo "sbatch unavailable. Printing equivalent local commands:"
  if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
    for fold in "${fold_arr[@]}"; do
      emit_learned_cmd "${fold}" "${MODEL}"
    done
  fi
  if [[ "${WHAT}" == "all" || "${WHAT}" == "aggregate" ]]; then
    emit_aggregate_cmd
  fi
  exit 0
fi

job_ids=()
if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
  model_lc=$(printf '%s' "${MODEL}" | tr '[:upper:]' '[:lower:]')
  for fold in "${fold_arr[@]}"; do
    run_id="upir_open_bigbind_${SPLIT}_seed${fold}_${model_lc}_${RUN_TAG}"
    cmd=$(emit_learned_cmd "${fold}" "${MODEL}")

    echo "Submitting ${run_id}"
    if [[ "${PRINT_ONLY}" == "1" ]]; then
      submit_wrap \
        "upir_fwd_${model_lc}cg_${fold}_${RUN_TAG}" \
        "${GPU_PARTITION}" \
        "${LEARNED_TIME}" \
        "${LEARNED_CPUS}" \
        "${LEARNED_MEM}" \
        "${LEARNED_GRES}" \
        "${cmd}"
    else
      submit_out=$(submit_wrap \
        "upir_fwd_${model_lc}cg_${fold}_${RUN_TAG}" \
        "${GPU_PARTITION}" \
        "${LEARNED_TIME}" \
        "${LEARNED_CPUS}" \
        "${LEARNED_MEM}" \
        "${LEARNED_GRES}" \
        "${cmd}")
      echo "${submit_out}"
      job_ids+=("$(printf '%s\n' "${submit_out}" | awk '/Submitted batch job/ {print $4}')")
    fi
  done
fi

if [[ "${WHAT}" == "all" || "${WHAT}" == "aggregate" ]]; then
  agg_cmd=$(emit_aggregate_cmd)
  if [[ "${PRINT_ONLY}" == "1" ]]; then
    echo "[PRINT_ONLY] sbatch --partition=${CPU_PARTITION} --time=${AGG_TIME} --cpus-per-task=${AGG_CPUS} --mem=${AGG_MEM} --wrap ${agg_cmd}"
  else
    dep_args=()
    if [[ "${WHAT}" == "all" && "${#job_ids[@]}" -gt 0 ]]; then
      dep=$(IFS=:; echo "${job_ids[*]}")
      dep_args+=(--dependency="afterok:${dep}")
    fi
    echo "Submitting aggregate job for ${RUN_TAG}"
    submit_sbatch \
      --job-name="upir_fwd_b1cg_agg_${RUN_TAG}" \
      --output="logs/slurm/%x_%j.out" \
      --error="logs/slurm/%x_%j.err" \
      --chdir="${ROOT_DIR}" \
      --partition="${CPU_PARTITION}" \
      --time="${AGG_TIME}" \
      --cpus-per-task="${AGG_CPUS}" \
      --mem="${AGG_MEM}" \
      "${dep_args[@]}" \
      --wrap "${agg_cmd}"
  fi
fi
