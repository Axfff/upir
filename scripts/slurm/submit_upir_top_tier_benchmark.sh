#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${ROOT_DIR}"

mkdir -p logs/slurm

WHAT=${WHAT:-all}
RUN_TAG=${RUN_TAG:-tt1}
DATASET_JSON=${DATASET_JSON:-"data/real_benchmarks/upir/UPIR_open_bigbind_forward.json"}
PY=${PY:-"venv/bin/python"}

QOS=${QOS:-}
ACCOUNT=${ACCOUNT:-}
INHERIT_SBATCH_QOS=${INHERIT_SBATCH_QOS:-0}
PRINT_ONLY=${PRINT_ONLY:-0}

CPU_PARTITION=${CPU_PARTITION:-"i64m512ue"}
GPU_PARTITION=${GPU_PARTITION:-"i64m1tga800ue"}

BASELINE_TIME=${BASELINE_TIME:-"06:00:00"}
BASELINE_CPUS=${BASELINE_CPUS:-"8"}
BASELINE_MEM=${BASELINE_MEM:-"48G"}

HARD_SLICE_TIME=${HARD_SLICE_TIME:-"02:00:00"}
HARD_SLICE_CPUS=${HARD_SLICE_CPUS:-"4"}
HARD_SLICE_MEM=${HARD_SLICE_MEM:-"16G"}

LEARNED_TIME=${LEARNED_TIME:-"12:00:00"}
LEARNED_CPUS=${LEARNED_CPUS:-"4"}
LEARNED_MEM=${LEARNED_MEM:-"32G"}
LEARNED_GRES=${LEARNED_GRES:-"gpu:1"}

BASELINE_TOPK=${BASELINE_TOPK:-"10,50"}
BASELINE_EF_PERCENTS=${BASELINE_EF_PERCENTS:-"1,5"}
BASELINE_FOLDS=${BASELINE_FOLDS:-"5"}
BASELINE_MIN_KNOWN_POS=${BASELINE_MIN_KNOWN_POS:-"-1"}
BASELINE_MIN_KNOWN_NEG=${BASELINE_MIN_KNOWN_NEG:-"-1"}
INCLUDE_SCAFFOLD_BASELINES=${INCLUDE_SCAFFOLD_BASELINES:-"1"}
INCLUDE_HARD_SLICE_ANALYSIS=${INCLUDE_HARD_SLICE_ANALYSIS:-"1"}

LEARNED_FOLDS=${LEARNED_FOLDS:-"0,1"}
LEARNED_MODELS=${LEARNED_MODELS:-"B1,M1"}
LEARNED_SCORE_BATCH_SIZE=${LEARNED_SCORE_BATCH_SIZE:-"4096"}
LEARNED_STREAM_EVAL=${LEARNED_STREAM_EVAL:-"1"}
LEARNED_SAVE_RANKINGS=${LEARNED_SAVE_RANKINGS:-"0"}
LEARNED_FORCE_REENCODE=${LEARNED_FORCE_REENCODE:-"0"}
LEARNED_CACHE_DIR=${LEARNED_CACHE_DIR:-"results/encoding_cache"}
LEARNED_MIN_KNOWN_POS=${LEARNED_MIN_KNOWN_POS:-"-1"}
LEARNED_MIN_KNOWN_NEG=${LEARNED_MIN_KNOWN_NEG:-"-1"}

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

emit_local_baseline_cmd() {
  local split="$1"
  local models="$2"
  local out_dir="$3"
  local extra=""
  if [[ "${BASELINE_MIN_KNOWN_POS}" != "-1" ]]; then
    extra="${extra} --min-known-pos-override ${BASELINE_MIN_KNOWN_POS}"
  fi
  if [[ "${BASELINE_MIN_KNOWN_NEG}" != "-1" ]]; then
    extra="${extra} --min-known-neg-override ${BASELINE_MIN_KNOWN_NEG}"
  fi
  echo "${PY} scripts/run_upir_official_baselines.py --dataset-path ${DATASET_JSON} --splits ${split} --folds ${BASELINE_FOLDS} --models ${models} --out-dir ${out_dir} --topk ${BASELINE_TOPK} --ef-percents ${BASELINE_EF_PERCENTS}${extra}"
}

emit_local_hard_slice_cmd() {
  echo "${PY} scripts/analyze_upir_open_hard_slices.py --dataset-path ${DATASET_JSON} --splits target_rec_cluster,scaffold --folds 5 --out-json results/upir_open_bigbind_hard_slice_analysis_${RUN_TAG}.json --out-md results/upir_open_bigbind_hard_slice_analysis_${RUN_TAG}.md"
}

emit_local_learned_cmd() {
  local fold="$1"
  local model="$2"
  local model_lc
  model_lc=$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')
  local run_id="upir_open_bigbind_target_rec_cluster_seed${fold}_${model_lc}_${RUN_TAG}"
  local extra_args="--dataset-path ${DATASET_JSON} --score-batch-size ${LEARNED_SCORE_BATCH_SIZE} --cache-dir ${LEARNED_CACHE_DIR}"
  if [[ "${LEARNED_STREAM_EVAL}" == "1" ]]; then
    extra_args="${extra_args} --stream-eval"
  fi
  if [[ "${LEARNED_SAVE_RANKINGS}" == "1" ]]; then
    extra_args="${extra_args} --save-rankings"
  fi
  if [[ "${LEARNED_FORCE_REENCODE}" == "1" ]]; then
    extra_args="${extra_args} --force-reencode"
  fi
  if [[ "${LEARNED_MIN_KNOWN_POS}" != "-1" ]]; then
    extra_args="${extra_args} --min-known-pos-override ${LEARNED_MIN_KNOWN_POS}"
  fi
  if [[ "${LEARNED_MIN_KNOWN_NEG}" != "-1" ]]; then
    extra_args="${extra_args} --min-known-neg-override ${LEARNED_MIN_KNOWN_NEG}"
  fi
  echo "${PY} scripts/run_bridge_experiment.py --run-id ${run_id} --dataset upir --split target_rec_cluster --models ${model} --seed ${fold} ${extra_args}"
}

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Dataset JSON not found: ${DATASET_JSON}"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1 || ! sbatch --version >/dev/null 2>&1; then
  echo "sbatch unavailable. Printing equivalent local commands:"
  if [[ "${WHAT}" == "all" || "${WHAT}" == "baselines" ]]; then
    emit_local_baseline_cmd "target_rec_cluster" "RANDOM,POPULARITY,PROT_KNN_POP,PROT_CHEM_CENTROID" "results/upir_open_bigbind_official_baselines_target_rec_cluster_${RUN_TAG}"
    if [[ "${INCLUDE_SCAFFOLD_BASELINES}" == "1" ]]; then
      emit_local_baseline_cmd "scaffold" "RANDOM,POPULARITY,PROT_CHEM_CENTROID" "results/upir_open_bigbind_official_baselines_scaffold_${RUN_TAG}"
    fi
  fi
  if [[ "${INCLUDE_HARD_SLICE_ANALYSIS}" == "1" && ( "${WHAT}" == "all" || "${WHAT}" == "hard_slices" ) ]]; then
    emit_local_hard_slice_cmd
  fi
  if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
    IFS=',' read -r -a fold_arr <<< "${LEARNED_FOLDS}"
    IFS=',' read -r -a model_arr <<< "${LEARNED_MODELS}"
    for fold in "${fold_arr[@]}"; do
      for model in "${model_arr[@]}"; do
        emit_local_learned_cmd "${fold}" "${model}"
      done
    done
  fi
  exit 0
fi

case "${WHAT}" in
  all|baselines|learned|hard_slices)
    ;;
  *)
    echo "Unsupported WHAT=${WHAT}. Use one of: all, baselines, learned, hard_slices"
    exit 2
    ;;
esac

if [[ "${WHAT}" == "all" || "${WHAT}" == "baselines" ]]; then
  target_out="results/upir_open_bigbind_official_baselines_target_rec_cluster_${RUN_TAG}"
  target_cmd=$(emit_local_baseline_cmd "target_rec_cluster" "RANDOM,POPULARITY,PROT_KNN_POP,PROT_CHEM_CENTROID" "${target_out}")
  echo "Submitting upir_bigbind_target_rec_cluster_baselines_${RUN_TAG}"
  submit_wrap \
    "upir_bigbind_tgt_base_${RUN_TAG}" \
    "${CPU_PARTITION}" \
    "${BASELINE_TIME}" \
    "${BASELINE_CPUS}" \
    "${BASELINE_MEM}" \
    "" \
    "${target_cmd}"

  if [[ "${INCLUDE_SCAFFOLD_BASELINES}" == "1" ]]; then
    scaffold_out="results/upir_open_bigbind_official_baselines_scaffold_${RUN_TAG}"
    scaffold_cmd=$(emit_local_baseline_cmd "scaffold" "RANDOM,POPULARITY,PROT_CHEM_CENTROID" "${scaffold_out}")
    echo "Submitting upir_bigbind_scaffold_baselines_${RUN_TAG}"
    submit_wrap \
      "upir_bigbind_scaf_base_${RUN_TAG}" \
      "${CPU_PARTITION}" \
      "${BASELINE_TIME}" \
      "${BASELINE_CPUS}" \
      "${BASELINE_MEM}" \
      "" \
      "${scaffold_cmd}"
  fi
fi

if [[ "${INCLUDE_HARD_SLICE_ANALYSIS}" == "1" && ( "${WHAT}" == "all" || "${WHAT}" == "hard_slices" ) ]]; then
  hard_slice_cmd=$(emit_local_hard_slice_cmd)
  echo "Submitting upir_bigbind_hard_slices_${RUN_TAG}"
  submit_wrap \
    "upir_bigbind_hslice_${RUN_TAG}" \
    "${CPU_PARTITION}" \
    "${HARD_SLICE_TIME}" \
    "${HARD_SLICE_CPUS}" \
    "${HARD_SLICE_MEM}" \
    "" \
    "${hard_slice_cmd}"
fi

if [[ "${WHAT}" == "all" || "${WHAT}" == "learned" ]]; then
  IFS=',' read -r -a fold_arr <<< "${LEARNED_FOLDS}"
  IFS=',' read -r -a model_arr <<< "${LEARNED_MODELS}"

  for fold in "${fold_arr[@]}"; do
    for model in "${model_arr[@]}"; do
      model_lc=$(printf '%s' "${model}" | tr '[:upper:]' '[:lower:]')
      run_id="upir_open_bigbind_target_rec_cluster_seed${fold}_${model_lc}_${RUN_TAG}"
      extra_args="--dataset-path ${DATASET_JSON} --score-batch-size ${LEARNED_SCORE_BATCH_SIZE} --cache-dir ${LEARNED_CACHE_DIR}"
      if [[ "${LEARNED_STREAM_EVAL}" == "1" ]]; then
        extra_args="${extra_args} --stream-eval"
      fi
      if [[ "${LEARNED_SAVE_RANKINGS}" == "1" ]]; then
        extra_args="${extra_args} --save-rankings"
      fi
      if [[ "${LEARNED_FORCE_REENCODE}" == "1" ]]; then
        extra_args="${extra_args} --force-reencode"
      fi
      if [[ "${LEARNED_MIN_KNOWN_POS}" != "-1" ]]; then
        extra_args="${extra_args} --min-known-pos-override ${LEARNED_MIN_KNOWN_POS}"
      fi
      if [[ "${LEARNED_MIN_KNOWN_NEG}" != "-1" ]]; then
        extra_args="${extra_args} --min-known-neg-override ${LEARNED_MIN_KNOWN_NEG}"
      fi

      echo "Submitting ${run_id}"
      if [[ "${PRINT_ONLY}" == "1" ]]; then
        echo "[PRINT_ONLY] sbatch --partition=${GPU_PARTITION} --time=${LEARNED_TIME} --cpus-per-task=${LEARNED_CPUS} --mem=${LEARNED_MEM} --gres=${LEARNED_GRES} --export=ALL,RUN_ID=${run_id},MODELS=${model},SPLIT=target_rec_cluster,SEED=${fold},DATASET=upir,EXTRA_ARGS=${extra_args} scripts/slurm/run_bridge_experiment.sbatch"
      else
        submit_sbatch \
          --partition="${GPU_PARTITION}" \
          --time="${LEARNED_TIME}" \
          --cpus-per-task="${LEARNED_CPUS}" \
          --mem="${LEARNED_MEM}" \
          --gres="${LEARNED_GRES}" \
          --export=ALL,RUN_ID="${run_id}",MODELS="${model}",SPLIT="target_rec_cluster",SEED="${fold}",DATASET="upir",EXTRA_ARGS="${extra_args}" \
          scripts/slurm/run_bridge_experiment.sbatch
      fi
    done
  done
fi
