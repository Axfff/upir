#!/usr/bin/env bash
#SBATCH --job-name=bridge_vs
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=i64m1tga800ue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -z "${RUN_ID:-}" ] || [ -z "${MODELS:-}" ] || [ -z "${SPLIT:-}" ] || [ -z "${SEED:-}" ]; then
  echo "Missing required env vars: RUN_ID MODELS SPLIT SEED"
  exit 2
fi

mkdir -p logs/slurm

PY=${PY:-venv/bin/python}
DATASET=${DATASET:-litpcba}
EXTRA_ARGS=${EXTRA_ARGS:-}

cmd="$PY scripts/run_bridge_experiment.py --run-id ${RUN_ID} --dataset ${DATASET} --split ${SPLIT} --models ${MODELS} --seed ${SEED} ${EXTRA_ARGS}"
echo "[SLURM] $cmd"
eval "$cmd"
