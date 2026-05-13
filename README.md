# UPIR

UPIR is a benchmark family for unified-pool protein-ligand retrieval. It supports:

- strict forward protein-to-ligand retrieval as a semantic calibration anchor;
- open BigBind-derived forward candidate generation for first-stage ligand screening;
- open BigBind-derived reverse candidate generation for ligand-to-protein target proposal;
- separate reporting for candidate-generation retrieval and negative-aware discriminative retrieval.

This repository is released as a **reconstruction-first benchmark package**. It contains source code, protocols, result summaries, paper source, and documentation. It intentionally does **not** redistribute large upstream archives or full derived benchmark JSON artifacts by default. See [DATA_LICENSES.md](DATA_LICENSES.md) and [RECONSTRUCT.md](RECONSTRUCT.md).

## Repository Layout

| Path | Purpose |
|---|---|
| `src/experiment_bridge/` | Shared dataset, model, and metric utilities |
| `scripts/` | Benchmark construction, evaluation, aggregation, and Slurm helpers |
| `paper/` | LaTeX source for the benchmark paper |
| `docs/` | Public benchmark card and compact result summaries |
| `refine-logs/` | Experiment plans and result summaries |
| `review-stage/` | Claim and review notes used to track paper readiness |
| `figures/` | Figure-generation helpers and lightweight figure assets |

Local `data/`, `results/`, `logs/`, virtual environments, and TeX build intermediates are ignored by Git.

## Installation

Create an environment with Python 3.10+ and install the core dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`rdkit` is required for ligand/scaffold construction scripts. If your `pip` environment cannot install it, use Conda/Mamba:

```bash
mamba install -c conda-forge rdkit pyarrow numpy
```

## Reconstructing The Benchmark

UPIR expects users to obtain upstream datasets from their original sources, then run the provided scripts to materialize local benchmark artifacts. The intended release protocol is:

1. Download upstream LIT-PCBA and BigBind/BayesBind resources according to their original terms.
2. Place the archives/tables under local `data/` paths.
3. Run the construction scripts documented in [RECONSTRUCT.md](RECONSTRUCT.md).
4. Run official baselines or learned baselines using `scripts/run_upir_official_baselines.py`, `scripts/run_upir_candidate_generation_table.py`, and `scripts/run_bridge_experiment.py`.

The paper tables are tied to versioned summaries and scripts; see [RELEASE_METADATA.md](RELEASE_METADATA.md).

## Key Reproducibility Commands

Open forward candidate-generation heuristics:

```bash
venv/bin/python scripts/run_upir_candidate_generation_table.py \
  --dataset-path data/real_benchmarks/upir/UPIR_open_bigbind_forward.json \
  --split target_rec_cluster \
  --folds 5 \
  --models RANDOM,POPULARITY \
  --out-dir results/upir_open_bigbind_candidate_generation_target_rec_cluster_tt3
```

Open forward learned `B1` candidate-generation anchor:

```bash
scripts/slurm/submit_upir_open_forward_b1_candidate_generation.sh
```

Aggregate finished learned folds:

```bash
venv/bin/python scripts/aggregate_upir_learned_candidate_generation.py \
  --run-tag b1cg2 \
  --folds 0,1,2,3,4 \
  --model B1
```

## Citation

If you use this benchmark, cite the archived software release and the upstream resources listed in [DATA_LICENSES.md](DATA_LICENSES.md). A `CITATION.cff` file is included for GitHub and Zenodo metadata.
