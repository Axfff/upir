# Reconstructing UPIR Artifacts

This document describes the intended reconstruction workflow for the public release. Exact source download URLs and checksums should be filled in once the final upstream archives are selected.

## 1. Prepare Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `rdkit` is unavailable through `pip`, install it with Conda/Mamba:

```bash
mamba install -c conda-forge rdkit pyarrow numpy
```

## 2. Place Upstream Data Locally

The public release does not ship upstream archives. Place locally downloaded files under:

```text
data/LIT-PCBA_full.tar
data/bigbind/BigBindV1.5.tar.gz
data/bigbind/bayesbind/BayesBindV1.5.tar.gz
data/bigbind/bayesbind/BayesBindMLV1.5.tar.gz
```

These filenames match the development snapshot. If upstream filenames differ, pass explicit paths to the relevant scripts.

## 3. Build Canonical Tables

The reconstruction scripts convert upstream resources into compact canonical protein, ligand, and edge tables:

```bash
venv/bin/python scripts/build_upir_proteins.py
venv/bin/python scripts/build_upir_ligands.py
venv/bin/python scripts/build_upir_canonical_table.py
```

For BigBind/BayesBind-style sources:

```bash
venv/bin/python scripts/build_upir_bigbind_proteins.py
venv/bin/python scripts/build_upir_bigbind_canonical_table.py
venv/bin/python scripts/build_upir_bayesbind_proteins.py
venv/bin/python scripts/build_upir_bayesbind_ligands.py
venv/bin/python scripts/build_upir_bayesbind_canonical_table.py
```

## 4. Build Benchmark JSON and Protocols

Strict forward/reverse:

```bash
venv/bin/python scripts/build_upir_protocol.py
venv/bin/python scripts/build_upir_strict_benchmark.py \
  --canonical-parquet data/real_benchmarks/upir/upir_canonical_edges.parquet \
  --proteins-parquet data/real_benchmarks/upir/upir_proteins.parquet \
  --ligands-parquet data/real_benchmarks/upir/upir_ligands.parquet \
  --out-forward-json data/real_benchmarks/upir/UPIR_strict_forward.json \
  --out-reverse-json data/real_benchmarks/upir/UPIR_strict_reverse.json \
  --forward-protocol-path data/real_benchmarks/upir/UPIR_strict_forward_protocol.json
```

Open BigBind forward:

```bash
venv/bin/python scripts/build_upir_generic_protocol.py
venv/bin/python scripts/build_upir_open_benchmark.py \
  --canonical-parquet data/real_benchmarks/upir/bigbind_canonical_edges.parquet \
  --proteins-parquet data/real_benchmarks/upir/bigbind_proteins.parquet \
  --ligands-parquet data/real_benchmarks/upir/bigbind_ligands.parquet \
  --out-json data/real_benchmarks/upir/UPIR_open_bigbind_forward.json \
  --protocol-path data/real_benchmarks/upir/UPIR_open_bigbind_forward_protocol.json \
  --benchmark-name UPIR_open_bigbind_forward \
  --source BigBindV1.5 \
  --benchmark-type upir_open_bigbind_forward
```

Open BigBind reverse:

```bash
venv/bin/python scripts/materialize_upir_open_reverse.py \
  --forward-path data/real_benchmarks/upir/UPIR_open_bigbind_forward.json \
  --out-dataset data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json \
  --out-protocol data/real_benchmarks/upir/UPIR_open_bigbind_reverse_protocol.json \
  --out-stats data/real_benchmarks/upir/UPIR_open_bigbind_reverse_stats.json
```

## 5. Run Reference Tables

Forward candidate-generation heuristics:

```bash
venv/bin/python scripts/run_upir_candidate_generation_table.py \
  --dataset-path data/real_benchmarks/upir/UPIR_open_bigbind_forward.json \
  --split target_rec_cluster \
  --folds 5 \
  --models RANDOM,POPULARITY \
  --out-dir results/upir_open_bigbind_candidate_generation_target_rec_cluster_tt3
```

Forward learned `B1` candidate-generation anchor:

```bash
scripts/slurm/submit_upir_open_forward_b1_candidate_generation.sh
```

Open reverse candidate-generation:

```bash
venv/bin/python scripts/run_upir_open_reverse_candidate_generation.py
```

Open reverse degree slices:

```bash
venv/bin/python scripts/analyze_upir_open_reverse_degree_slices.py
```

## 6. Verify

Expected key summaries:

```text
results/upir_open_bigbind_candidate_generation_target_rec_cluster_tt3/SUMMARY.md
results/upir_open_bigbind_forward_b1_candidate_generation_b1cg2/SUMMARY.md
results/upir_open_bigbind_reverse_candidate_generation_standard_v1/SUMMARY.md
results/upir_open_bigbind_reverse_degree_slices_v1/SUMMARY.md
```

