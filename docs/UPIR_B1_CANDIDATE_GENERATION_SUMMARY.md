# UPIR Open Forward Learned Candidate-Generation Anchor

- dataset: `UPIR_open_bigbind_forward`
- split: `target_rec_cluster`
- evaluation mode: `candidate_generation_retrieval`
- metric semantics: full shared ligand pool
- query rule used: `>= 1` known positive(s), `>= 0` known negative(s)
- model: `B1`
- run tag: `b1cg2`

## Aggregate Results

| Model | AvgEvalQ/Fold | AvgCoverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 | MRR | Median Pos Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 211.2 | 0.961 | 0.0072 | 0.0333 | 0.0000 | 0.0002 | 0.0040 | 186266.8 |

## Fold Results

| Fold | EvalQ | Coverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 | MRR | Runtime(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 221 | 0.825 | 0.0045 | 0.0317 | 0.0000 | 0.0001 | 0.0030 | 71.5 |
| 1 | 245 | 0.996 | 0.0163 | 0.0653 | 0.0001 | 0.0005 | 0.0093 | 67.4 |
| 2 | 192 | 1.000 | 0.0052 | 0.0156 | 0.0000 | 0.0001 | 0.0023 | 52.8 |
| 3 | 214 | 0.991 | 0.0047 | 0.0374 | 0.0000 | 0.0001 | 0.0034 | 60.1 |
| 4 | 184 | 0.995 | 0.0054 | 0.0163 | 0.0000 | 0.0001 | 0.0019 | 50.4 |

## Interpretation Boundary

These metrics measure recovery of known positives in the full shared ligand pool. Unlabeled candidates are not counted as negatives, so this table supports first-stage candidate-generation claims rather than final specificity claims.
