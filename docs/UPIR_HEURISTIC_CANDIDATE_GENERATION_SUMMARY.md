# UPIR Candidate-Generation Baselines

- dataset: `UPIR_open_bigbind_forward`
- split: `target_rec_cluster`
- evaluation mode: `candidate_generation_retrieval`
- query rule used: `>= 1` known positive(s), `>= 0` known negative(s)`
- models: `RANDOM, POPULARITY`

## Aggregate Results

| Split | Model | AvgEvalQ/Fold | AvgCoverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_rec_cluster | RANDOM | 211 | 0.961 | 0.0177 | 0.0498 | 0.0000 | 0.0001 |
| target_rec_cluster | POPULARITY | 211 | 0.961 | 0.1773 | 0.1874 | 0.0125 | 0.0231 |
