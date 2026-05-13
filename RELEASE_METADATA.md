# UPIR Release Metadata

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20155018.svg)](https://doi.org/10.5281/zenodo.20155018)
**Snapshot date**: 2026-05-13

## Repository

- Development repository path: `/hpc2hdd/home/jzhao908/multi_proteo`
- Public repository URL: `https://github.com/Axfff/upir`
- Archival DOI: to be minted from the GitHub release through Zenodo

## Software License

- Code license: MIT

## Data Provenance and Redistribution

| Component | Source | Derived artifact | Redistribution status | Notes |
|---|---|---|---|---|
| `UPIR-Strict forward` | `LIT-PCBA` | `data/real_benchmarks/upir/UPIR_strict_forward.json` | reconstruction-first; full derived JSON not shipped by default | strict active/inactive semantic anchor |
| `UPIR-Open-BigBind forward` | `BigBind` | `data/real_benchmarks/upir/UPIR_open_bigbind_forward.json` | reconstruction-first; full derived JSON not shipped by default | open binary activity-label track |
| `UPIR-Open-BigBind reverse` | transposed `UPIR-Open-BigBind forward` | `data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json` | reconstruction-first; follows BigBind-derived policy | target-proposal candidate-generation track |
| Protocol files | UPIR construction scripts | `*_protocol.json` | releasable as generated metadata if upstream terms allow; otherwise reconstructed locally | fixed folds and reporting rules |
| Result summaries | UPIR baseline scripts | `results/**/summary.json`, `results/**/SUMMARY.md` | releasable when they do not expose restricted upstream content | reproducibility artifacts |

## Submission Gate

Before journal submission, this file must be updated with:

- archival DOI;
- confirmed final release tag;
- confirmed redistribution terms or reconstruction-only statement for LIT-PCBA-derived and BigBind-derived artifacts.
