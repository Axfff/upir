#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper-facing UPIR benchmark-family comparison table.")
    p.add_argument("--out-json", default="results/upir_benchmark_family_table.json")
    p.add_argument("--out-md", default="results/upir_benchmark_family_table.md")
    return p.parse_args()


def _load(path: str) -> Dict:
    return load_json(Path(path))


def main() -> None:
    args = parse_args()
    strict = _load("data/real_benchmarks/upir/UPIR_strict_forward_stats.json")
    open_fwd = _load("data/real_benchmarks/upir/UPIR_open_bigbind_forward_stats.json")
    open_rev = _load("data/real_benchmarks/upir/UPIR_open_bigbind_reverse_stats.json")
    rev_cg = _load("results/upir_open_bigbind_reverse_candidate_generation_standard_v1/summary.json")
    rev_disc = _load("results/upir_open_bigbind_reverse_discriminative_standard_v1/summary.json")
    fwd_cg = _load("results/upir_open_bigbind_candidate_generation_target_rec_cluster_tt3_fast/summary.json")

    rows: List[Dict] = [
        {
            "track": "UPIR-Strict forward",
            "direction": "protein -> ligand",
            "label_regime": "strict active/inactive",
            "queries": strict["counts"]["num_queries"],
            "candidates": strict["counts"]["num_candidates"],
            "known_pairs": strict["counts"]["num_known_pairs"],
            "primary_mode": "discriminative anchor",
            "candidate_generation_coverage": "not primary",
            "discriminative_coverage": "full strict labels",
            "main_role": "semantic calibration",
        },
        {
            "track": "UPIR-open BigBind forward",
            "direction": "protein -> ligand",
            "label_regime": "open BigBind binary activity",
            "queries": open_fwd["counts"]["num_queries"],
            "candidates": open_fwd["counts"]["num_candidates"],
            "known_pairs": open_fwd["counts"]["num_known_pairs"],
            "primary_mode": "candidate generation",
            "candidate_generation_coverage": f"{fwd_cg['summary_by_model']['POPULARITY']['evaluable_query_coverage']:.3f} on target_rec_cluster",
            "discriminative_coverage": "0.828 on target_rec_cluster",
            "main_role": "large-scale forward screening",
        },
        {
            "track": "UPIR-open BigBind reverse",
            "direction": "ligand -> protein",
            "label_regime": "open BigBind binary activity transposed",
            "queries": open_rev["counts"]["num_queries"],
            "candidates": open_rev["counts"]["num_candidates"],
            "known_pairs": open_rev["counts"]["num_known_pairs"],
            "primary_mode": "candidate generation",
            "candidate_generation_coverage": f"{rev_cg['summary_by_model']['POPULARITY']['evaluable_query_coverage']:.3f} held-out per fold; 0.870 full graph",
            "discriminative_coverage": f"{rev_disc['summary_by_model']['POPULARITY']['evaluable_query_coverage']:.4f} held-out per fold",
            "main_role": "target proposal / target fishing",
        },
    ]
    payload = {
        "name": "upir_benchmark_family_table",
        "rows": rows,
        "key_takeaways": [
            "UPIR is now a benchmark family rather than one forward-only track.",
            "Candidate generation is the primary mode for open forward and open reverse first-stage screening.",
            "Strict and discriminative modes remain necessary for specificity calibration but should not define every track.",
            "Open reverse is feasible for target proposal, while reverse discriminative evaluation is too sparse and saturated for headline use.",
        ],
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    save_json(out_json, payload)
    ensure_dir(out_md.parent)
    lines = [
        "# UPIR Benchmark-Family Comparison",
        "",
        "| Track | Direction | Label Regime | Queries | Candidates | Known Pairs | Primary Mode | Candidate-Generation Coverage | Discriminative Coverage | Main Role |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['track']}` | {row['direction']} | {row['label_regime']} | {row['queries']} | {row['candidates']} | {row['known_pairs']} | {row['primary_mode']} | {row['candidate_generation_coverage']} | {row['discriminative_coverage']} | {row['main_role']} |"
        )
    lines.extend(["", "## Key Takeaways", ""])
    for item in payload["key_takeaways"]:
        lines.append(f"- {item}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
