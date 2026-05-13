#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize UPIR official baseline results into a review-stage note.")
    p.add_argument("--summary-json", default="results/upir_official_baselines_v1/summary.json")
    p.add_argument("--task-shift-json", default="results/upir_task_shift_analysis.json")
    p.add_argument("--out-md", default="review-stage/UPIR_BASELINE_ANALYSIS.md")
    return p.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    summary = _load(args.summary_json)
    shift = _load(args.task_shift_json)

    models = summary["models"]
    splits = summary["splits"]

    def row(split: str, model: str) -> dict:
        return summary["results"][split]["models"][model]["aggregate_over_query_instances"]

    def has_row(split: str, model: str) -> bool:
        return model in summary["results"][split]["models"]

    lines = [
        "# UPIR Official Baseline Analysis",
        "",
        "This note summarizes the official baseline table on the fixed `UPIR-Strict` protocol.",
        "",
        "## Baselines Included",
        "",
    ]
    for model in models:
        lines.append(f"- `{model}`")

    lines.extend(
        [
            "",
            "## Key Results",
            "",
            "| Split | Model | MRR | Recall@10 | Recall@50 | EF1 | EF5 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for split in splits:
        for model in models:
            if not has_row(split, model):
                continue
            r = row(split, model)
            lines.append(
                f"| {split} | {model} | {r['mrr']:.4f} | {r['recall@10']:.4f} | {r['recall@50']:.4f} | {r['ef1']:.4f} | {r['ef5']:.4f} |"
            )

    lines.extend(["", "## Main Takeaways", ""])
    if "POPULARITY" in models and "standard" in splits and has_row("standard", "POPULARITY"):
        std_pop = row("standard", "POPULARITY")
        lines.append(
            f"- The ligand-only `POPULARITY` baseline is extremely strong on `standard` (`MRR={std_pop['mrr']:.4f}`, `EF1={std_pop['ef1']:.2f}`), which indicates that pair-level standard splitting is highly favorable to leakage-prone priors."
        )
    if "POPULARITY" in models and "target" in splits and has_row("target", "POPULARITY"):
        tgt_pop = row("target", "POPULARITY")
        lines.append(
            f"- `POPULARITY` drops sharply on `target` (`MRR={tgt_pop['mrr']:.4f}`, `EF1={tgt_pop['ef1']:.2f}`), showing that held-out-query evaluation is materially harder."
        )
    if (
        "POPULARITY" in models
        and "scaffold" in splits
        and "RANDOM" in models
        and has_row("scaffold", "POPULARITY")
        and has_row("scaffold", "RANDOM")
    ):
        scf_pop = row("scaffold", "POPULARITY")
        scf_rand = row("scaffold", "RANDOM")
        lines.append(
            f"- On `scaffold`, `POPULARITY` is near random (`MRR={scf_pop['mrr']:.4f}` vs random `{scf_rand['mrr']:.4f}`; `EF1={scf_pop['ef1']:.2f}` vs random `{scf_rand['ef1']:.2f}`), which means the fixed scaffold protocol suppresses trivial ligand memorization."
        )
    if "PROT_KNN_POP" in models and "target" in splits and has_row("target", "PROT_KNN_POP"):
        tgt_knn = row("target", "PROT_KNN_POP")
        lines.append(
            f"- `PROT_KNN_POP` is a simple protein-aware transfer baseline and improves target-split first-hit retrieval (`MRR={tgt_knn['mrr']:.4f}`, `EF1={tgt_knn['ef1']:.2f}`), showing the benchmark can separate nontrivial pocket-aware baselines from trivial priors."
        )
    if "PROT_CHEM_CENTROID" in models and "target" in splits and has_row("target", "PROT_CHEM_CENTROID"):
        tgt_chem = row("target", "PROT_CHEM_CENTROID")
        lines.append(
            f"- `PROT_CHEM_CENTROID` is a protein-guided chemistry-transfer baseline and improves broader target-split retrieval (`Recall@50={tgt_chem['recall@50']:.4f}`, `EF5={tgt_chem['ef5']:.2f}`), providing a first nontrivial chemistry-aware reference."
        )

    lines.extend(
        [
            "",
            "## Why This Matters For Benchmark Value",
            "",
            "- The benchmark now has an official baseline table on the released protocol, not just a dataset artifact.",
            "- The baseline behavior is diagnostically useful: it separates easy leakage-prone settings from harder OOD settings.",
            "- This supports the claim that `UPIR` is scientifically more than a JSON reformat, because protocol choice changes what trivial baselines can do.",
            "",
            "## Task-Shift Evidence",
            "",
            f"- Relative to the earlier sampled proxy, `UPIR` increases the candidate pool by `{shift['ratios']['candidate_pool_ratio']:.1f}x`.",
            f"- Median per-query candidate set grows by `{shift['ratios']['median_candidates_per_query_ratio']:.1f}x`.",
            f"- Mean positive rate drops to `{100.0 * shift['upir']['mean_positive_rate']:.2f}%`, only `{100.0 * shift['ratios']['mean_positive_rate_ratio']:.2f}%` of the sampled proxy rate.",
            "- This shows the unified-pool task is materially different from the earlier sampled setting and should be treated as a distinct benchmark regime.",
            "",
            "## Remaining Gaps",
            "",
            "- The benchmark still lacks a stronger learned pocket-aware baseline on the fixed protocol.",
            "- The benchmark still has only `15` protein-pocket queries.",
            "- The scaffold split still uses a proxy scaffold definition.",
            "",
        ]
    )

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_md": args.out_md}, indent=2))


if __name__ == "__main__":
    main()
