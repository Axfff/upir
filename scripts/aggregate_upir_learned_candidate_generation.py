#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate learned UPIR candidate-generation folds.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--run-prefix", default="upir_open_bigbind_target_rec_cluster")
    p.add_argument("--run-tag", default="b1cg1")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--model", default="B1")
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def _parse_folds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: Dict, key: str) -> float:
    return float(metrics.get(key, float("nan")))


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    folds = _parse_folds(args.folds)
    model = args.model
    model_lc = model.lower()
    out_dir = Path(args.out_dir) if args.out_dir else results_root / f"upir_open_bigbind_forward_{model_lc}_candidate_generation_{args.run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    missing = []
    for fold in folds:
        run_id = f"{args.run_prefix}_seed{fold}_{model_lc}_{args.run_tag}"
        summary_path = results_root / run_id / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        summary = _load_json(summary_path)
        if model not in summary.get("results", {}):
            raise SystemExit(f"Model {model} missing from {summary_path}")
        if summary.get("evaluation_mode") != "candidate_generation_retrieval":
            raise SystemExit(f"{summary_path} is not candidate-generation mode")
        if summary.get("metric_semantics") != "full_candidate_pool":
            raise SystemExit(f"{summary_path} does not use full-candidate-pool semantics")
        metrics = summary["results"][model]["metrics"]
        efficiency = summary["results"][model].get("efficiency", {})
        fold_rows.append(
            {
                "fold": fold,
                "run_id": run_id,
                "summary_path": str(summary_path),
                "num_total_eval_queries": _metric(metrics, "num_total_eval_pockets"),
                "num_evaluable_queries": _metric(metrics, "num_evaluable_pockets"),
                "coverage": _metric(metrics, "evaluable_query_coverage"),
                "mrr": _metric(metrics, "mrr"),
                "hit@10": _metric(metrics, "hit@10"),
                "hit@50": _metric(metrics, "hit@50"),
                "recall@10": _metric(metrics, "recall@10"),
                "recall@50": _metric(metrics, "recall@50"),
                "median_rank_true_binders": _metric(metrics, "median_rank_true_binders"),
                "scoring_qps": float(efficiency.get("scoring_qps", float("nan"))),
                "index_memory_mb": float(efficiency.get("index_memory_mb", float("nan"))),
                "runtime_sec": float(summary["results"][model].get("runtime_sec", float("nan"))),
            }
        )

    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Missing fold summaries:\n{joined}")
    if not fold_rows:
        raise SystemExit("No fold summaries found")

    metric_keys = [
        "num_total_eval_queries",
        "num_evaluable_queries",
        "coverage",
        "mrr",
        "hit@10",
        "hit@50",
        "recall@10",
        "recall@50",
        "median_rank_true_binders",
        "scoring_qps",
        "index_memory_mb",
        "runtime_sec",
    ]
    aggregate = {}
    for key in metric_keys:
        values = [row[key] for row in fold_rows if not math.isnan(row[key])]
        aggregate[key] = {"mean": _mean(values), "std": _std(values)}

    payload = {
        "dataset": "UPIR_open_bigbind_forward",
        "split": "target_rec_cluster",
        "evaluation_mode": "candidate_generation_retrieval",
        "metric_semantics": "full_candidate_pool",
        "query_rule_used": {"min_known_pos": 1, "min_known_neg": 0},
        "model": model,
        "run_tag": args.run_tag,
        "folds": fold_rows,
        "aggregate": aggregate,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    agg = aggregate
    lines = [
        "# UPIR Open Forward Learned Candidate-Generation Anchor",
        "",
        "- dataset: `UPIR_open_bigbind_forward`",
        "- split: `target_rec_cluster`",
        "- evaluation mode: `candidate_generation_retrieval`",
        "- metric semantics: full shared ligand pool",
        "- query rule used: `>= 1` known positive(s), `>= 0` known negative(s)",
        f"- model: `{model}`",
        f"- run tag: `{args.run_tag}`",
        "",
        "## Aggregate Results",
        "",
        "| Model | AvgEvalQ/Fold | AvgCoverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 | MRR | Median Pos Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {model} | {agg['num_evaluable_queries']['mean']:.1f} | {agg['coverage']['mean']:.3f} | "
            f"{agg['hit@10']['mean']:.4f} | {agg['hit@50']['mean']:.4f} | "
            f"{agg['recall@10']['mean']:.4f} | {agg['recall@50']['mean']:.4f} | "
            f"{agg['mrr']['mean']:.4f} | {agg['median_rank_true_binders']['mean']:.1f} |"
        ),
        "",
        "## Fold Results",
        "",
        "| Fold | EvalQ | Coverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 | MRR | Runtime(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['num_evaluable_queries']:.0f} | {row['coverage']:.3f} | "
            f"{row['hit@10']:.4f} | {row['hit@50']:.4f} | {row['recall@10']:.4f} | "
            f"{row['recall@50']:.4f} | {row['mrr']:.4f} | {row['runtime_sec']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These metrics measure recovery of known positives in the full shared ligand pool. Unlabeled candidates are not counted as negatives, so this table supports first-stage candidate-generation claims rather than final specificity claims.",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary_json": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
