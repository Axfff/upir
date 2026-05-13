#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import build_split_view, get_primary_query_rule, load_dataset
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze hard-slice viability for UPIR open-track evaluation.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_forward.json")
    p.add_argument("--splits", default="target_rec_cluster,scaffold")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--neg-thresholds", default="1,5,10,20,50")
    p.add_argument("--pos-rate-bins", default="0.0,0.05,0.2,0.5,0.8,0.95,1.01")
    p.add_argument("--out-json", default="results/upir_open_bigbind_hard_slice_analysis.json")
    p.add_argument("--out-md", default="results/upir_open_bigbind_hard_slice_analysis.md")
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _query_stats(candidate_ids: Sequence[str], label_row: Dict[str, int]) -> Dict[str, float]:
    known = [int(label_row[lid]) for lid in candidate_ids if lid in label_row]
    num_known = len(known)
    num_pos = sum(1 for x in known if x == 1)
    num_neg = sum(1 for x in known if x == 0)
    pos_rate = (num_pos / num_known) if num_known else 0.0
    return {
        "known_pairs": float(num_known),
        "known_pos": float(num_pos),
        "known_neg": float(num_neg),
        "positive_rate": float(pos_rate),
    }


def _bucket_counts(values: Sequence[float], bins: Sequence[float]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = sum(1 for v in values if lo <= v < hi)
        out.append({"lo": float(lo), "hi": float(hi), "count": float(count)})
    return out


def _mean(rows: Sequence[float]) -> float:
    return sum(rows) / len(rows) if rows else float("nan")


def main() -> None:
    args = parse_args()
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    ensure_dir(out_json.parent)
    ensure_dir(out_md.parent)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    neg_thresholds = _parse_ints(args.neg_thresholds)
    pos_rate_bins = _parse_floats(args.pos_rate_bins)

    bundle = load_dataset(Path(args.dataset_path))
    query_rule = get_primary_query_rule(bundle)

    payload: Dict[str, object] = {
        "dataset_path": args.dataset_path,
        "dataset_name": bundle.name,
        "query_rule": query_rule,
        "splits": {},
    }

    md_lines = [
        "# UPIR Open-Track Hard-Slice Analysis",
        "",
        f"- dataset: `{bundle.name}`",
        f"- dataset path: `{args.dataset_path}`",
        "",
    ]

    for split_name in splits:
        split_rows: List[Dict[str, object]] = []
        all_pos_rates: List[float] = []
        summary_rows: List[Dict[str, float]] = []

        for fold in range(args.folds):
            split = build_split_view(bundle, split_name=split_name, seed=fold)
            fold_query_rows: List[Dict[str, float]] = []
            for pid in split.eval_pocket_ids:
                stats = _query_stats(split.candidate_ligand_ids, bundle.labels.get(pid, {}))
                stats["pid"] = pid
                fold_query_rows.append(stats)
                all_pos_rates.append(float(stats["positive_rate"]))

            total_queries = len(fold_query_rows)
            summary: Dict[str, float] = {
                "fold": float(fold),
                "num_eval_queries": float(total_queries),
                "avg_known_pairs": _mean([r["known_pairs"] for r in fold_query_rows]),
                "avg_known_pos": _mean([r["known_pos"] for r in fold_query_rows]),
                "avg_known_neg": _mean([r["known_neg"] for r in fold_query_rows]),
                "avg_positive_rate": _mean([r["positive_rate"] for r in fold_query_rows]),
                "query_rule_coverage": (
                    sum(
                        1
                        for r in fold_query_rows
                        if r["known_pos"] >= query_rule["min_known_pos"] and r["known_neg"] >= query_rule["min_known_neg"]
                    )
                    / max(1, total_queries)
                ),
            }
            for t in neg_thresholds:
                summary[f"coverage_neg_ge_{t}"] = (
                    sum(1 for r in fold_query_rows if r["known_neg"] >= t) / max(1, total_queries)
                )
            split_rows.append(
                {
                    "fold": fold,
                    "notes": split.notes,
                    "summary": summary,
                    "positive_rate_buckets": _bucket_counts([r["positive_rate"] for r in fold_query_rows], pos_rate_bins),
                }
            )
            summary_rows.append(summary)

        mean_summary: Dict[str, float] = {}
        if summary_rows:
            numeric_keys = sorted({k for row in summary_rows for k in row.keys() if k != "fold"})
            for key in numeric_keys:
                mean_summary[key] = _mean([float(row[key]) for row in summary_rows])

        payload["splits"][split_name] = {
            "folds": split_rows,
            "mean_over_folds": mean_summary,
            "global_positive_rate_buckets": _bucket_counts(all_pos_rates, pos_rate_bins),
        }

        md_lines.extend(
            [
                f"## {split_name}",
                "",
                "| Metric | Mean Over Folds |",
                "|---|---:|",
            ]
        )
        for key, value in sorted(mean_summary.items()):
            md_lines.append(f"| {key} | {value:.4f} |")
        md_lines.extend(["", "### Global Positive-Rate Buckets", "", "| Range | Queries |", "|---|---:|"])
        for bucket in payload["splits"][split_name]["global_positive_rate_buckets"]:
            md_lines.append(f"| [{bucket['lo']:.2f}, {bucket['hi']:.2f}) | {bucket['count']:.0f} |")
        md_lines.append("")

    save_json(out_json, payload)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
