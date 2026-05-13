#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import build_split_view, get_primary_query_rule, load_dataset
from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official reverse baselines on UPIR strict reverse.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_strict_reverse.json")
    p.add_argument("--split", default="standard", choices=["standard"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="RANDOM,POPULARITY")
    p.add_argument("--out-dir", default="results/upir_strict_reverse_official_baselines_v1")
    p.add_argument("--topk", default="1,5,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-known-pos-override", type=int, default=-1)
    p.add_argument("--min-known-neg-override", type=int, default=-1)
    p.add_argument("--max-eval-queries", type=int, default=0)
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _reverse_metrics(
    ranking_ids: Sequence[str],
    label_row: Dict[str, int],
    topk_values: Sequence[int],
    min_known_pos: int,
    min_known_neg: int,
) -> Dict[str, float]:
    ranked_ids = [cid for cid in ranking_ids if cid in label_row]
    positives = [cid for cid in ranked_ids if int(label_row.get(cid, 0)) == 1]
    num_pos = len(positives)
    num_neg = len(ranked_ids) - num_pos
    if num_pos < int(min_known_pos) or num_neg < int(min_known_neg):
        return {}

    pos_ranks = [idx + 1 for idx, cid in enumerate(ranked_ids) if int(label_row.get(cid, 0)) == 1]
    first_pos_rank = min(pos_ranks)
    out: Dict[str, float] = {
        "mrr": 1.0 / first_pos_rank,
        "num_pos": float(num_pos),
        "num_known_pairs": float(len(ranked_ids)),
    }

    for k in topk_values:
        top_ids = ranked_ids[:k]
        tp = sum(1 for cid in top_ids if int(label_row.get(cid, 0)) == 1)
        out[f"recall@{k}"] = tp / num_pos
        out[f"hit@{k}"] = 1.0 if tp > 0 else 0.0
    return out


def _aggregate_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out = {"num_eval_queries": float(len(rows))}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def _average_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def _stable_query_subset(query_ids: Sequence[str], max_queries: int, seed: int) -> List[str]:
    if max_queries <= 0 or len(query_ids) <= max_queries:
        return list(query_ids)
    ordered = sorted(query_ids, key=lambda qid: stable_seed(f"revq:{seed}:{qid}"))
    return ordered[:max_queries]


def _random_order(candidate_ids: Sequence[str], seed: int, query_id: str) -> List[str]:
    return sorted(candidate_ids, key=lambda cid: stable_seed(f"reverse_random:{seed}:{query_id}:{cid}"))


def _popularity_order(train_pairs: Sequence[tuple[str, str, int]], candidate_ids: Sequence[str]) -> List[str]:
    pos_counts = defaultdict(int)
    known_counts = defaultdict(int)
    for _, cid, label in train_pairs:
        known_counts[cid] += 1
        if int(label) == 1:
            pos_counts[cid] += 1
    return sorted(
        candidate_ids,
        key=lambda cid: (
            -(pos_counts[cid] / max(1, known_counts[cid])),
            -pos_counts[cid],
            stable_seed(cid),
        ),
    )


def _write_summary_md(
    out_path: Path,
    dataset_name: str,
    split_name: str,
    models: Sequence[str],
    query_rule: Dict[str, int],
    aggregates: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# UPIR Reverse Official Baselines",
        "",
        f"- dataset: `{dataset_name}`",
        f"- split: `{split_name}`",
        f"- models: `{', '.join(models)}`",
        f"- query rule used: `>= {query_rule['min_known_pos']}` known positive(s), `>= {query_rule['min_known_neg']}` known negative(s)`",
        "",
        "## Aggregate Results",
        "",
        "| Model | AvgEvalQ/Fold | AvgCoverage | MRR | Hit@1 | Hit@5 | Recall@1 | Recall@5 | Recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in models:
        m = aggregates[model_name]
        lines.append(
            f"| {model_name} | {m.get('num_total_eval_queries', 0.0):.0f} | {m.get('evaluable_query_coverage', 0.0):.3f} | "
            f"{m.get('mrr', 0.0):.4f} | {m.get('hit@1', 0.0):.4f} | {m.get('hit@5', 0.0):.4f} | "
            f"{m.get('recall@1', 0.0):.4f} | {m.get('recall@5', 0.0):.4f} | {m.get('recall@10', 0.0):.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    bundle = load_dataset(Path(args.dataset_path))
    if bundle.meta and bundle.meta.get("query_entity") != "ligand":
        raise ValueError("This script expects a reverse dataset with ligand queries.")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    topk_values = _parse_ints(args.topk)

    rule = get_primary_query_rule(bundle)
    min_known_pos = args.min_known_pos_override if args.min_known_pos_override >= 0 else int(rule["min_known_pos"])
    default_neg = max(1, int(rule["min_known_neg"]))
    min_known_neg = args.min_known_neg_override if args.min_known_neg_override >= 0 else default_neg

    results = {
        "dataset_name": bundle.name,
        "dataset_path": str(Path(args.dataset_path)),
        "split": args.split,
        "folds": args.folds,
        "models": model_names,
        "query_rule_used": {"min_known_pos": min_known_pos, "min_known_neg": min_known_neg},
        "results": {},
    }

    for model_name in model_names:
        fold_rows: List[Dict[str, float]] = []
        for fold_seed in range(args.folds):
            split = build_split_view(bundle, split_name=args.split, seed=fold_seed)
            eval_query_ids = _stable_query_subset(split.eval_pocket_ids, args.max_eval_queries, seed=fold_seed)
            candidate_ids = list(split.candidate_ligand_ids)

            if model_name == "POPULARITY":
                base_order = _popularity_order(split.train_pairs, candidate_ids)
            else:
                base_order = []

            query_rows: List[Dict[str, float]] = []
            for qid in eval_query_ids:
                if model_name == "RANDOM":
                    ranking = _random_order(candidate_ids, seed=fold_seed, query_id=qid)
                elif model_name == "POPULARITY":
                    ranking = base_order
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                row = _reverse_metrics(
                    ranking_ids=ranking,
                    label_row=bundle.labels[qid],
                    topk_values=topk_values,
                    min_known_pos=min_known_pos,
                    min_known_neg=min_known_neg,
                )
                if row:
                    query_rows.append(row)

            agg = _aggregate_rows(query_rows)
            agg["num_total_eval_queries"] = float(len(eval_query_ids))
            agg["num_evaluable_queries"] = float(len(query_rows))
            agg["evaluable_query_coverage"] = len(query_rows) / max(1, len(eval_query_ids))
            agg["fold"] = float(fold_seed)
            fold_rows.append(agg)

        results["results"][model_name] = {
            "aggregate_over_folds": _average_dicts(fold_rows),
            "folds": fold_rows,
        }

    save_json(out_dir / "summary.json", results)
    _write_summary_md(
        out_dir / "SUMMARY.md",
        dataset_name=bundle.name,
        split_name=args.split,
        models=model_names,
        query_rule={"min_known_pos": min_known_pos, "min_known_neg": min_known_neg},
        aggregates={m: results["results"][m]["aggregate_over_folds"] for m in model_names},
    )

    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
