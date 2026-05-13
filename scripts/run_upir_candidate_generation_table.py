#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import build_split_view, load_dataset
from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast candidate-generation baseline table for UPIR.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_forward.json")
    p.add_argument("--split", default="target_rec_cluster")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="RANDOM,POPULARITY")
    p.add_argument("--out-dir", default="results/upir_candidate_generation_table_v1")
    p.add_argument("--topk", default="10,50")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _stable_random_order(candidate_ids: Sequence[str], seed: int) -> List[str]:
    return sorted(candidate_ids, key=lambda lid: stable_seed(f"random:{seed}:{lid}"))


def _popularity_order(train_pairs, candidate_ids: Sequence[str]) -> List[str]:
    pos_counts = defaultdict(int)
    known_counts = defaultdict(int)
    for _, lid, label in train_pairs:
        known_counts[lid] += 1
        if int(label) == 1:
            pos_counts[lid] += 1
    return sorted(
        candidate_ids,
        key=lambda lid: (
            -(pos_counts[lid] / max(1, known_counts[lid])),
            -pos_counts[lid],
            stable_seed(lid),
        ),
    )


def _query_metrics(pos_ranks: List[int], topk_values: Sequence[int]) -> Dict[str, float]:
    num_pos = len(pos_ranks)
    out: Dict[str, float] = {"num_pos": float(num_pos)}
    for k in topk_values:
        tp = sum(1 for r in pos_ranks if r <= k)
        out[f"recall@{k}"] = tp / num_pos
        out[f"hit@{k}"] = 1.0 if tp > 0 else 0.0
    return out


def _aggregate(rows: List[Dict[str, float]], topk_values: Sequence[int]) -> Dict[str, float]:
    if not rows:
        out = {"num_evaluable_queries": 0.0}
        for k in topk_values:
            out[f"recall@{k}"] = 0.0
            out[f"hit@{k}"] = 0.0
        return out
    out = {"num_evaluable_queries": float(len(rows))}
    for k in topk_values:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
        out[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / len(rows)
    out["num_pos"] = sum(r["num_pos"] for r in rows) / len(rows)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    bundle = load_dataset(Path(args.dataset_path))
    topk_values = _parse_ints(args.topk)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {"dataset_name": bundle.name, "evaluation_mode": "candidate_generation_retrieval", "query_rule_used": {"min_known_pos": 1, "min_known_neg": 0}, "split": args.split, "folds": []}
    aggregate_by_model = defaultdict(list)

    for fold in range(args.folds):
        split = build_split_view(bundle, split_name=args.split, seed=fold)
        total_eval_queries = len(split.eval_pocket_ids)
        candidate_ids = list(split.candidate_ligand_ids)
        fold_entry = {"fold": fold, "num_total_eval_queries": total_eval_queries, "models": {}}

        rankings = {}
        if "RANDOM" in model_names:
            rankings["RANDOM"] = _stable_random_order(candidate_ids, seed=args.seed + fold)
        if "POPULARITY" in model_names:
            rankings["POPULARITY"] = _popularity_order(split.train_pairs, candidate_ids)

        rank_maps = {name: {lid: idx + 1 for idx, lid in enumerate(order)} for name, order in rankings.items()}
        allowed = set(candidate_ids)

        for model_name in model_names:
            rows = []
            rank_map = rank_maps[model_name]
            for pid in split.eval_pocket_ids:
                label_row = bundle.labels.get(pid, {})
                pos_ids = [lid for lid, lab in label_row.items() if int(lab) == 1 and lid in allowed]
                if not pos_ids:
                    continue
                pos_ranks = sorted(rank_map[lid] for lid in pos_ids if lid in rank_map)
                if not pos_ranks:
                    continue
                rows.append(_query_metrics(pos_ranks, topk_values))
            agg = _aggregate(rows, topk_values)
            agg["num_total_eval_queries"] = float(total_eval_queries)
            agg["evaluable_query_coverage"] = len(rows) / max(1, total_eval_queries)
            fold_entry["models"][model_name] = agg
            aggregate_by_model[model_name].append(agg)
        results["folds"].append(fold_entry)

    summary = {}
    for model_name, rows in aggregate_by_model.items():
        mean_row = {"num_total_eval_queries": sum(r["num_total_eval_queries"] for r in rows) / len(rows), "num_evaluable_queries": sum(r["num_evaluable_queries"] for r in rows) / len(rows), "evaluable_query_coverage": sum(r["evaluable_query_coverage"] for r in rows) / len(rows), "num_pos": sum(r["num_pos"] for r in rows) / len(rows)}
        for k in topk_values:
            mean_row[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / len(rows)
            mean_row[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in rows) / len(rows)
        summary[model_name] = mean_row
    results["summary_by_model"] = summary

    save_json(out_dir / "summary.json", results)

    lines = [
        "# UPIR Candidate-Generation Baselines",
        "",
        f"- dataset: `{bundle.name}`",
        f"- split: `{args.split}`",
        f"- evaluation mode: `candidate_generation_retrieval`",
        f"- query rule used: `>= 1` known positive(s), `>= 0` known negative(s)`",
        f"- models: `{', '.join(model_names)}`",
        "",
        "## Aggregate Results",
        "",
        "| Split | Model | AvgEvalQ/Fold | AvgCoverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in model_names:
        agg = summary[model_name]
        lines.append(
            "| {split} | {model} | {evalq:.0f} | {cov:.3f} | {h10:.4f} | {h50:.4f} | {r10:.4f} | {r50:.4f} |".format(
                split=args.split,
                model=model_name,
                evalq=agg["num_evaluable_queries"],
                cov=agg["evaluable_query_coverage"],
                h10=agg["hit@10"],
                h50=agg["hit@50"],
                r10=agg["recall@10"],
                r50=agg["recall@50"],
            )
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
