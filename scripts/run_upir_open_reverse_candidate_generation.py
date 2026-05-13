#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import load_dataset
from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run leakage-safe candidate-generation baselines on UPIR open reverse.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json")
    p.add_argument("--out-dir", default="results/upir_open_bigbind_reverse_candidate_generation_standard_v1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="RANDOM,POPULARITY")
    p.add_argument("--topk", default="1,5,10,50")
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _random_order(candidate_ids: Sequence[str], fold: int) -> List[str]:
    return sorted(candidate_ids, key=lambda cid: stable_seed(f"open_reverse_random:{fold}:{cid}"))


def _popularity_order(train_pairs: Sequence[tuple[str, str, int]], candidate_ids: Sequence[str]) -> List[str]:
    pos_counts = Counter()
    known_counts = Counter()
    for _, cid, label in train_pairs:
        known_counts[cid] += 1
        if int(label) == 1:
            pos_counts[cid] += 1
    return sorted(
        candidate_ids,
        key=lambda cid: (
            -pos_counts[cid],
            -(pos_counts[cid] / max(1, known_counts[cid])),
            stable_seed(cid),
        ),
    )


def _split_pairs(labels: Dict[str, Dict[str, int]], fold: int, folds: int):
    train_pairs = []
    eval_labels = defaultdict(dict)
    for qid, row in labels.items():
        for cid, label in row.items():
            pair_fold = stable_seed(f"{qid}|{cid}") % folds
            if pair_fold == fold:
                eval_labels[qid][cid] = int(label)
            else:
                train_pairs.append((qid, cid, int(label)))
    return train_pairs, eval_labels


def _query_metrics(order: Sequence[str], eval_row: Dict[str, int], rank: Dict[str, int], topk_values: Sequence[int]) -> Dict[str, float]:
    pos_ids = [cid for cid, label in eval_row.items() if int(label) == 1]
    if not pos_ids:
        return {}
    pos_ranks = sorted(rank[cid] for cid in pos_ids if cid in rank)
    if not pos_ranks:
        return {}
    out = {
        "mrr": 1.0 / pos_ranks[0],
        "best_positive_rank": float(pos_ranks[0]),
        "num_pos": float(len(pos_ranks)),
        "num_known_pairs": float(len(eval_row)),
    }
    for k in topk_values:
        hits = sum(1 for r in pos_ranks if r <= k)
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        out[f"recall@{k}"] = hits / len(pos_ranks)
    return out


def _aggregate(rows: Sequence[Dict[str, float]], topk_values: Sequence[int]) -> Dict[str, float]:
    if not rows:
        out = {"num_eval_queries": 0.0, "mrr": 0.0, "best_positive_rank": 0.0, "num_pos": 0.0, "num_known_pairs": 0.0}
        for k in topk_values:
            out[f"hit@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
        return out
    keys = sorted({k for row in rows for k in row})
    out = {"num_eval_queries": float(len(rows))}
    for key in keys:
        out[key] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    return out


def _average(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def _write_md(path: Path, payload: Dict, topk_values: Sequence[int]) -> None:
    lines = [
        "# UPIR Open Reverse Candidate-Generation Baselines",
        "",
        f"- dataset: `{payload['dataset_name']}`",
        f"- split: `standard` pair-hash `{payload['folds']}`-fold",
        "- evaluation mode: `candidate_generation_retrieval`",
        "- query rule: held-out labels contain `>=1` known positive protein",
        "- note: evaluation uses held-out pair labels only; training labels are excluded from the evaluated fold",
        "",
        "## Aggregate Results",
        "",
        "| Model | AvgEvalQ/Fold | AvgCoverage | MRR | Hit@1 | Hit@5 | Hit@10 | Hit@50 | Recall@1 | Recall@5 | Recall@10 | Recall@50 | BestPosRank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in payload["models"]:
        row = payload["summary_by_model"][model_name]
        lines.append(
            f"| {model_name} | {row.get('num_eval_queries', 0.0):.0f} | {row.get('evaluable_query_coverage', 0.0):.3f} | {row.get('mrr', 0.0):.4f} | "
            f"{row.get('hit@1', 0.0):.4f} | {row.get('hit@5', 0.0):.4f} | {row.get('hit@10', 0.0):.4f} | {row.get('hit@50', 0.0):.4f} | "
            f"{row.get('recall@1', 0.0):.4f} | {row.get('recall@5', 0.0):.4f} | {row.get('recall@10', 0.0):.4f} | {row.get('recall@50', 0.0):.4f} | "
            f"{row.get('best_positive_rank', 0.0):.2f} |"
        )

    lines.extend(["", "## Fold Coverage", "", "| Fold | Eval Positive Queries | Coverage | Eval Pos Pairs | Eval Neg Pairs |", "|---:|---:|---:|---:|---:|"])
    for fold in payload["folds_detail"]:
        lines.append(
            f"| {fold['fold']} | {fold['num_evaluable_queries']} | {fold['evaluable_query_coverage']:.4f} | {fold['num_eval_pos_pairs']} | {fold['num_eval_neg_pairs']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    topk_values = _parse_ints(args.topk)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    bundle = load_dataset(Path(args.dataset_path))
    if bundle.meta and bundle.meta.get("query_entity") != "ligand":
        raise ValueError("Expected a reverse dataset with ligand queries.")

    query_ids = [p.pid for p in bundle.pockets]
    candidate_ids = [l.lid for l in bundle.ligands]
    results = {
        "dataset_name": bundle.name,
        "dataset_path": args.dataset_path,
        "evaluation_mode": "candidate_generation_retrieval",
        "split": "standard_pair_hash",
        "folds": args.folds,
        "models": model_names,
        "query_rule_used": {"min_known_pos": 1, "min_known_neg": 0, "label_scope": "held_out_pairs_only"},
        "num_total_queries": len(query_ids),
        "num_candidate_proteins": len(candidate_ids),
        "folds_detail": [],
        "summary_by_model": {},
    }
    per_model_folds = defaultdict(list)

    for fold in range(args.folds):
        train_pairs, eval_labels = _split_pairs(bundle.labels, fold=fold, folds=args.folds)
        num_eval_pos_pairs = sum(1 for row in eval_labels.values() for label in row.values() if int(label) == 1)
        num_eval_neg_pairs = sum(1 for row in eval_labels.values() for label in row.values() if int(label) == 0)
        eval_query_ids = [qid for qid, row in eval_labels.items() if any(int(v) == 1 for v in row.values())]
        fold_entry = {
            "fold": fold,
            "num_eval_pos_pairs": num_eval_pos_pairs,
            "num_eval_neg_pairs": num_eval_neg_pairs,
            "num_evaluable_queries": len(eval_query_ids),
            "evaluable_query_coverage": len(eval_query_ids) / max(1, len(query_ids)),
            "models": {},
        }

        rankings = {}
        if "RANDOM" in model_names:
            rankings["RANDOM"] = _random_order(candidate_ids, fold)
        if "POPULARITY" in model_names:
            rankings["POPULARITY"] = _popularity_order(train_pairs, candidate_ids)

        for model_name in model_names:
            order = rankings[model_name]
            rank = {cid: idx + 1 for idx, cid in enumerate(order)}
            rows = []
            for qid in eval_query_ids:
                row = _query_metrics(order, eval_labels[qid], rank, topk_values)
                if row:
                    rows.append(row)
            agg = _aggregate(rows, topk_values)
            agg["num_total_queries"] = float(len(query_ids))
            agg["evaluable_query_coverage"] = len(rows) / max(1, len(query_ids))
            fold_entry["models"][model_name] = agg
            per_model_folds[model_name].append(agg)
        results["folds_detail"].append(fold_entry)

    for model_name in model_names:
        results["summary_by_model"][model_name] = _average(per_model_folds[model_name])

    save_json(out_dir / "summary.json", results)
    _write_md(out_dir / "SUMMARY.md", results, topk_values)
    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
