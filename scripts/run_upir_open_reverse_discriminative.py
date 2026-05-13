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
    p = argparse.ArgumentParser(description="Run leakage-safe discriminative diagnostics on UPIR open reverse.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json")
    p.add_argument("--out-dir", default="results/upir_open_bigbind_reverse_discriminative_standard_v1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="RANDOM,POPULARITY")
    p.add_argument("--topk", default="1,5,10,50")
    p.add_argument("--ef-percents", default="1,5")
    p.add_argument("--min-known-neg", type=int, default=1)
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _top_k_size(total: int, percent: float) -> int:
    return max(1, int(round(total * percent / 100.0)))


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


def _random_order(candidate_ids: Sequence[str], fold: int, qid: str) -> List[str]:
    return sorted(candidate_ids, key=lambda cid: stable_seed(f"open_reverse_disc_random:{fold}:{qid}:{cid}"))


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


def _metrics(
    ranking: Sequence[str],
    eval_row: Dict[str, int],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
    min_known_neg: int,
) -> Dict[str, float]:
    ranked_ids = [cid for cid in ranking if cid in eval_row]
    positives = [cid for cid in ranked_ids if int(eval_row[cid]) == 1]
    negatives = [cid for cid in ranked_ids if int(eval_row[cid]) == 0]
    if len(positives) < 1 or len(negatives) < min_known_neg:
        return {}

    pos_ranks = [idx + 1 for idx, cid in enumerate(ranked_ids) if int(eval_row[cid]) == 1]
    first = min(pos_ranks)
    out: Dict[str, float] = {
        "mrr": 1.0 / first,
        "best_positive_rank": float(first),
        "num_pos": float(len(positives)),
        "num_neg": float(len(negatives)),
        "num_known_pairs": float(len(ranked_ids)),
    }
    for k in topk_values:
        top_ids = ranked_ids[:k]
        tp = sum(1 for cid in top_ids if int(eval_row.get(cid, 0)) == 1)
        out[f"hit@{k}"] = 1.0 if tp else 0.0
        out[f"recall@{k}"] = tp / len(positives)
    base_rate = len(positives) / max(1, len(ranked_ids))
    for p in ef_percents:
        top_n = _top_k_size(len(ranked_ids), p)
        top_ids = ranked_ids[:top_n]
        top_rate = sum(1 for cid in top_ids if int(eval_row.get(cid, 0)) == 1) / max(1, top_n)
        out[f"ef{int(p)}"] = top_rate / max(base_rate, 1e-12)
    return out


def _aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"num_eval_queries": 0.0}
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


def _write_md(path: Path, payload: Dict) -> None:
    lines = [
        "# UPIR Open Reverse Discriminative Diagnostics",
        "",
        f"- dataset: `{payload['dataset_name']}`",
        f"- split: `standard` pair-hash `{payload['folds']}`-fold",
        "- evaluation mode: `discriminative_retrieval`",
        f"- query rule: held-out labels contain `>=1` known positive and `>={payload['query_rule_used']['min_known_neg']}` known negative protein(s)",
        "- note: this is an auxiliary diagnostic, not the reverse headline task",
        "",
        "## Aggregate Results",
        "",
        "| Model | AvgEvalQ/Fold | Coverage | MRR | Hit@1 | Hit@5 | Hit@10 | Recall@10 | Recall@50 | EF1 | EF5 | BestPosRank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in payload["models"]:
        row = payload["summary_by_model"][model]
        lines.append(
            f"| {model} | {row.get('num_eval_queries', 0.0):.0f} | {row.get('evaluable_query_coverage', 0.0):.4f} | "
            f"{row.get('mrr', 0.0):.4f} | {row.get('hit@1', 0.0):.4f} | {row.get('hit@5', 0.0):.4f} | {row.get('hit@10', 0.0):.4f} | "
            f"{row.get('recall@10', 0.0):.4f} | {row.get('recall@50', 0.0):.4f} | {row.get('ef1', 0.0):.4f} | {row.get('ef5', 0.0):.4f} | "
            f"{row.get('best_positive_rank', 0.0):.2f} |"
        )
    lines.extend(["", "## Fold Coverage", "", "| Fold | Discriminative Queries | Coverage | Eval Pos Pairs | Eval Neg Pairs |", "|---:|---:|---:|---:|---:|"])
    for fold in payload["folds_detail"]:
        lines.append(f"| {fold['fold']} | {fold['num_evaluable_queries']} | {fold['evaluable_query_coverage']:.4f} | {fold['num_eval_pos_pairs']} | {fold['num_eval_neg_pairs']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    topk_values = _parse_ints(args.topk)
    ef_percents = _parse_floats(args.ef_percents)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    bundle = load_dataset(Path(args.dataset_path))
    candidate_ids = [l.lid for l in bundle.ligands]
    total_queries = len(bundle.pockets)
    results = {
        "dataset_name": bundle.name,
        "dataset_path": args.dataset_path,
        "evaluation_mode": "discriminative_retrieval",
        "split": "standard_pair_hash",
        "folds": args.folds,
        "models": model_names,
        "query_rule_used": {"min_known_pos": 1, "min_known_neg": args.min_known_neg, "label_scope": "held_out_pairs_only"},
        "num_total_queries": total_queries,
        "num_candidate_proteins": len(candidate_ids),
        "folds_detail": [],
        "summary_by_model": {},
    }
    per_model_folds = defaultdict(list)
    for fold in range(args.folds):
        train_pairs, eval_labels = _split_pairs(bundle.labels, fold=fold, folds=args.folds)
        eval_query_ids = [
            qid
            for qid, row in eval_labels.items()
            if any(int(v) == 1 for v in row.values()) and sum(1 for v in row.values() if int(v) == 0) >= args.min_known_neg
        ]
        num_eval_pos_pairs = sum(1 for row in eval_labels.values() for label in row.values() if int(label) == 1)
        num_eval_neg_pairs = sum(1 for row in eval_labels.values() for label in row.values() if int(label) == 0)
        fold_entry = {
            "fold": fold,
            "num_eval_pos_pairs": num_eval_pos_pairs,
            "num_eval_neg_pairs": num_eval_neg_pairs,
            "num_evaluable_queries": len(eval_query_ids),
            "evaluable_query_coverage": len(eval_query_ids) / max(1, total_queries),
            "models": {},
        }
        popularity = _popularity_order(train_pairs, candidate_ids) if "POPULARITY" in model_names else []
        for model in model_names:
            rows = []
            for qid in eval_query_ids:
                if model == "RANDOM":
                    ranking = _random_order(candidate_ids, fold, qid)
                elif model == "POPULARITY":
                    ranking = popularity
                else:
                    raise ValueError(f"Unsupported model: {model}")
                row = _metrics(ranking, eval_labels[qid], topk_values, ef_percents, args.min_known_neg)
                if row:
                    rows.append(row)
            agg = _aggregate(rows)
            agg["num_total_queries"] = float(total_queries)
            agg["evaluable_query_coverage"] = len(rows) / max(1, total_queries)
            fold_entry["models"][model] = agg
            per_model_folds[model].append(agg)
        results["folds_detail"].append(fold_entry)
    for model in model_names:
        results["summary_by_model"][model] = _average(per_model_folds[model])
    save_json(out_dir / "summary.json", results)
    _write_md(out_dir / "SUMMARY.md", results)
    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
