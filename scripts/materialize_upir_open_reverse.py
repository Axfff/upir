#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize UPIR open reverse from a forward UPIR open benchmark.")
    p.add_argument("--forward-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_forward.json")
    p.add_argument("--out-dataset", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json")
    p.add_argument("--out-protocol", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse_protocol.json")
    p.add_argument("--out-stats", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse_stats.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--timestamp", default="")
    return p.parse_args()


def _protein_scaffold(pid: str) -> str:
    body = pid.split("::", 1)[-1]
    family = body.split("_", 1)[0]
    return f"BIGBIND_PROTEIN::{family}"


def _summary(values: List[int]) -> Dict[str, float]:
    vals = sorted(values)
    if not vals:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return {"min": float(vals[0]), "median": float(median), "max": float(vals[-1]), "mean": float(sum(vals)) / n}


def _build_protocol(labels: Dict[str, Dict[str, int]], num_queries: int, num_folds: int) -> Dict:
    folds = []
    for fold in range(num_folds):
        eval_pairs = 0
        eval_pos = 0
        eval_neg = 0
        eval_pos_queries = set()
        eval_discr_queries = set()
        per_query = defaultdict(lambda: [0, 0])
        for qid, row in labels.items():
            for cid, label in row.items():
                if stable_seed(f"{qid}|{cid}") % num_folds != fold:
                    continue
                eval_pairs += 1
                if int(label) == 1:
                    eval_pos += 1
                    eval_pos_queries.add(qid)
                    per_query[qid][0] += 1
                else:
                    eval_neg += 1
                    per_query[qid][1] += 1
        for qid, (pos, neg) in per_query.items():
            if pos >= 1 and neg >= 1:
                eval_discr_queries.add(qid)
        folds.append(
            {
                "fold_id": fold,
                "num_eval_pairs": eval_pairs,
                "num_eval_pos": eval_pos,
                "num_eval_neg": eval_neg,
                "num_candidate_generation_queries": len(eval_pos_queries),
                "num_discriminative_queries": len(eval_discr_queries),
                "candidate_generation_coverage_over_all_queries": len(eval_pos_queries) / max(1, num_queries),
                "discriminative_coverage_over_all_queries": len(eval_discr_queries) / max(1, num_queries),
            }
        )
    return {
        "name": "UPIR_open_bigbind_reverse_protocol",
        "protocol_version": "upir_open_bigbind_reverse_v1",
        "benchmark_name": "UPIR_open_bigbind_reverse",
        "benchmark_scope": "ligand_to_unified_protein_pocket_pool",
        "label_regime": "binary_activity_flag_from_bigbind_transposed",
        "primary_reporting": {
            "query_aggregation": "macro",
            "fold_aggregation": "mean_over_fixed_folds",
            "evaluation_modes": {
                "candidate_generation_retrieval": {
                    "query_rule": {"min_known_pos": 1, "min_known_neg": 0},
                    "recommended_metrics": ["Hit@1", "Hit@5", "Hit@10", "Hit@50", "Recall@1", "Recall@5", "Recall@10", "Recall@50", "best_positive_rank"],
                    "goal": "positive-focused ligand-to-protein target proposal",
                },
                "discriminative_retrieval": {
                    "query_rule": {"min_known_pos": 1, "min_known_neg": 1},
                    "recommended_metrics": ["MRR", "EF1", "EF5", "Recall@K"],
                    "goal": "auxiliary negative-aware calibration where known negatives exist",
                },
            },
            "notes": [
                "Use held-out pair labels for standard split evaluation.",
                "Treat candidate-generation retrieval as the primary reverse mode.",
                "Treat discriminative reverse as auxiliary because query coverage is much lower.",
            ],
        },
        "splits": {
            "standard": {
                "type": "pair_hash_kfold",
                "num_folds": num_folds,
                "hash_rule": "stable_seed over query_id|candidate_id modulo num_folds",
                "folds": folds,
            }
        },
    }


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    forward_path = Path(args.forward_path)
    out_dataset = Path(args.out_dataset)
    out_protocol = Path(args.out_protocol)
    out_stats = Path(args.out_stats)
    stamped_dataset = out_dataset.with_name(f"{out_dataset.stem}_{stamp}{out_dataset.suffix}")
    stamped_protocol = out_protocol.with_name(f"{out_protocol.stem}_{stamp}{out_protocol.suffix}")
    stamped_stats = out_stats.with_name(f"{out_stats.stem}_{stamp}{out_stats.suffix}")

    with forward_path.open("r", encoding="utf-8") as f:
        forward = json.load(f)

    query_records = [
        {"pid": item["lid"], "text": item.get("text", ""), "text_kind": item.get("text_kind", "smiles")}
        for item in forward["ligands"]
    ]
    candidate_records = [
        {
            "lid": item["pid"],
            "text": item.get("text", ""),
            "text_kind": item.get("text_kind", "protein_residue"),
            "scaffold": _protein_scaffold(item["pid"]),
        }
        for item in forward["pockets"]
    ]

    labels: Dict[str, Dict[str, int]] = defaultdict(dict)
    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    for pid, row in forward["labels"].items():
        for lid, label in row.items():
            labels[lid][pid] = int(label)
            if int(label) == 1:
                pos_counts[lid] += 1
            else:
                neg_counts[lid] += 1

    label_dict = {qid: dict(row) for qid, row in labels.items()}
    for item in query_records:
        label_dict.setdefault(item["pid"], {})

    positive_pairs = sum(pos_counts.values())
    negative_pairs = sum(neg_counts.values())
    meta = {
        "benchmark_type": "upir_open_bigbind_reverse",
        "source": "BigBind",
        "source_forward_benchmark": forward.get("name", forward_path.stem),
        "label_regime": "binary_activity_flag_from_bigbind_transposed",
        "query_entity": "ligand",
        "candidate_entity": "protein_pocket",
        "protocol_path": out_protocol.name,
        "primary_evaluable_query_rule": {"min_known_pos": 1, "min_known_neg": 0},
        "tokenization": forward.get("meta", {}).get("tokenization", {}),
        "counts": {
            "num_queries": len(query_records),
            "num_candidates": len(candidate_records),
            "num_known_pairs": positive_pairs + negative_pairs,
            "num_pos": positive_pairs,
            "num_neg": negative_pairs,
        },
        "note": "Open reverse UPIR benchmark materialized by transposing UPIR_open_bigbind_forward labels.",
    }

    payload = {
        "name": "UPIR_open_bigbind_reverse",
        "dim": int(forward.get("dim", 64)),
        "pockets": query_records,
        "ligands": candidate_records,
        "labels": label_dict,
        "meta": meta,
    }
    protocol = _build_protocol(label_dict, num_queries=len(query_records), num_folds=args.folds)
    stats = {
        "name": "UPIR_open_bigbind_reverse_stats",
        "protocol_version": "upir_open_bigbind_reverse_v1",
        "counts": meta["counts"],
        "query_positive_count_stats": _summary([pos_counts.get(item["pid"], 0) for item in query_records]),
        "query_negative_count_stats": _summary([neg_counts.get(item["pid"], 0) for item in query_records]),
        "protocol_health": {
            "standard_fold_candidate_generation_queries": [f["num_candidate_generation_queries"] for f in protocol["splits"]["standard"]["folds"]],
            "standard_fold_discriminative_queries": [f["num_discriminative_queries"] for f in protocol["splits"]["standard"]["folds"]],
            "standard_fold_eval_pos": [f["num_eval_pos"] for f in protocol["splits"]["standard"]["folds"]],
            "standard_fold_eval_neg": [f["num_eval_neg"] for f in protocol["splits"]["standard"]["folds"]],
        },
        "warnings": [
            "Reverse candidate generation is primary; reverse discriminative coverage is much lower.",
            "Most positive ligand queries have one known protein target, so Hit@K and best-positive-rank should be prominent.",
            "BigBind open labels are not strict confirmatory negatives.",
        ],
    }

    save_json(stamped_dataset, payload)
    save_json(stamped_protocol, protocol)
    save_json(stamped_stats, stats)
    ensure_dir(out_dataset.parent)
    shutil.copyfile(stamped_dataset, out_dataset)
    shutil.copyfile(stamped_protocol, out_protocol)
    shutil.copyfile(stamped_stats, out_stats)
    print(json.dumps({"dataset": str(out_dataset), "protocol": str(out_protocol), "stats": str(out_stats), "timestamp": stamp}, indent=2))


if __name__ == "__main__":
    main()
