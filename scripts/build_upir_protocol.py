#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.text_tokens import stable_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fixed benchmark protocol metadata for UPIR-Strict.")
    p.add_argument("--canonical-parquet", default="data/real_benchmarks/upir/upir_canonical_edges.parquet")
    p.add_argument("--ligands-parquet", default="data/real_benchmarks/upir/upir_ligands.parquet")
    p.add_argument("--proteins-parquet", default="data/real_benchmarks/upir/upir_proteins.parquet")
    p.add_argument("--out-protocol-json", default="data/real_benchmarks/upir/UPIR_strict_forward_protocol.json")
    p.add_argument("--out-stats-json", default="data/real_benchmarks/upir/UPIR_strict_forward_stats.json")
    p.add_argument("--num-folds", type=int, default=5)
    return p.parse_args()


def _sorted_stats(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    return {
        "min": vals[0],
        "median": float(statistics.median(vals)),
        "max": vals[-1],
        "mean": float(sum(vals) / len(vals)),
    }


def _greedy_partition(
    items: List[Dict],
    num_folds: int,
    score_keys: Tuple[str, ...],
    max_items_per_fold: int | None = None,
) -> List[List[Dict]]:
    folds: List[List[Dict]] = [[] for _ in range(num_folds)]
    fold_scores = [{k: 0 for k in score_keys} for _ in range(num_folds)]
    for item in items:
        best_idx = 0
        best_tuple = None
        for idx in range(num_folds):
            if max_items_per_fold is not None and len(folds[idx]) >= max_items_per_fold:
                continue
            candidate = tuple(fold_scores[idx][k] for k in score_keys) + (len(folds[idx]), idx)
            if best_tuple is None or candidate < best_tuple:
                best_tuple = candidate
                best_idx = idx
        folds[best_idx].append(item)
        for key in score_keys:
            fold_scores[best_idx][key] += int(item.get(key, 0))
    return folds


def _pair_fold(pid: str, lid: str, num_folds: int) -> int:
    return stable_seed(f"{pid}|{lid}") % num_folds


def main() -> None:
    args = parse_args()

    proteins = pq.read_table(args.proteins_parquet).to_pylist()
    ligands = pq.read_table(args.ligands_parquet).to_pylist()
    scaffold_by_ligand = {row["ligand_id"]: row["scaffold"] for row in ligands}
    scaffold_semantics = ligands[0].get("scaffold_semantics", "proxy_string_scaffold") if ligands else "proxy_string_scaffold"

    query_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num_pos": 0, "num_neg": 0})
    ligand_degree: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num_pos": 0, "num_neg": 0})
    scaffold_pos_total: Dict[str, int] = defaultdict(int)
    scaffold_neg_total: Dict[str, int] = defaultdict(int)
    scaffold_query_pos: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scaffold_query_neg: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    standard_fold_stats = [{"num_eval_pairs": 0, "num_eval_pos": 0, "num_eval_neg": 0} for _ in range(args.num_folds)]

    pf = pq.ParquetFile(args.canonical_parquet)
    num_edges = 0
    num_pos = 0
    num_neg = 0
    for batch in pf.iter_batches():
        cols = batch.to_pydict()
        for pid, lid, label in zip(cols["query_id"], cols["candidate_id"], cols["label"]):
            label = int(label)
            scaffold = scaffold_by_ligand.get(lid, "UNKNOWN")
            num_edges += 1
            if label == 1:
                query_stats[pid]["num_pos"] += 1
                ligand_degree[lid]["num_pos"] += 1
                scaffold_pos_total[scaffold] += 1
                scaffold_query_pos[scaffold][pid] += 1
                num_pos += 1
            else:
                query_stats[pid]["num_neg"] += 1
                ligand_degree[lid]["num_neg"] += 1
                scaffold_neg_total[scaffold] += 1
                scaffold_query_neg[scaffold][pid] += 1
                num_neg += 1

            fold = _pair_fold(pid, lid, args.num_folds)
            standard_fold_stats[fold]["num_eval_pairs"] += 1
            if label == 1:
                standard_fold_stats[fold]["num_eval_pos"] += 1
            else:
                standard_fold_stats[fold]["num_eval_neg"] += 1

    ligand_count_by_scaffold: Dict[str, int] = defaultdict(int)
    for row in ligands:
        ligand_count_by_scaffold[row["scaffold"]] += 1

    target_items = []
    for row in proteins:
        qid = row["protein_id"]
        target_items.append(
            {
                "query_id": qid,
                "num_pos": query_stats[qid]["num_pos"],
                "num_neg": query_stats[qid]["num_neg"],
                "num_pairs": query_stats[qid]["num_pos"] + query_stats[qid]["num_neg"],
            }
        )
    target_items.sort(key=lambda x: (x["num_pos"], x["num_pairs"], x["query_id"]), reverse=True)
    target_folds = _greedy_partition(
        target_items,
        args.num_folds,
        score_keys=("num_pos", "num_pairs"),
        max_items_per_fold=max(1, len(target_items) // args.num_folds),
    )

    scaffold_items = []
    for scaffold in sorted(ligand_count_by_scaffold.keys()):
        scaffold_items.append(
            {
                "scaffold": scaffold,
                "num_ligands": ligand_count_by_scaffold[scaffold],
                "num_pos": scaffold_pos_total.get(scaffold, 0),
                "num_neg": scaffold_neg_total.get(scaffold, 0),
            }
        )
    scaffold_items.sort(key=lambda x: (x["num_ligands"], x["num_pos"], x["scaffold"]), reverse=True)
    scaffold_folds = _greedy_partition(scaffold_items, args.num_folds, score_keys=("num_ligands", "num_pos"))

    protocol = {
        "name": "UPIR_strict_forward_protocol",
        "protocol_version": "upir_strict_v2_murcko" if scaffold_semantics == "bemis_murcko" else "upir_strict_v1",
        "benchmark_name": "UPIR_strict_forward",
        "benchmark_scope": "protein_pocket_to_unified_ligand_pool",
        "label_regime": "strict",
        "primary_reporting": {
            "query_aggregation": "macro",
            "fold_aggregation": "mean_over_fixed_folds",
            "recommended_metrics": ["MRR", "Recall@10", "Recall@50", "EF1", "EF5"],
            "evaluation_modes": {
                "discriminative_retrieval": {
                    "query_rule": {"min_known_pos": 1, "min_known_neg": 1},
                    "recommended_metrics": ["MRR", "Recall@10", "Recall@50", "EF1", "EF5"],
                    "goal": "negative-aware ranking quality under unified-pool retrieval",
                },
                "candidate_generation_retrieval": {
                    "query_rule": {"min_known_pos": 1, "min_known_neg": 0},
                    "recommended_metrics": ["Recall@10", "Recall@50", "Hit@10", "Hit@50"],
                    "goal": "positive-focused first-stage retrieval without requiring explicit negatives",
                },
            },
            "notes": [
                "Use only known labels for scoring.",
                "Report per-fold and mean-over-fold results.",
                "Use target holdout as the primary OOD setting.",
                "Two reporting modes are valid: discriminative retrieval for negative-aware ranking and candidate-generation retrieval for first-stage filtering.",
            ],
        },
        "splits": {
            "standard": {
                "type": "pair_hash_kfold",
                "num_folds": args.num_folds,
                "hash_rule": "md5_mod_k over query_id|candidate_id",
                "folds": [
                    {
                        "fold_id": idx,
                        **stats,
                    }
                    for idx, stats in enumerate(standard_fold_stats)
                ],
            },
            "target": {
                "type": "fixed_query_kfold",
                "num_folds": args.num_folds,
                "folds": [],
            },
            "scaffold": {
                "type": "fixed_scaffold_kfold",
                "scaffold_field": "ligand.scaffold",
                "scaffold_semantics": scaffold_semantics,
                "num_folds": args.num_folds,
                "folds": [],
            },
            "joint_ood": {
                "type": "paired_target_scaffold_kfold",
                "num_folds": args.num_folds,
                "folds": [],
            },
        },
    }

    for fold_id, items in enumerate(target_folds):
        eval_query_ids = sorted(item["query_id"] for item in items)
        protocol["splits"]["target"]["folds"].append(
            {
                "fold_id": fold_id,
                "eval_query_ids": eval_query_ids,
                "num_eval_queries": len(eval_query_ids),
                "num_eval_pos": int(sum(item["num_pos"] for item in items)),
                "num_eval_pairs": int(sum(item["num_pairs"] for item in items)),
            }
        )

    for fold_id, items in enumerate(scaffold_folds):
        eval_scaffolds = sorted(item["scaffold"] for item in items)
        eval_pos_per_query = defaultdict(int)
        eval_neg_per_query = defaultdict(int)
        num_candidate_ligands = 0
        for item in items:
            scaffold = item["scaffold"]
            num_candidate_ligands += int(item["num_ligands"])
            for qid, count in scaffold_query_pos.get(scaffold, {}).items():
                eval_pos_per_query[qid] += int(count)
            for qid, count in scaffold_query_neg.get(scaffold, {}).items():
                eval_neg_per_query[qid] += int(count)
        positive_covered_queries = sum(1 for qid in query_stats if eval_pos_per_query.get(qid, 0) > 0)
        protocol["splits"]["scaffold"]["folds"].append(
            {
                "fold_id": fold_id,
                "eval_scaffolds": eval_scaffolds,
                "num_eval_scaffolds": len(eval_scaffolds),
                "num_candidate_ligands": num_candidate_ligands,
                "num_eval_pos": int(sum(item["num_pos"] for item in items)),
                "num_eval_neg": int(sum(item["num_neg"] for item in items)),
                "num_queries_with_positive_eval": positive_covered_queries,
                "eval_positive_per_query": {qid: int(v) for qid, v in sorted(eval_pos_per_query.items()) if v > 0},
                "eval_negative_per_query": {qid: int(v) for qid, v in sorted(eval_neg_per_query.items()) if v > 0},
            }
        )

    for fold_id in range(args.num_folds):
        target_fold = protocol["splits"]["target"]["folds"][fold_id]
        scaffold_fold = protocol["splits"]["scaffold"]["folds"][fold_id]
        protocol["splits"]["joint_ood"]["folds"].append(
            {
                "fold_id": fold_id,
                "eval_query_ids": target_fold["eval_query_ids"],
                "eval_scaffolds": scaffold_fold["eval_scaffolds"],
                "num_eval_queries": target_fold["num_eval_queries"],
                "num_eval_scaffolds": scaffold_fold["num_eval_scaffolds"],
                "num_scaffold_eval_pos": scaffold_fold["num_eval_pos"],
            }
        )

    target_positive_rates = []
    for qid, stats in sorted(query_stats.items()):
        denom = stats["num_pos"] + stats["num_neg"]
        target_positive_rates.append(stats["num_pos"] / max(1, denom))

    ligand_pos_degrees = [stats["num_pos"] for stats in ligand_degree.values()]
    stats_payload = {
        "name": "UPIR_strict_forward_stats",
        "protocol_version": protocol["protocol_version"],
        "scaffold_semantics": scaffold_semantics,
        "counts": {
            "num_queries": len(proteins),
            "num_candidates": len(ligands),
            "num_known_pairs": num_edges,
            "num_pos": num_pos,
            "num_neg": num_neg,
            "num_unique_scaffolds": len(ligand_count_by_scaffold),
        },
        "query_positive_count_stats": _sorted_stats(stats["num_pos"] for stats in query_stats.values()),
        "query_positive_rate_stats": _sorted_stats(target_positive_rates),
        "ligand_positive_degree_stats": _sorted_stats(ligand_pos_degrees),
        "protocol_health": {
            "target_fold_eval_queries": [fold["num_eval_queries"] for fold in protocol["splits"]["target"]["folds"]],
            "target_fold_eval_pos": [fold["num_eval_pos"] for fold in protocol["splits"]["target"]["folds"]],
            "scaffold_fold_candidates": [fold["num_candidate_ligands"] for fold in protocol["splits"]["scaffold"]["folds"]],
            "scaffold_fold_queries_with_positive_eval": [
                fold["num_queries_with_positive_eval"] for fold in protocol["splits"]["scaffold"]["folds"]
            ],
        },
        "warnings": [],
    }

    if scaffold_semantics != "bemis_murcko" and len(ligand_count_by_scaffold) > len(ligands) * 0.5:
        stats_payload["warnings"].append(
            "Scaffold cardinality is high relative to ligand count; current scaffold field behaves like a proxy, not a chemistry-grade Murcko scaffold."
        )
    if len(proteins) < 20:
        stats_payload["warnings"].append(
            "Target count is still small; report fold-level uncertainty and avoid over-claiming held-out-target generalization."
        )
    target_fold_pos = stats_payload["protocol_health"]["target_fold_eval_pos"]
    if target_fold_pos and max(target_fold_pos) > 5 * max(1, min(target_fold_pos)):
        stats_payload["warnings"].append(
            "Held-out-target folds remain label-imbalanced because a few assays dominate the positive edge mass; macro averaging is required."
        )
    if max(ligand_pos_degrees, default=0) <= 1:
        stats_payload["warnings"].append(
            "Most ligands have only one positive protein edge in the strict source, so reverse retrieval remains benchmark materialization only."
        )
    scaffold_counts = stats_payload["protocol_health"]["scaffold_fold_candidates"]
    if scaffold_counts and max(scaffold_counts) > 3 * max(1, min(scaffold_counts)):
        stats_payload["warnings"].append(
            "Scaffold folds remain imbalanced; report fold-level coverage and treat scaffold results cautiously."
        )
    if scaffold_semantics == "bemis_murcko":
        stats_payload["warnings"].append(
            "Acyclic ligands fall into the `NO_MURCKO` bucket under chemistry-grade scaffold extraction; coverage should be reported explicitly."
        )

    protocol_path = Path(args.out_protocol_json)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    stats_path = Path(args.out_stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_protocol_json": str(protocol_path),
                "out_stats_json": str(stats_path),
                "counts": stats_payload["counts"],
                "warnings": stats_payload["warnings"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
