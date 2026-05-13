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
    p = argparse.ArgumentParser(description="Build a generic fixed protocol for a UPIR-style forward benchmark.")
    p.add_argument("--canonical-parquet", required=True)
    p.add_argument("--ligands-parquet", required=True)
    p.add_argument("--proteins-parquet", required=True)
    p.add_argument("--out-protocol-json", required=True)
    p.add_argument("--out-stats-json", required=True)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--protocol-name", default="UPIR_open_forward_protocol")
    p.add_argument("--protocol-version", default="upir_open_v0")
    p.add_argument("--benchmark-name", default="UPIR_open_forward")
    p.add_argument("--benchmark-scope", default="protein_pocket_to_unified_ligand_pool")
    p.add_argument("--label-regime", default="mixed_assay_plus_decoy")
    p.add_argument("--min-known-pos", type=int, default=1)
    p.add_argument("--min-known-neg", type=int, default=0)
    return p.parse_args()


def _sorted_stats(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    return {"min": vals[0], "median": float(statistics.median(vals)), "max": vals[-1], "mean": float(sum(vals) / len(vals))}


def _greedy_partition(items: List[Dict], num_folds: int, score_keys: Tuple[str, ...], max_items_per_fold=None) -> List[List[Dict]]:
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


def _is_evaluable(num_pos: int, num_neg: int, min_known_pos: int, min_known_neg: int) -> bool:
    return int(num_pos) >= int(min_known_pos) and int(num_neg) >= int(min_known_neg)


def main() -> None:
    args = parse_args()
    proteins = pq.read_table(args.proteins_parquet).to_pylist()
    ligands = pq.read_table(args.ligands_parquet).to_pylist()
    scaffold_by_ligand = {row["ligand_id"]: row["scaffold"] for row in ligands}
    scaffold_semantics = ligands[0].get("scaffold_semantics", "proxy_string_scaffold") if ligands else "proxy_string_scaffold"
    protein_cluster_by_query = {row["protein_id"]: row.get("protein_cluster", row["protein_id"]) for row in proteins}

    query_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num_pos": 0, "num_neg": 0})
    ligand_degree: Dict[str, Dict[str, int]] = defaultdict(lambda: {"num_pos": 0, "num_neg": 0})
    scaffold_pos_total: Dict[str, int] = defaultdict(int)
    scaffold_neg_total: Dict[str, int] = defaultdict(int)
    scaffold_query_pos: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    scaffold_query_neg: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    standard_fold_stats = [{"num_eval_pairs": 0, "num_eval_pos": 0, "num_eval_neg": 0} for _ in range(args.num_folds)]
    standard_fold_query_stats: List[Dict[str, Dict[str, int]]] = [defaultdict(lambda: {"num_pos": 0, "num_neg": 0}) for _ in range(args.num_folds)]

    num_edges = 0
    num_pos = 0
    num_neg = 0
    pf = pq.ParquetFile(args.canonical_parquet)
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
                standard_fold_query_stats[fold][pid]["num_pos"] += 1
            else:
                standard_fold_stats[fold]["num_eval_neg"] += 1
                standard_fold_query_stats[fold][pid]["num_neg"] += 1

    ligand_count_by_scaffold: Dict[str, int] = defaultdict(int)
    for row in ligands:
        ligand_count_by_scaffold[row["scaffold"]] += 1

    target_items = []
    rec_cluster_items_map: Dict[str, Dict[str, object]] = {}
    for row in proteins:
        qid = row["protein_id"]
        item = {
            "query_id": qid,
            "num_pos": query_stats[qid]["num_pos"],
            "num_neg": query_stats[qid]["num_neg"],
            "num_pairs": query_stats[qid]["num_pos"] + query_stats[qid]["num_neg"],
        }
        target_items.append(item)
        cluster_id = str(protein_cluster_by_query.get(qid, qid))
        if cluster_id not in rec_cluster_items_map:
            rec_cluster_items_map[cluster_id] = {
                "cluster_id": cluster_id,
                "num_pos": 0,
                "num_neg": 0,
                "num_pairs": 0,
                "num_queries": 0,
                "query_ids": [],
            }
        cluster_item = rec_cluster_items_map[cluster_id]
        cluster_item["num_pos"] += item["num_pos"]
        cluster_item["num_neg"] += item["num_neg"]
        cluster_item["num_pairs"] += item["num_pairs"]
        cluster_item["num_queries"] += 1
        cluster_item["query_ids"].append(qid)
    target_items.sort(key=lambda x: (x["num_pos"], x["num_pairs"], x["query_id"]), reverse=True)
    target_folds = _greedy_partition(target_items, args.num_folds, score_keys=("num_pos", "num_pairs"), max_items_per_fold=max(1, len(target_items) // args.num_folds))
    rec_cluster_items = list(rec_cluster_items_map.values())
    rec_cluster_items.sort(key=lambda x: (x["num_pos"], x["num_pairs"], x["num_queries"], x["cluster_id"]), reverse=True)
    rec_cluster_folds = _greedy_partition(rec_cluster_items, args.num_folds, score_keys=("num_pos", "num_pairs", "num_queries"))

    scaffold_items = []
    for scaffold in sorted(ligand_count_by_scaffold.keys()):
        scaffold_items.append({
            "scaffold": scaffold,
            "num_ligands": ligand_count_by_scaffold[scaffold],
            "num_pos": scaffold_pos_total.get(scaffold, 0),
            "num_neg": scaffold_neg_total.get(scaffold, 0),
        })
    scaffold_items.sort(key=lambda x: (x["num_ligands"], x["num_pos"], x["scaffold"]), reverse=True)
    scaffold_folds = _greedy_partition(scaffold_items, args.num_folds, score_keys=("num_ligands", "num_pos"))

    protocol = {
        "name": args.protocol_name,
        "protocol_version": args.protocol_version,
        "benchmark_name": args.benchmark_name,
        "benchmark_scope": args.benchmark_scope,
        "label_regime": args.label_regime,
        "primary_reporting": {
            "query_aggregation": "macro",
            "fold_aggregation": "mean_over_fixed_folds",
            "recommended_metrics": ["MRR", "Recall@10", "Recall@50", "EF1", "EF5"],
            "evaluation_modes": {
                "discriminative_retrieval": {
                    "query_rule": {"min_known_pos": max(1, args.min_known_pos), "min_known_neg": max(1, args.min_known_neg)},
                    "recommended_metrics": ["MRR", "Recall@10", "Recall@50", "EF1", "EF5"],
                    "goal": "negative-aware ranking quality under unified-pool retrieval",
                },
                "candidate_generation_retrieval": {
                    "query_rule": {"min_known_pos": max(1, args.min_known_pos), "min_known_neg": 0},
                    "recommended_metrics": ["Recall@10", "Recall@50", "Hit@10", "Hit@50"],
                    "goal": "positive-focused first-stage retrieval without requiring explicit negatives",
                },
            },
            "notes": [
                "Use only known labels for scoring.",
                f"Primary evaluable query rule: >= {args.min_known_pos} known positive(s) and >= {args.min_known_neg} known negative(s) within the evaluation candidate pool.",
                "Report per-fold and mean-over-fold results.",
                "Treat target holdout as the primary OOD setting.",
                "Two reporting modes are valid: discriminative retrieval for negative-aware ranking and candidate-generation retrieval for first-stage filtering.",
            ],
        },
        "splits": {
            "standard": {"type": "pair_hash_kfold", "num_folds": args.num_folds, "hash_rule": "md5_mod_k over query_id|candidate_id", "folds": [{"fold_id": idx, **stats} for idx, stats in enumerate(standard_fold_stats)]},
            "target": {"type": "fixed_query_kfold", "num_folds": args.num_folds, "folds": []},
            "target_rec_cluster": {"type": "fixed_rec_cluster_kfold", "cluster_field": "protein.protein_cluster", "num_folds": args.num_folds, "folds": []},
            "scaffold": {"type": "fixed_scaffold_kfold", "scaffold_field": "ligand.scaffold", "scaffold_semantics": scaffold_semantics, "num_folds": args.num_folds, "folds": []},
            "joint_ood": {"type": "paired_target_scaffold_kfold", "num_folds": args.num_folds, "folds": []},
        },
    }

    for fold_id in range(args.num_folds):
        protocol["splits"]["standard"]["folds"][fold_id]["num_evaluable_queries"] = sum(
            1
            for qstats in standard_fold_query_stats[fold_id].values()
            if _is_evaluable(qstats["num_pos"], qstats["num_neg"], args.min_known_pos, args.min_known_neg)
        )

    for fold_id, items in enumerate(target_folds):
        eval_query_ids = sorted(item["query_id"] for item in items)
        protocol["splits"]["target"]["folds"].append({
            "fold_id": fold_id,
            "eval_query_ids": eval_query_ids,
            "num_eval_queries": len(eval_query_ids),
            "num_eval_pos": int(sum(item["num_pos"] for item in items)),
            "num_eval_pairs": int(sum(item["num_pairs"] for item in items)),
            "num_evaluable_queries": int(
                sum(1 for item in items if _is_evaluable(item["num_pos"], item["num_neg"], args.min_known_pos, args.min_known_neg))
            ),
        })

    for fold_id, items in enumerate(rec_cluster_folds):
        eval_query_ids = sorted(qid for item in items for qid in item["query_ids"])
        protocol["splits"]["target_rec_cluster"]["folds"].append({
            "fold_id": fold_id,
            "eval_rec_clusters": sorted(str(item["cluster_id"]) for item in items),
            "eval_query_ids": eval_query_ids,
            "num_eval_rec_clusters": len(items),
            "num_eval_queries": len(eval_query_ids),
            "num_eval_pos": int(sum(item["num_pos"] for item in items)),
            "num_eval_pairs": int(sum(item["num_pairs"] for item in items)),
            "num_evaluable_queries": int(
                sum(
                    1
                    for qid in eval_query_ids
                    if _is_evaluable(query_stats[qid]["num_pos"], query_stats[qid]["num_neg"], args.min_known_pos, args.min_known_neg)
                )
            ),
        })

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
        evaluable_queries = sum(
            1
            for qid in query_stats
            if _is_evaluable(eval_pos_per_query.get(qid, 0), eval_neg_per_query.get(qid, 0), args.min_known_pos, args.min_known_neg)
        )
        protocol["splits"]["scaffold"]["folds"].append({
            "fold_id": fold_id,
            "eval_scaffolds": eval_scaffolds,
            "num_eval_scaffolds": len(eval_scaffolds),
            "num_candidate_ligands": num_candidate_ligands,
            "num_eval_pos": int(sum(item["num_pos"] for item in items)),
            "num_eval_neg": int(sum(item["num_neg"] for item in items)),
            "num_queries_with_positive_eval": positive_covered_queries,
            "num_evaluable_queries": evaluable_queries,
        })

    for fold_id in range(args.num_folds):
        target_fold = protocol["splits"]["target"]["folds"][fold_id]
        scaffold_fold = protocol["splits"]["scaffold"]["folds"][fold_id]
        protocol["splits"]["joint_ood"]["folds"].append({
            "fold_id": fold_id,
            "eval_query_ids": target_fold["eval_query_ids"],
            "eval_scaffolds": scaffold_fold["eval_scaffolds"],
            "num_eval_queries": target_fold["num_eval_queries"],
            "num_eval_scaffolds": scaffold_fold["num_eval_scaffolds"],
            "num_scaffold_eval_pos": scaffold_fold["num_eval_pos"],
        })

    target_positive_rates = []
    for qid, stats in sorted(query_stats.items()):
        denom = stats["num_pos"] + stats["num_neg"]
        target_positive_rates.append(stats["num_pos"] / max(1, denom))

    ligand_pos_degrees = [stats["num_pos"] for stats in ligand_degree.values()]
    stats_payload = {
        "name": f"{args.benchmark_name}_stats",
        "protocol_version": args.protocol_version,
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
            "target_fold_evaluable_queries": [fold["num_evaluable_queries"] for fold in protocol["splits"]["target"]["folds"]],
            "target_fold_eval_pos": [fold["num_eval_pos"] for fold in protocol["splits"]["target"]["folds"]],
            "target_rec_cluster_fold_eval_queries": [fold["num_eval_queries"] for fold in protocol["splits"]["target_rec_cluster"]["folds"]],
            "target_rec_cluster_fold_evaluable_queries": [fold["num_evaluable_queries"] for fold in protocol["splits"]["target_rec_cluster"]["folds"]],
            "target_rec_cluster_fold_clusters": [fold["num_eval_rec_clusters"] for fold in protocol["splits"]["target_rec_cluster"]["folds"]],
            "scaffold_fold_candidates": [fold["num_candidate_ligands"] for fold in protocol["splits"]["scaffold"]["folds"]],
            "scaffold_fold_queries_with_positive_eval": [fold["num_queries_with_positive_eval"] for fold in protocol["splits"]["scaffold"]["folds"]],
            "scaffold_fold_evaluable_queries": [fold["num_evaluable_queries"] for fold in protocol["splits"]["scaffold"]["folds"]],
        },
        "warnings": [],
    }

    if len(proteins) < 40:
        stats_payload["warnings"].append("Target count is improved over 15 but still below the stronger 40-50 query benchmark goal.")
    if target_positive_rates and max(target_positive_rates) > 5 * max(1e-12, min(target_positive_rates)):
        stats_payload["warnings"].append("Target positive rates remain imbalanced; macro-over-query reporting is required.")
    if args.label_regime != "strict":
        stats_payload["warnings"].append("Labels are not strict assay-confirmed negatives across the whole benchmark; keep claims scoped to open/mixed-label retrieval.")
    if args.min_known_neg > 0:
        stats_payload["warnings"].append("Primary ranking tables should report only evaluable queries with both known positives and known negatives in the evaluation pool.")

    protocol_path = Path(args.out_protocol_json)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    stats_path = Path(args.out_stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_protocol_json": str(protocol_path), "out_stats_json": str(stats_path)}, indent=2))


if __name__ == "__main__":
    main()
