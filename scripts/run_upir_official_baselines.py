#!/usr/bin/env python3
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import build_split_view, get_primary_query_rule, load_dataset
from experiment_bridge.models import ModelConfig, encode_ligand, encode_pocket
from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official lightweight baselines on UPIR fixed protocol.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_strict_forward.json")
    p.add_argument("--splits", default="standard,target,scaffold,joint_ood")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", default="RANDOM,POPULARITY,B0_FAST,B1_FAST")
    p.add_argument("--out-dir", default="results/upir_official_baselines_v1")
    p.add_argument("--topk", default="10,50")
    p.add_argument("--ef-percents", default="1,5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=400)
    p.add_argument("--min-known-pos-override", type=int, default=-1)
    p.add_argument("--min-known-neg-override", type=int, default=-1)
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _top_k_size(total: int, percent: float) -> int:
    return max(1, int(round(total * percent / 100.0)))


def _rank_metrics(
    ranking_ids: Sequence[str],
    label_row: Dict[str, int],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
    min_known_pos: int,
    min_known_neg: int,
) -> Dict[str, float]:
    ranked_ids = [lid for lid in ranking_ids if lid in label_row]
    positives = [lid for lid in ranked_ids if int(label_row.get(lid, 0)) == 1]
    num_pos = len(positives)
    num_neg = len(ranked_ids) - num_pos
    if num_pos < int(min_known_pos) or num_neg < int(min_known_neg):
        return {}

    pos_ranks = [idx + 1 for idx, lid in enumerate(ranked_ids) if int(label_row.get(lid, 0)) == 1]
    first_pos_rank = min(pos_ranks)
    out: Dict[str, float] = {
        "mrr": 1.0 / first_pos_rank,
        "num_pos": float(num_pos),
        "num_known_pairs": float(len(ranked_ids)),
    }

    for k in topk_values:
        top_ids = ranked_ids[:k]
        tp = sum(1 for lid in top_ids if int(label_row.get(lid, 0)) == 1)
        out[f"recall@{k}"] = tp / num_pos
        out[f"hit@{k}"] = 1.0 if tp > 0 else 0.0

    baseline_rate = num_pos / max(1, len(ranked_ids))
    for p in ef_percents:
        top_n = _top_k_size(len(ranked_ids), p)
        top_ids = ranked_ids[:top_n]
        actives = sum(1 for lid in top_ids if int(label_row.get(lid, 0)) == 1)
        top_rate = actives / max(1, top_n)
        out[f"ef{int(p)}"] = top_rate / max(1e-12, baseline_rate)
    return out


def _aggregate_query_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    metric_keys = sorted({k for row in rows for k in row.keys() if k not in {"pid"}})
    out: Dict[str, float] = {"num_eval_queries": float(len(rows))}
    for key in metric_keys:
        out[key] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    return out


def _average_numeric_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def _bootstrap_ci(rows: Sequence[Dict[str, float]], key: str, n_boot: int = 400) -> Tuple[float, float]:
    if not rows:
        return (float("nan"), float("nan"))
    if int(n_boot) <= 0:
        mean_val = float(sum(float(r.get(key, 0.0)) for r in rows) / max(1, len(rows)))
        return (mean_val, mean_val)
    vals = np.array([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(12345 + stable_seed(key))
    means = np.empty(n_boot, dtype=np.float64)
    n = vals.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(vals[idx].mean())
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return rows / norms


def _hash_text_vector(text: str, ngram: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    s = (text or "").strip()
    if not s:
        return vec
    if len(s) <= ngram:
        grams = [s]
    else:
        grams = [s[i : i + ngram] for i in range(len(s) - ngram + 1)]
    for gram in grams:
        h = stable_seed(f"h:{gram}")
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _protein_feature_matrix(bundle, dim: int = 128) -> Tuple[List[str], np.ndarray]:
    ids = [p.pid for p in bundle.pockets]
    mat = np.stack([_hash_text_vector(p.text or "", ngram=2, dim=dim) for p in bundle.pockets], axis=0)
    return ids, mat


def _ligand_feature_matrix(bundle, candidate_ids: Sequence[str], dim: int = 128) -> Tuple[List[str], np.ndarray]:
    ligand_lookup = {l.lid: l for l in bundle.ligands}
    kept = list(candidate_ids)
    mat = np.stack([_hash_text_vector(ligand_lookup[lid].text or "", ngram=3, dim=dim) for lid in kept], axis=0)
    return kept, mat


def _encode_single_vector_matrix(bundle, candidate_ids: Sequence[str], model_name: str, cfg: ModelConfig) -> Tuple[List[str], np.ndarray, float]:
    ligand_lookup = {l.lid: l for l in bundle.ligands}
    rows: List[np.ndarray] = []
    start = time.perf_counter()
    kept_ids: List[str] = []
    for lid in candidate_ids:
        enc, _ = encode_ligand("B0" if model_name == "B0_FAST" else "B1", ligand_lookup[lid], cfg)
        if not enc:
            continue
        kept_ids.append(lid)
        rows.append(np.asarray(enc[0], dtype=np.float32))
    matrix = np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)
    matrix = _normalize_rows(matrix)
    return kept_ids, matrix, time.perf_counter() - start


def _encode_query_vector(bundle, pid: str, model_name: str, cfg: ModelConfig) -> np.ndarray:
    pocket_lookup = {p.pid: p for p in bundle.pockets}
    enc, _ = encode_pocket("B0" if model_name == "B0_FAST" else "B1", pocket_lookup[pid], cfg)
    vec = np.asarray(enc[0], dtype=np.float32)
    denom = max(1e-12, float(np.linalg.norm(vec)))
    return vec / denom


def _stable_random_order(candidate_ids: Sequence[str], seed: int) -> List[str]:
    return sorted(candidate_ids, key=lambda lid: stable_seed(f"random:{seed}:{lid}"))


def _popularity_order(train_pairs: Sequence[Tuple[str, str, int]], candidate_ids: Sequence[str]) -> List[str]:
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


def _filter_ranking(full_ranking: Sequence[str], allowed: set[str]) -> List[str]:
    return [lid for lid in full_ranking if lid in allowed]


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    topk_values = _parse_ints(args.topk)
    ef_percents = _parse_floats(args.ef_percents)

    bundle = load_dataset(dataset_path)
    query_rule = get_primary_query_rule(bundle)
    min_known_pos = args.min_known_pos_override if args.min_known_pos_override >= 0 else int(query_rule["min_known_pos"])
    min_known_neg = args.min_known_neg_override if args.min_known_neg_override >= 0 else int(query_rule["min_known_neg"])
    evaluation_mode = "discriminative_retrieval" if int(min_known_neg) >= 1 else "candidate_generation_retrieval"
    all_candidate_ids = [l.lid for l in bundle.ligands]
    all_query_ids = [p.pid for p in bundle.pockets]
    cfg = ModelConfig(seed=args.seed)
    candidate_index = {lid: i for i, lid in enumerate(all_candidate_ids)}

    full_rankings: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    efficiency: Dict[str, Dict[str, float]] = {}
    protein_ids, protein_mat = _protein_feature_matrix(bundle, dim=128)
    protein_index = {pid: i for i, pid in enumerate(protein_ids)}
    ligand_hash_ids: List[str] = []
    ligand_hash_mat: np.ndarray | None = None

    if any(m in {"PROT_CHEM_CENTROID", "PROT_KNN_POP"} for m in model_names):
        if "PROT_CHEM_CENTROID" in model_names:
            ligand_hash_ids, ligand_hash_mat = _ligand_feature_matrix(bundle, all_candidate_ids, dim=128)
        else:
            ligand_hash_ids = list(all_candidate_ids)

    for model_name in model_names:
        if model_name == "RANDOM":
            start = time.perf_counter()
            order = _stable_random_order(all_candidate_ids, seed=args.seed)
            for pid in all_query_ids:
                full_rankings[model_name][pid] = order
            efficiency[model_name] = {"prep_sec": time.perf_counter() - start, "scoring_qps": 0.0}
            continue

        if model_name == "POPULARITY":
            # Built per fold later because it depends on train pairs.
            efficiency[model_name] = {"prep_sec": 0.0, "scoring_qps": 0.0}
            continue

        if model_name in {"PROT_KNN_POP", "PROT_CHEM_CENTROID"}:
            efficiency[model_name] = {"prep_sec": 0.0, "scoring_qps": 0.0}
            continue

        kept_ids, cand_matrix, encode_sec = _encode_single_vector_matrix(bundle, all_candidate_ids, model_name, cfg)
        rank_start = time.perf_counter()
        for pid in all_query_ids:
            q = _encode_query_vector(bundle, pid, model_name, cfg)
            scores = cand_matrix @ q
            order_idx = np.argsort(-scores, kind="stable")
            full_rankings[model_name][pid] = [kept_ids[int(i)] for i in order_idx]
        rank_sec = time.perf_counter() - rank_start
        total_pairs = len(all_query_ids) * len(kept_ids)
        efficiency[model_name] = {
            "prep_sec": encode_sec,
            "ranking_sec": rank_sec,
            "scoring_qps": total_pairs / max(1e-12, rank_sec),
            "index_memory_mb": float(cand_matrix.nbytes) / (1024.0 * 1024.0),
        }

    results: Dict[str, Dict] = {}
    aggregate_rows: List[Dict[str, object]] = []

    for split_name in splits:
        split_result: Dict[str, object] = {"folds": [], "models": {}}
        per_model_all_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for fold in range(args.folds):
            split = build_split_view(bundle, split_name=split_name, seed=fold)
            allowed_candidates = set(split.candidate_ligand_ids)
            total_eval_queries = len(split.eval_pocket_ids)
            fold_entry = {
                "fold": fold,
                "notes": split.notes,
                "num_eval_queries": len(split.eval_pocket_ids),
                "num_eval_candidates": len(split.candidate_ligand_ids),
                "models": {},
            }

            popularity_rank = None
            if "POPULARITY" in model_names:
                popularity_rank = _popularity_order(split.train_pairs, split.candidate_ligand_ids)

            train_query_ids = sorted({pid for pid, _, _ in split.train_pairs})
            train_qidx = [protein_index[pid] for pid in train_query_ids]
            train_row_lookup = {pid: [] for pid in train_query_ids}
            train_pos_lookup = {pid: [] for pid in train_query_ids}
            for pid, lid, label in split.train_pairs:
                train_row_lookup[pid].append((lid, int(label)))
                if int(label) == 1:
                    train_pos_lookup[pid].append(lid)

            for model_name in model_names:
                query_rows: List[Dict[str, float]] = []
                for pid in split.eval_pocket_ids:
                    if model_name == "POPULARITY":
                        ranking_ids = popularity_rank
                    elif model_name == "PROT_KNN_POP":
                        qvec = protein_mat[protein_index[pid]]
                        sims = protein_mat[train_qidx] @ qvec
                        sims = np.maximum(sims, 0.0)
                        if float(sims.sum()) <= 0.0:
                            sims = np.ones_like(sims)
                        score_map = defaultdict(float)
                        for idx, sim in enumerate(sims.tolist()):
                            if sim <= 0.0:
                                continue
                            tpid = train_query_ids[idx]
                            for lid, label in train_row_lookup[tpid]:
                                if label == 1 and lid in allowed_candidates:
                                    score_map[lid] += float(sim)
                        scored_ids = sorted(score_map.keys(), key=lambda lid: (-score_map[lid], stable_seed(lid)))
                        scored_set = set(scored_ids)
                        remainder = [lid for lid in split.candidate_ligand_ids if lid not in scored_set]
                        ranking_ids = scored_ids + remainder
                    elif model_name == "PROT_CHEM_CENTROID":
                        qvec = protein_mat[protein_index[pid]]
                        sims = protein_mat[train_qidx] @ qvec
                        sims = np.maximum(sims, 0.0)
                        if float(sims.sum()) <= 0.0:
                            sims = np.ones_like(sims)
                        ligand_weights = defaultdict(float)
                        for idx, sim in enumerate(sims.tolist()):
                            if sim <= 0.0:
                                continue
                            tpid = train_query_ids[idx]
                            for lid in train_pos_lookup[tpid]:
                                if lid in allowed_candidates:
                                    ligand_weights[lid] += float(sim)
                        if not ligand_weights:
                            ranking_ids = _stable_random_order(split.candidate_ligand_ids, seed=args.seed + stable_seed(pid))
                        else:
                            centroid = np.zeros(ligand_hash_mat.shape[1], dtype=np.float32)
                            for lid, weight in ligand_weights.items():
                                cidx = candidate_index.get(lid)
                                if cidx is None:
                                    continue
                                centroid += float(weight) * ligand_hash_mat[cidx]
                            norm = float(np.linalg.norm(centroid))
                            if norm > 0.0:
                                centroid /= norm
                            score_vec = ligand_hash_mat @ centroid
                            order_idx = np.argsort(-score_vec, kind="stable")
                            ranking_ids = [ligand_hash_ids[int(i)] for i in order_idx if ligand_hash_ids[int(i)] in allowed_candidates]
                    else:
                        ranking_ids = _filter_ranking(full_rankings[model_name][pid], allowed_candidates)
                    m = _rank_metrics(
                        ranking_ids,
                        bundle.labels.get(pid, {}),
                        topk_values,
                        ef_percents,
                        min_known_pos=min_known_pos,
                        min_known_neg=min_known_neg,
                    )
                    if not m:
                        continue
                    m["pid"] = pid
                    query_rows.append(m)

                agg = _aggregate_query_metrics(query_rows)
                agg["num_total_eval_queries"] = float(total_eval_queries)
                agg["num_evaluable_queries"] = float(len(query_rows))
                agg["evaluable_query_coverage"] = len(query_rows) / max(1, total_eval_queries)
                fold_entry["models"][model_name] = {
                    "aggregate": agg,
                    "query_rows": query_rows,
                }
                per_model_all_rows[model_name].extend(query_rows)
            split_result["folds"].append(fold_entry)

        summary_by_model: Dict[str, Dict[str, object]] = {}
        for model_name in model_names:
            rows = per_model_all_rows[model_name]
            agg = _aggregate_query_metrics(rows)
            total_eval_queries_all = sum(fold_entry["num_eval_queries"] for fold_entry in split_result["folds"])
            agg["num_total_eval_queries"] = float(total_eval_queries_all)
            agg["num_evaluable_queries"] = float(len(rows))
            agg["evaluable_query_coverage"] = len(rows) / max(1, total_eval_queries_all)
            fold_aggregates = [
                dict(fold_entry["models"][model_name]["aggregate"])
                for fold_entry in split_result["folds"]
                if model_name in fold_entry["models"] and fold_entry["models"][model_name]["aggregate"]
            ]
            agg_fold_mean = _average_numeric_dicts(fold_aggregates)
            ci = {}
            for metric in ["mrr"] + [f"recall@{k}" for k in topk_values] + [f"ef{int(p)}" for p in ef_percents]:
                if metric in agg:
                    lo, hi = _bootstrap_ci(rows, metric, n_boot=args.n_bootstrap)
                    ci[metric] = {"lo": lo, "hi": hi}
            summary_by_model[model_name] = {
                "aggregate_over_folds": agg_fold_mean,
                "aggregate_over_query_instances": agg,
                "bootstrap_ci": ci,
                "efficiency": efficiency.get(model_name, {}),
            }
            aggregate_rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "aggregate_over_folds": agg_fold_mean,
                    "aggregate": agg,
                    "ci": ci,
                    "efficiency": efficiency.get(model_name, {}),
                }
            )
        split_result["models"] = summary_by_model
        results[split_name] = split_result

    payload = {
        "dataset_path": str(dataset_path),
        "dataset_name": bundle.name,
        "evaluation_mode": evaluation_mode,
        "query_rule_used": {"min_known_pos": min_known_pos, "min_known_neg": min_known_neg},
        "models": model_names,
        "splits": splits,
        "folds": args.folds,
        "results": results,
    }
    save_json(out_dir / "summary.json", payload)

    table_lines = [
        "# UPIR Official Baselines",
        "",
        f"- dataset: `{bundle.name}`",
        f"- evaluation mode: `{evaluation_mode}`",
        f"- models: `{', '.join(model_names)}`",
        f"- splits: `{', '.join(splits)}`",
        f"- query rule used: `>= {min_known_pos}` known positive(s), `>= {min_known_neg}` known negative(s)`",
        "",
        "## Aggregate Results",
        "",
    ]
    if evaluation_mode == "candidate_generation_retrieval":
        table_lines.extend([
            "| Split | Model | AvgEvalQ/Fold | AvgCoverage | Hit@10 | Hit@50 | Recall@10 | Recall@50 | QPS | Mem(MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
    else:
        table_lines.extend([
            "| Split | Model | AvgEvalQ/Fold | AvgCoverage | MRR | Recall@10 | Recall@50 | EF1 | EF5 | QPS | Mem(MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
    for row in aggregate_rows:
        agg = row["aggregate_over_folds"] or row["aggregate"]
        eff = row["efficiency"]
        if evaluation_mode == "candidate_generation_retrieval":
            table_lines.append(
                "| {split} | {model} | {evalq:.0f} | {cov:.3f} | {h10:.4f} | {h50:.4f} | {r10:.4f} | {r50:.4f} | {qps:.1f} | {mem:.2f} |".format(
                    split=row["split"],
                    model=row["model"],
                    evalq=agg.get("num_evaluable_queries", float("nan")),
                    cov=agg.get("evaluable_query_coverage", float("nan")),
                    h10=agg.get("hit@10", float("nan")),
                    h50=agg.get("hit@50", float("nan")),
                    r10=agg.get("recall@10", float("nan")),
                    r50=agg.get("recall@50", float("nan")),
                    qps=eff.get("scoring_qps", 0.0),
                    mem=eff.get("index_memory_mb", 0.0),
                )
            )
        else:
            table_lines.append(
                "| {split} | {model} | {evalq:.0f} | {cov:.3f} | {mrr:.4f} | {r10:.4f} | {r50:.4f} | {ef1:.4f} | {ef5:.4f} | {qps:.1f} | {mem:.2f} |".format(
                    split=row["split"],
                    model=row["model"],
                    evalq=agg.get("num_evaluable_queries", float("nan")),
                    cov=agg.get("evaluable_query_coverage", float("nan")),
                    mrr=agg.get("mrr", float("nan")),
                    r10=agg.get("recall@10", float("nan")),
                    r50=agg.get("recall@50", float("nan")),
                    ef1=agg.get("ef1", float("nan")),
                    ef5=agg.get("ef5", float("nan")),
                    qps=eff.get("scoring_qps", 0.0),
                    mem=eff.get("index_memory_mb", 0.0),
                )
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
