#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import DatasetBundle, SplitView, build_split_view, get_primary_query_rule, load_or_create_dataset
from experiment_bridge.metrics import (
    aggregate_query_metrics,
    evaluate_candidate_generation_full_pool,
    evaluate_query_ranking,
    is_query_evaluable,
)
from experiment_bridge.models import (
    Calibrator,
    ModelConfig,
    apply_calibrator,
    encode_ligand,
    encode_pocket,
    fit_calibrator,
    pair_raw_score,
)
from experiment_bridge.utils import chunked, ensure_dir, memory_bytes_for_vectors, save_json


@dataclass
class CandidateCache:
    ids: List[str]
    id_to_index: Dict[str, int]
    reps: np.ndarray
    cache_path: str
    cache_hit: bool
    diag_summary: Optional[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment-bridge retrieval experiments.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="litpcba")
    parser.add_argument(
        "--split",
        choices=["standard", "scaffold", "target", "target_rec_cluster", "joint_ood"],
        default="standard",
    )
    parser.add_argument("--models", default="B0,B1")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--num-pockets", type=int, default=60)
    parser.add_argument("--num-ligands", type=int, default=800)
    parser.add_argument("--pocket-tokens", type=int, default=16)
    parser.add_argument("--ligand-tokens", type=int, default=18)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--positive-rate", type=float, default=0.08)
    parser.add_argument("--true-tau", type=float, default=0.22)

    parser.add_argument("--tau-match", type=float, default=0.2)
    parser.add_argument("--tau-assign", type=float, default=0.7)
    parser.add_argument("--attn-scale", type=float, default=1.5)
    parser.add_argument("--k-slots", type=int, default=4)
    parser.add_argument("--t-slots", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-pairs", type=int, default=60000)

    parser.add_argument("--topk", default="10,50,100")
    parser.add_argument("--ef-percents", default="1,5")
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=16384)
    parser.add_argument("--cache-dir", default="results/encoding_cache")
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--stream-eval", action="store_true")
    parser.add_argument("--min-known-pos-override", type=int, default=-1)
    parser.add_argument("--min-known-neg-override", type=int, default=-1)

    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--save-rankings", action="store_true")
    parser.add_argument("--output-dir", default="results")

    return parser.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _subset_train_pairs(split: SplitView, max_train_pairs: int, seed: int) -> List[Tuple[str, str, int]]:
    pairs = split.train_pairs[:]
    if len(pairs) <= max_train_pairs:
        return pairs
    rng = random.Random(seed + 1234)
    rng.shuffle(pairs)
    return pairs[:max_train_pairs]


def _write_training_curve(path: Path, model_name: str, losses: Sequence[float]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_logloss"])
        for i, loss in enumerate(losses, start=1):
            writer.writerow([model_name, i, f"{loss:.8f}"])


def _write_rankings_csv(path: Path, ranked: Dict[str, List[Tuple[str, float]]], labels: Dict[str, Dict[str, int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pocket_id", "rank", "ligand_id", "score", "label"])
        for pid, items in ranked.items():
            for rank, (lid, score) in enumerate(items, start=1):
                writer.writerow([pid, rank, lid, f"{score:.8f}", labels.get(pid, {}).get(lid, -1)])


def _write_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    ensure_dir(path.parent)
    metric_names = sorted({k for row in rows for k in row if k not in {"model"}})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *metric_names])
        for row in rows:
            writer.writerow([row["model"], *[f"{row.get(metric, float('nan')):.8f}" for metric in metric_names]])


def _repr_to_numpy(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _cache_key(dataset_path: Path, model_name: str, cfg: ModelConfig, bundle: DatasetBundle) -> str:
    payload = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_name": bundle.name,
        "model": model_name,
        "dim": bundle.dim,
        "num_ligands": len(bundle.ligands),
        "cfg": {
            "seed": cfg.seed,
            "tau_match": cfg.tau_match,
            "tau_assign": cfg.tau_assign,
            "attn_scale": cfg.attn_scale,
            "k_slots": cfg.k_slots,
            "t_slots": cfg.t_slots,
        },
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_diag_average(rows: Sequence[Dict[str, float]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(r.get(key, 0.0) for r in rows) / len(rows)


def _load_or_build_candidate_cache(
    cache_dir: Path,
    dataset_path: Path,
    bundle: DatasetBundle,
    model_name: str,
    cfg: ModelConfig,
    force_reencode: bool,
) -> CandidateCache:
    ensure_dir(cache_dir)
    key = _cache_key(dataset_path, model_name, cfg, bundle)
    meta_path = cache_dir / f"{key}.json"
    reps_path = cache_dir / f"{key}.npy"

    if meta_path.exists() and reps_path.exists() and not force_reencode:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        reps = np.load(reps_path, mmap_mode="r")
        return CandidateCache(
            ids=list(meta["ids"]),
            id_to_index={lid: idx for idx, lid in enumerate(meta["ids"])},
            reps=reps,
            cache_path=str(reps_path),
            cache_hit=True,
            diag_summary=meta.get("diag_summary"),
        )

    first_repr, _ = encode_ligand(model_name, bundle.ligands[0], cfg)
    first_arr = _repr_to_numpy(first_repr)
    reps = np.lib.format.open_memmap(
        reps_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(bundle.ligands), first_arr.shape[0], first_arr.shape[1]),
    )

    ids: List[str] = []
    diag_rows: List[Dict[str, float]] = []
    for idx, ligand in enumerate(bundle.ligands):
        enc, diag = encode_ligand(model_name, ligand, cfg)
        reps[idx] = _repr_to_numpy(enc)
        ids.append(ligand.lid)
        if diag:
            diag_rows.append(diag)
    reps.flush()

    diag_summary = None
    if diag_rows:
        diag_summary = {
            "assignment_entropy": _cache_diag_average(diag_rows, "assignment_entropy"),
            "slot_balance": _cache_diag_average(diag_rows, "slot_balance"),
            "inter_slot_cosine": _cache_diag_average(diag_rows, "inter_slot_cosine"),
            "inter_slot_cosine_std": _cache_diag_average(diag_rows, "inter_slot_cosine_std"),
        }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ids": ids,
                "shape": list(reps.shape),
                "dtype": str(reps.dtype),
                "diag_summary": diag_summary,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    loaded = np.load(reps_path, mmap_mode="r")
    return CandidateCache(
        ids=ids,
        id_to_index={lid: idx for idx, lid in enumerate(ids)},
        reps=loaded,
        cache_path=str(reps_path),
        cache_hit=False,
        diag_summary=diag_summary,
    )


def _logsumexp_numpy(values: np.ndarray, axis: int, temperature: float) -> np.ndarray:
    t = max(1e-8, float(temperature))
    scaled = values / t
    m = np.max(scaled, axis=axis, keepdims=True)
    total = np.sum(np.exp(scaled - m), axis=axis)
    return (t * (np.log(np.maximum(total, 1e-12)) + np.squeeze(m, axis=axis))).astype(np.float32)


def _score_chunk_numpy(model_name: str, pocket_repr: np.ndarray, ligand_chunk: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    if model_name in {"B0", "B1"}:
        p = pocket_repr[0].astype(np.float32)
        lig = ligand_chunk[:, 0, :].astype(np.float32)
        p_norm = np.linalg.norm(p) + 1e-12
        lig_norm = np.linalg.norm(lig, axis=1) + 1e-12
        return (lig @ p) / (lig_norm * p_norm)

    if model_name == "B3_SLOT_COS":
        p = pocket_repr.mean(axis=0).astype(np.float32)
        lig = ligand_chunk.mean(axis=1).astype(np.float32)
        p_norm = np.linalg.norm(p) + 1e-12
        lig_norm = np.linalg.norm(lig, axis=1) + 1e-12
        return (lig @ p) / (lig_norm * p_norm)

    if model_name in {"U1", "M1"}:
        sims = np.einsum("kd,ctd->ckt", pocket_repr.astype(np.float32), ligand_chunk.astype(np.float32), optimize=True)
        return _logsumexp_numpy(sims, axis=2, temperature=cfg.tau_match).mean(axis=1)

    if model_name in {"B2_COLBERT", "B4_SLOT_MAXSIM"}:
        sims = np.einsum("kd,ctd->ckt", pocket_repr.astype(np.float32), ligand_chunk.astype(np.float32), optimize=True)
        return sims.max(axis=2).mean(axis=1).astype(np.float32)

    raise ValueError(f"Unsupported vectorized scoring path for model: {model_name}")


def _append_rankings_csv(
    path: Path,
    pocket_id: str,
    ranked_ids: Sequence[str],
    ranked_scores: Sequence[float],
    label_row: Dict[str, int],
    write_header: bool,
) -> None:
    ensure_dir(path.parent)
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["pocket_id", "rank", "ligand_id", "score", "label"])
        for rank, (lid, score) in enumerate(zip(ranked_ids, ranked_scores), start=1):
            writer.writerow([pocket_id, rank, lid, f"{float(score):.8f}", label_row.get(lid, -1)])


def main() -> None:
    args = parse_args()

    if args.sanity:
        args.num_pockets = min(args.num_pockets, 24)
        args.num_ligands = min(args.num_ligands, 220)
        args.epochs = min(args.epochs, 8)
        args.max_train_pairs = min(args.max_train_pairs, 10000)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    topk_values = _parse_int_list(args.topk)
    ef_percents = _parse_float_list(args.ef_percents)

    run_dir = Path(args.output_dir) / args.run_id
    ensure_dir(run_dir)

    dataset_path = Path(args.dataset_path) if args.dataset_path else Path("data") / f"{args.dataset}_synthetic_seed{args.seed}.json"
    bundle: DatasetBundle = load_or_create_dataset(
        path=dataset_path,
        dataset_name=args.dataset,
        seed=args.seed,
        dim=args.dim,
        num_pockets=args.num_pockets,
        num_ligands=args.num_ligands,
        pocket_tokens=args.pocket_tokens,
        ligand_tokens=args.ligand_tokens,
        positive_rate=args.positive_rate,
        true_tau=args.true_tau,
    )
    split = build_split_view(bundle, split_name=args.split, seed=args.seed)
    query_rule = get_primary_query_rule(bundle)
    min_known_pos = args.min_known_pos_override if args.min_known_pos_override >= 0 else int(query_rule["min_known_pos"])
    min_known_neg = args.min_known_neg_override if args.min_known_neg_override >= 0 else int(query_rule["min_known_neg"])
    evaluation_mode = "candidate_generation_retrieval" if int(min_known_neg) == 0 else "discriminative_retrieval"

    pocket_lookup = {p.pid: p for p in bundle.pockets}
    ligand_lookup = {l.lid: l for l in bundle.ligands}

    train_pairs = _subset_train_pairs(split, args.max_train_pairs, seed=args.seed)
    eval_pocket_ids = split.eval_pocket_ids
    candidate_lids = split.candidate_ligand_ids

    cfg = ModelConfig(
        seed=args.seed,
        tau_match=args.tau_match,
        tau_assign=args.tau_assign,
        attn_scale=args.attn_scale,
        k_slots=args.k_slots,
        t_slots=args.t_slots,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    summary = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "split_notes": split.notes,
        "models": model_names,
        "seed": args.seed,
        "num_train_pairs": len(train_pairs),
        "num_eval_pockets": len(eval_pocket_ids),
        "num_candidate_ligands": len(candidate_lids),
        "sanity": args.sanity,
        "stream_eval": args.stream_eval or len(candidate_lids) > 50000,
        "score_batch_size": args.score_batch_size,
        "cache_dir": args.cache_dir,
        "evaluation_mode": evaluation_mode,
        "metric_semantics": "full_candidate_pool" if evaluation_mode == "candidate_generation_retrieval" else "known_labeled_subset",
        "query_rule_used": {"min_known_pos": min_known_pos, "min_known_neg": min_known_neg},
        "results": {},
    }

    metric_rows: List[Dict[str, float]] = []
    use_stream_eval = args.stream_eval or len(candidate_lids) > 50000

    for model_name in model_names:
        model_start = time.perf_counter()

        # Encode candidates once and cache them across runs.
        t0 = time.perf_counter()
        candidate_cache = _load_or_build_candidate_cache(
            cache_dir=Path(args.cache_dir),
            dataset_path=dataset_path,
            bundle=bundle,
            model_name=model_name,
            cfg=cfg,
            force_reencode=args.force_reencode,
        )
        candidate_indices = np.asarray([candidate_cache.id_to_index[lid] for lid in candidate_lids], dtype=np.int64)
        ligand_encode_sec = time.perf_counter() - t0

        # Encode pockets for train/eval.
        t0 = time.perf_counter()
        pocket_cache: Dict[str, np.ndarray] = {}
        pocket_diag_map: Dict[str, Dict[str, float]] = {}
        pocket_diag: List[Dict[str, float]] = []
        for pid in set([pid for pid, _, _ in train_pairs] + eval_pocket_ids):
            enc, diag = encode_pocket(model_name, pocket_lookup[pid], cfg)
            pocket_cache[pid] = _repr_to_numpy(enc)
            if diag:
                pocket_diag.append(diag)
                pocket_diag_map[pid] = diag
        pocket_encode_sec = time.perf_counter() - t0

        raw_train: List[float] = []
        y_train: List[int] = []
        for pid, lid, label in train_pairs:
            lid_idx = candidate_cache.id_to_index[lid]
            ligand_repr_np = np.asarray(candidate_cache.reps[lid_idx])
            if model_name in {"B0", "B1", "U1", "M1", "B2_COLBERT", "B3_SLOT_COS", "B4_SLOT_MAXSIM"}:
                raw = float(_score_chunk_numpy(model_name, pocket_cache[pid], ligand_repr_np[None, ...], cfg)[0])
            else:
                raw = pair_raw_score(
                    model_name,
                    pocket_cache[pid].tolist(),
                    ligand_repr_np.tolist(),
                    cfg,
                    pocket_diag=pocket_diag_map.get(pid),
                    ligand_diag=None,
                )
            raw_train.append(raw)
            y_train.append(label)

        calibrator: Calibrator = fit_calibrator(raw_train, y_train, cfg)
        _write_training_curve(run_dir / f"{model_name}_training.csv", model_name, calibrator.train_loss)

        ranking_path = run_dir / f"{model_name}_rankings.csv"
        if ranking_path.exists() and args.save_rankings:
            ranking_path.unlink()
        query_metrics: List[Dict[str, float]] = []
        raw_scoring_sec = 0.0
        scoring_sec = 0.0
        ranking_csv_header_written = False

        t_score = time.perf_counter()
        for pid in eval_pocket_ids:
            pocket_repr = pocket_cache[pid]
            raw_scores = np.empty(len(candidate_indices), dtype=np.float32)
            offset = 0

            pocket_score_start = time.perf_counter()
            for idx_chunk in chunked(candidate_indices, args.score_batch_size):
                chunk_arr = np.asarray(candidate_cache.reps[idx_chunk])
                if model_name in {"B0", "B1", "U1", "M1", "B2_COLBERT", "B3_SLOT_COS", "B4_SLOT_MAXSIM"}:
                    chunk_scores = _score_chunk_numpy(model_name, pocket_repr, chunk_arr, cfg)
                else:
                    chunk_scores = np.asarray(
                        [
                            pair_raw_score(
                                model_name,
                                pocket_repr.tolist(),
                                np.asarray(candidate_cache.reps[idx]).tolist(),
                                cfg,
                                pocket_diag=pocket_diag_map.get(pid),
                                ligand_diag=None,
                            )
                            for idx in idx_chunk
                        ],
                        dtype=np.float32,
                    )
                raw_scores[offset : offset + len(idx_chunk)] = chunk_scores
                offset += len(idx_chunk)
            raw_scoring_sec += time.perf_counter() - pocket_score_start

            order = np.argsort(-raw_scores, kind="stable")
            ranked_ids = [candidate_lids[idx] for idx in order]
            ranked_scores = calibrator.alpha * raw_scores[order] + calibrator.beta

            if args.save_rankings:
                _append_rankings_csv(
                    ranking_path,
                    pocket_id=pid,
                    ranked_ids=ranked_ids,
                    ranked_scores=ranked_scores.tolist(),
                    label_row=bundle.labels[pid],
                    write_header=not ranking_csv_header_written,
                )
                ranking_csv_header_written = True

            if evaluation_mode == "candidate_generation_retrieval":
                metrics_for_query = evaluate_candidate_generation_full_pool(
                    ranked_ids=ranked_ids,
                    label_row=bundle.labels[pid],
                    topk_values=topk_values,
                    ef_percents=ef_percents,
                    min_known_pos=min_known_pos,
                )
            else:
                metrics_for_query = evaluate_query_ranking(
                    ranked_ids=ranked_ids,
                    label_row=bundle.labels[pid],
                    topk_values=topk_values,
                    ef_percents=ef_percents,
                    min_known_pos=min_known_pos,
                    min_known_neg=min_known_neg,
                )
            if metrics_for_query:
                query_metrics.append(metrics_for_query)

            # extra repeats for stress profiling without resorting or metric recompute.
            if args.profile_repeats > 1:
                for _ in range(args.profile_repeats - 1):
                    for idx_chunk in chunked(candidate_indices, args.score_batch_size):
                        chunk_arr = np.asarray(candidate_cache.reps[idx_chunk])
                        if model_name in {"B0", "B1", "U1", "M1", "B2_COLBERT", "B3_SLOT_COS", "B4_SLOT_MAXSIM"}:
                            _ = _score_chunk_numpy(model_name, pocket_repr, chunk_arr, cfg)
                        else:
                            for idx in idx_chunk:
                                _ = pair_raw_score(
                                    model_name,
                                    pocket_repr.tolist(),
                                    np.asarray(candidate_cache.reps[idx]).tolist(),
                                    cfg,
                                    pocket_diag=pocket_diag_map.get(pid),
                                    ligand_diag=None,
                                )

        scoring_sec = time.perf_counter() - t_score
        if not raw_scoring_sec:
            raw_scoring_sec = scoring_sec

        metrics = aggregate_query_metrics(query_metrics, topk_values, ef_percents)
        num_evaluable_queries = sum(
            1
            for pid in eval_pocket_ids
            if is_query_evaluable(
                candidate_lids,
                bundle.labels[pid],
                min_known_pos=min_known_pos,
                min_known_neg=min_known_neg,
            )
        )
        metrics["num_total_eval_pockets"] = float(len(eval_pocket_ids))
        metrics["num_evaluable_pockets"] = float(num_evaluable_queries)
        metrics["evaluable_query_coverage"] = num_evaluable_queries / max(1, len(eval_pocket_ids))

        total_pairs = len(eval_pocket_ids) * len(candidate_lids)
        qps = (total_pairs * max(1, args.profile_repeats)) / max(1e-12, raw_scoring_sec)
        full_index_bytes = int(candidate_cache.reps.nbytes)
        index_bytes = int(full_index_bytes * (len(candidate_lids) / max(1, len(candidate_cache.ids))))

        model_result = {
            "model": model_name,
            "metrics": metrics,
            "calibrator": {
                "alpha": calibrator.alpha,
                "beta": calibrator.beta,
                "final_train_logloss": calibrator.train_loss[-1] if calibrator.train_loss else None,
            },
            "efficiency": {
                "pocket_encode_sec": pocket_encode_sec,
                "ligand_encode_sec": ligand_encode_sec,
                "scoring_sec": scoring_sec,
                "raw_scoring_sec": raw_scoring_sec,
                "scoring_qps": qps,
                "index_memory_mb": index_bytes / (1024.0 * 1024.0),
            },
            "cache": {
                "candidate_cache_path": candidate_cache.cache_path,
                "candidate_cache_hit": candidate_cache.cache_hit,
                "cached_num_ligands": len(candidate_cache.ids),
            },
            "runtime_sec": time.perf_counter() - model_start,
        }

        if model_name in {"M1", "M2_ADAPTIVE"}:
            def _avg(diags: Sequence[Dict[str, float]], key: str) -> float:
                if not diags:
                    return float("nan")
                return sum(d.get(key, 0.0) for d in diags) / len(diags)

            model_result["slot_diagnostics"] = {
                "pocket_assignment_entropy": _avg(pocket_diag, "assignment_entropy"),
                "pocket_slot_balance": _avg(pocket_diag, "slot_balance"),
                "pocket_inter_slot_cosine": _avg(pocket_diag, "inter_slot_cosine"),
                "pocket_inter_slot_cosine_std": _avg(pocket_diag, "inter_slot_cosine_std"),
                "ligand_assignment_entropy": (
                    candidate_cache.diag_summary.get("assignment_entropy", float("nan"))
                    if candidate_cache.diag_summary
                    else float("nan")
                ),
                "ligand_slot_balance": (
                    candidate_cache.diag_summary.get("slot_balance", float("nan"))
                    if candidate_cache.diag_summary
                    else float("nan")
                ),
                "ligand_inter_slot_cosine": (
                    candidate_cache.diag_summary.get("inter_slot_cosine", float("nan"))
                    if candidate_cache.diag_summary
                    else float("nan")
                ),
                "ligand_inter_slot_cosine_std": (
                    candidate_cache.diag_summary.get("inter_slot_cosine_std", float("nan"))
                    if candidate_cache.diag_summary
                    else float("nan")
                ),
            }

        save_json(run_dir / f"{model_name}.json", model_result)
        summary["results"][model_name] = model_result

        metric_row = {"model": model_name}
        metric_row.update(metrics)
        metric_row["scoring_qps"] = qps
        metric_row["index_memory_mb"] = model_result["efficiency"]["index_memory_mb"]
        metric_rows.append(metric_row)

    _write_metrics_csv(run_dir / "metrics.csv", metric_rows)
    save_json(run_dir / "summary.json", summary)

    print(json.dumps({
        "run_id": args.run_id,
        "models": model_names,
        "dataset": str(dataset_path),
        "split": args.split,
        "num_train_pairs": len(train_pairs),
        "num_eval_pockets": len(eval_pocket_ids),
        "num_candidate_ligands": len(candidate_lids),
        "output": str(run_dir / "summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
