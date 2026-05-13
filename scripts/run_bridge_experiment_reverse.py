#!/usr/bin/env python3
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import Ligand, Pocket, build_split_view, get_primary_query_rule, load_dataset
from experiment_bridge.models import Calibrator, ModelConfig, encode_ligand, encode_pocket, fit_calibrator
from experiment_bridge.text_tokens import stable_seed
from experiment_bridge.utils import ensure_dir, save_json


@dataclass
class ReverseBundle:
    query_ligands: List[Ligand]
    candidate_proteins: List[Pocket]
    labels: Dict[str, Dict[str, int]]
    name: str
    source_path: str
    meta: Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run reverse-direction UPIR retrieval experiments.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_strict_reverse.json")
    p.add_argument("--split", choices=["standard"], default="standard")
    p.add_argument("--models", default="B1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-train-pairs", type=int, default=60000)
    p.add_argument("--topk", default="1,5,10")
    p.add_argument("--max-eval-queries", type=int, default=0)
    p.add_argument("--tau-match", type=float, default=0.2)
    p.add_argument("--tau-assign", type=float, default=0.7)
    p.add_argument("--attn-scale", type=float, default=1.5)
    p.add_argument("--k-slots", type=int, default=4)
    p.add_argument("--t-slots", type=int, default=4)
    p.add_argument("--profile-repeats", type=int, default=1)
    p.add_argument("--min-known-pos-override", type=int, default=-1)
    p.add_argument("--min-known-neg-override", type=int, default=-1)
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _as_ligand(query: Pocket) -> Ligand:
    return Ligand(
        lid=query.pid,
        scaffold="QUERY",
        tokens=query.tokens,
        text=query.text,
        token_dim=query.token_dim,
        token_kmer=query.token_kmer,
        token_max_tokens=query.token_max_tokens,
        token_base_seed=query.token_base_seed,
    )


def _as_pocket(candidate: Ligand) -> Pocket:
    return Pocket(
        pid=candidate.lid,
        tokens=candidate.tokens,
        text=candidate.text,
        token_dim=candidate.token_dim,
        token_kmer=candidate.token_kmer,
        token_max_tokens=candidate.token_max_tokens,
        token_base_seed=candidate.token_base_seed,
    )


def _reverse_metrics(
    ranked_ids: Sequence[str],
    label_row: Dict[str, int],
    topk_values: Sequence[int],
    min_known_pos: int,
    min_known_neg: int,
) -> Dict[str, float]:
    ranked_ids = [cid for cid in ranked_ids if cid in label_row]
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


def _aggregate_query_metrics(rows: Sequence[Dict[str, float]], topk_values: Sequence[int]) -> Dict[str, float]:
    if not rows:
        out = {"mrr": 0.0, "num_eval_queries": 0.0, "num_pos": 0.0, "num_known_pairs": 0.0}
        for k in topk_values:
            out[f"recall@{k}"] = 0.0
            out[f"hit@{k}"] = 0.0
        return out
    out: Dict[str, float] = {
        "mrr": _mean([row["mrr"] for row in rows]),
        "num_eval_queries": float(len(rows)),
        "num_pos": _mean([row["num_pos"] for row in rows]),
        "num_known_pairs": _mean([row["num_known_pairs"] for row in rows]),
    }
    for k in topk_values:
        out[f"recall@{k}"] = _mean([row[f"recall@{k}"] for row in rows])
        out[f"hit@{k}"] = _mean([row[f"hit@{k}"] for row in rows])
    return out


def _write_training_curve(path: Path, model_name: str, losses: Sequence[float]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_logloss"])
        for i, loss in enumerate(losses, start=1):
            writer.writerow([model_name, i, f"{loss:.8f}"])


def _write_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    ensure_dir(path.parent)
    metric_names = sorted({k for row in rows for k in row if k not in {"model"}})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *metric_names])
        for row in rows:
            writer.writerow([row["model"], *[f"{row.get(metric, float('nan')):.8f}" for metric in metric_names]])


def _subset_train_pairs(split_pairs: Sequence[Tuple[str, str, int]], max_train_pairs: int, seed: int) -> List[Tuple[str, str, int]]:
    pairs = list(split_pairs)
    if len(pairs) <= max_train_pairs:
        return pairs
    ordered = sorted(pairs, key=lambda x: stable_seed(f"revtrain:{seed}:{x[0]}:{x[1]}:{x[2]}"))
    return ordered[:max_train_pairs]


def _subset_eval_queries(query_ids: Sequence[str], max_eval_queries: int, seed: int) -> List[str]:
    if max_eval_queries <= 0 or len(query_ids) <= max_eval_queries:
        return list(query_ids)
    ordered = sorted(query_ids, key=lambda qid: stable_seed(f"reveval:{seed}:{qid}"))
    return ordered[:max_eval_queries]


def _reverse_raw_score(model_name: str, query_repr: Sequence[Sequence[float]], candidate_repr: Sequence[Sequence[float]]) -> float:
    q = np.asarray(query_repr, dtype=np.float32)
    c = np.asarray(candidate_repr, dtype=np.float32)
    if model_name in {"B0", "B1"}:
        qv = q[0]
        cv = c[0]
        return float((qv @ cv) / ((np.linalg.norm(qv) + 1e-12) * (np.linalg.norm(cv) + 1e-12)))
    sims = q @ c.T
    if model_name in {"U1", "M1"}:
        tau = 0.2
        scaled = sims / tau
        m = np.max(scaled, axis=1, keepdims=True)
        return float((tau * (np.log(np.maximum(np.exp(scaled - m).sum(axis=1), 1e-12)) + m.squeeze(1))).mean())
    if model_name in {"B2_COLBERT", "B4_SLOT_MAXSIM"}:
        return float(sims.max(axis=1).mean())
    raise ValueError(f"Unsupported reverse model: {model_name}")


def main() -> None:
    args = parse_args()

    if args.sanity and args.max_eval_queries <= 0:
        args.max_eval_queries = 2000
        args.epochs = min(args.epochs, 8)
        args.max_train_pairs = min(args.max_train_pairs, 10000)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    topk_values = _parse_int_list(args.topk)
    run_dir = Path(args.output_dir) / args.run_id
    ensure_dir(run_dir)

    bundle = load_dataset(Path(args.dataset_path))
    if bundle.meta and bundle.meta.get("query_entity") != "ligand":
        raise ValueError("Expected reverse dataset with ligand queries.")

    split = build_split_view(bundle, split_name=args.split, seed=args.seed)
    query_rule = get_primary_query_rule(bundle)
    min_known_pos = args.min_known_pos_override if args.min_known_pos_override >= 0 else int(query_rule["min_known_pos"])
    default_neg = max(1, int(query_rule["min_known_neg"]))
    min_known_neg = args.min_known_neg_override if args.min_known_neg_override >= 0 else default_neg

    query_lookup = {q.pid: _as_ligand(q) for q in bundle.pockets}
    candidate_lookup = {c.lid: _as_pocket(c) for c in bundle.ligands}
    eval_query_ids = _subset_eval_queries(split.eval_pocket_ids, args.max_eval_queries, args.seed)
    candidate_ids = list(split.candidate_ligand_ids)
    train_pairs = _subset_train_pairs(split.train_pairs, args.max_train_pairs, args.seed)

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
        "dataset_path": args.dataset_path,
        "dataset_name": bundle.name,
        "split": args.split,
        "split_notes": split.notes,
        "direction": "reverse",
        "models": model_names,
        "seed": args.seed,
        "num_train_pairs": len(train_pairs),
        "num_eval_queries": len(eval_query_ids),
        "num_candidate_proteins": len(candidate_ids),
        "sanity": args.sanity,
        "max_eval_queries": args.max_eval_queries,
        "query_rule_used": {"min_known_pos": min_known_pos, "min_known_neg": min_known_neg},
        "results": {},
    }

    metric_rows: List[Dict[str, float]] = []

    for model_name in model_names:
        model_start = time.perf_counter()

        t0 = time.perf_counter()
        candidate_repr = {}
        for cid in candidate_ids:
            enc, _ = encode_pocket(model_name, candidate_lookup[cid], cfg)
            candidate_repr[cid] = enc
        candidate_encode_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        query_repr = {}
        for qid in set([qid for qid, _, _ in train_pairs] + eval_query_ids):
            enc, _ = encode_ligand(model_name, query_lookup[qid], cfg)
            query_repr[qid] = enc
        query_encode_sec = time.perf_counter() - t0

        raw_train: List[float] = []
        y_train: List[int] = []
        for qid, cid, label in train_pairs:
            raw_train.append(_reverse_raw_score(model_name, query_repr[qid], candidate_repr[cid]))
            y_train.append(int(label))

        calibrator: Calibrator = fit_calibrator(raw_train, y_train, cfg)
        _write_training_curve(run_dir / f"{model_name}_training.csv", model_name, calibrator.train_loss)

        query_rows: List[Dict[str, float]] = []
        score_start = time.perf_counter()
        for qid in eval_query_ids:
            raw_scores = []
            for cid in candidate_ids:
                raw = _reverse_raw_score(model_name, query_repr[qid], candidate_repr[cid])
                score = calibrator.alpha * raw + calibrator.beta
                raw_scores.append((cid, score))
            raw_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_ids = [cid for cid, _ in raw_scores]
            row = _reverse_metrics(
                ranked_ids=ranked_ids,
                label_row=bundle.labels[qid],
                topk_values=topk_values,
                min_known_pos=min_known_pos,
                min_known_neg=min_known_neg,
            )
            if row:
                query_rows.append(row)

            if args.profile_repeats > 1:
                for _ in range(args.profile_repeats - 1):
                    for cid in candidate_ids:
                        _ = _reverse_raw_score(model_name, query_repr[qid], candidate_repr[cid])

        scoring_sec = time.perf_counter() - score_start

        metrics = _aggregate_query_metrics(query_rows, topk_values)
        num_evaluable = len(query_rows)
        metrics["num_total_eval_queries"] = float(len(eval_query_ids))
        metrics["num_evaluable_queries"] = float(num_evaluable)
        metrics["evaluable_query_coverage"] = num_evaluable / max(1, len(eval_query_ids))

        total_pairs = len(eval_query_ids) * len(candidate_ids)
        qps = (total_pairs * max(1, args.profile_repeats)) / max(1e-12, scoring_sec)

        model_result = {
            "model": model_name,
            "metrics": metrics,
            "calibrator": {
                "alpha": calibrator.alpha,
                "beta": calibrator.beta,
                "final_train_logloss": calibrator.train_loss[-1] if calibrator.train_loss else None,
            },
            "efficiency": {
                "query_encode_sec": query_encode_sec,
                "candidate_encode_sec": candidate_encode_sec,
                "scoring_sec": scoring_sec,
                "scoring_qps": qps,
            },
            "runtime_sec": time.perf_counter() - model_start,
        }

        save_json(run_dir / f"{model_name}.json", model_result)
        summary["results"][model_name] = model_result

        metric_row = {"model": model_name}
        metric_row.update(metrics)
        metric_row["scoring_qps"] = qps
        metric_rows.append(metric_row)

    _write_metrics_csv(run_dir / "metrics.csv", metric_rows)
    save_json(run_dir / "summary.json", summary)

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "models": model_names,
                "dataset": args.dataset_path,
                "split": args.split,
                "num_train_pairs": len(train_pairs),
                "num_eval_queries": len(eval_query_ids),
                "num_candidate_proteins": len(candidate_ids),
                "output": str(run_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
