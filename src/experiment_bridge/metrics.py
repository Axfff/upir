from typing import Dict, List, Sequence, Tuple

from .utils import median


def is_query_evaluable(
    ranked_ids: Sequence[str],
    label_row: Dict[str, int],
    min_known_pos: int = 1,
    min_known_neg: int = 0,
) -> bool:
    ranked_ids = [lid for lid in ranked_ids if lid in label_row]
    num_pos = sum(1 for lid in ranked_ids if label_row.get(lid, 0) == 1)
    num_neg = len(ranked_ids) - num_pos
    return num_pos >= int(min_known_pos) and num_neg >= int(min_known_neg)


def _top_k_size(total: int, percent: float) -> int:
    return max(1, int(round(total * percent / 100.0)))


def evaluate_query_ranking(
    ranked_ids: Sequence[str],
    label_row: Dict[str, int],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
    min_known_pos: int = 1,
    min_known_neg: int = 0,
) -> Dict[str, float]:
    # Ignore unknown labels (sparse real datasets): evaluate on known pairs only.
    ranked_ids = [lid for lid in ranked_ids if lid in label_row]
    positives = [lid for lid in ranked_ids if label_row.get(lid, 0) == 1]
    num_pos = len(positives)
    num_neg = len(ranked_ids) - num_pos
    if num_pos < int(min_known_pos) or num_neg < int(min_known_neg):
        return {}

    pos_ranks = [idx + 1 for idx, lid in enumerate(ranked_ids) if label_row.get(lid, 0) == 1]
    first_pos_rank = min(pos_ranks)

    out: Dict[str, float] = {
        "rr": 1.0 / first_pos_rank,
        "median_rank_true_binders": median(pos_ranks),
    }

    for k in topk_values:
        top_ids = ranked_ids[:k]
        tp = sum(1 for lid in top_ids if label_row.get(lid, 0) == 1)
        out[f"recall@{k}"] = tp / num_pos
        out[f"hit@{k}"] = 1.0 if tp > 0 else 0.0

    baseline_rate = num_pos / max(1, len(ranked_ids))
    for p in ef_percents:
        top_n = _top_k_size(len(ranked_ids), p)
        top_ids = ranked_ids[:top_n]
        actives = sum(1 for lid in top_ids if label_row.get(lid, 0) == 1)
        top_rate = actives / top_n
        out[f"ef{int(p)}"] = top_rate / max(1e-12, baseline_rate)

    return out


def evaluate_candidate_generation_full_pool(
    ranked_ids: Sequence[str],
    label_row: Dict[str, int],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
    min_known_pos: int = 1,
) -> Dict[str, float]:
    """Evaluate positive recovery against the full ranked candidate pool.

    Unlike discriminative retrieval, candidate generation must not shrink the
    ranking to known labels: unlabeled candidates are unscored, not negatives.
    """
    pos_ranks = [idx + 1 for idx, lid in enumerate(ranked_ids) if label_row.get(lid, 0) == 1]
    num_pos = len(pos_ranks)
    if num_pos < int(min_known_pos):
        return {}

    first_pos_rank = min(pos_ranks)
    ranked_set_size = max(1, len(ranked_ids))
    out: Dict[str, float] = {
        "rr": 1.0 / first_pos_rank,
        "median_rank_true_binders": median(pos_ranks),
    }

    for k in topk_values:
        tp = sum(1 for rank in pos_ranks if rank <= k)
        out[f"recall@{k}"] = tp / num_pos
        out[f"hit@{k}"] = 1.0 if tp > 0 else 0.0

    baseline_rate = num_pos / ranked_set_size
    for p in ef_percents:
        top_n = _top_k_size(ranked_set_size, p)
        actives = sum(1 for rank in pos_ranks if rank <= top_n)
        top_rate = actives / top_n
        out[f"ef{int(p)}"] = top_rate / max(1e-12, baseline_rate)

    return out


def aggregate_query_metrics(
    query_metrics: Sequence[Dict[str, float]],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["num_eval_pockets"] = float(len(query_metrics))
    if not query_metrics:
        out["mrr"] = 0.0
        out["median_rank_true_binders"] = float("nan")
        for k in topk_values:
            out[f"recall@{k}"] = 0.0
            out[f"hit@{k}"] = 0.0
        for p in ef_percents:
            out[f"ef{int(p)}"] = 0.0
        return out

    out["mrr"] = sum(q["rr"] for q in query_metrics) / len(query_metrics)
    out["median_rank_true_binders"] = median([q["median_rank_true_binders"] for q in query_metrics])

    for k in topk_values:
        out[f"recall@{k}"] = sum(q[f"recall@{k}"] for q in query_metrics) / len(query_metrics)
        out[f"hit@{k}"] = sum(q[f"hit@{k}"] for q in query_metrics) / len(query_metrics)
    for p in ef_percents:
        key = f"ef{int(p)}"
        out[key] = sum(q[key] for q in query_metrics) / len(query_metrics)

    return out


def evaluate_retrieval(
    ranked_by_pocket: Dict[str, List[Tuple[str, float]]],
    labels: Dict[str, Dict[str, int]],
    topk_values: Sequence[int],
    ef_percents: Sequence[float],
    min_known_pos: int = 1,
    min_known_neg: int = 0,
) -> Dict[str, float]:
    query_metrics: List[Dict[str, float]] = []
    for pid, ranking in ranked_by_pocket.items():
        ranked_ids = [lid for lid, _ in ranking]
        metrics = evaluate_query_ranking(
            ranked_ids,
            labels[pid],
            topk_values,
            ef_percents,
            min_known_pos=min_known_pos,
            min_known_neg=min_known_neg,
        )
        if metrics:
            query_metrics.append(metrics)
    return aggregate_query_metrics(query_metrics, topk_values, ef_percents)
