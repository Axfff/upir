#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
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
    p = argparse.ArgumentParser(
        description="Audit feasibility of UPIR open reverse retrieval by transposing an open forward graph."
    )
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_forward.json")
    p.add_argument("--protocol-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_forward_protocol.json")
    p.add_argument("--out-json", default="results/upir_open_bigbind_reverse_audit.json")
    p.add_argument("--out-md", default="results/upir_open_bigbind_reverse_audit.md")
    p.add_argument("--timestamp", default="")
    p.add_argument("--topk", default="1,5,10,50")
    p.add_argument("--top-popularities", type=int, default=20)
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _percentile(sorted_values: Sequence[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _summary(values: Sequence[int]) -> Dict[str, float]:
    vals = sorted(int(v) for v in values)
    if not vals:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(vals[0]),
        "p25": _percentile(vals, 0.25),
        "median": _percentile(vals, 0.50),
        "p75": _percentile(vals, 0.75),
        "p90": _percentile(vals, 0.90),
        "p95": _percentile(vals, 0.95),
        "max": float(vals[-1]),
        "mean": float(sum(vals)) / len(vals),
    }


def _histogram(values: Sequence[int], bins: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for b in bins:
        out[f">={b}"] = sum(1 for v in values if v >= b)
    return out


def _gini(values: Sequence[int]) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 0.0:
        return 0.0
    weighted = sum((idx + 1) * val for idx, val in enumerate(vals))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _popularity_order(pos_degree_by_protein: Dict[str, int], protein_ids: Sequence[str]) -> List[str]:
    return sorted(protein_ids, key=lambda pid: (-pos_degree_by_protein.get(pid, 0), stable_seed(pid)))


def _query_popularity_metrics(
    positives_by_ligand: Dict[str, set],
    order: Sequence[str],
    topk_values: Sequence[int],
) -> Dict[str, float]:
    rank = {pid: idx + 1 for idx, pid in enumerate(order)}
    rows: List[Dict[str, float]] = []
    for pos_set in positives_by_ligand.values():
        if not pos_set:
            continue
        pos_ranks = sorted(rank[pid] for pid in pos_set if pid in rank)
        if not pos_ranks:
            continue
        row = {
            "best_positive_rank": float(pos_ranks[0]),
            "num_pos": float(len(pos_ranks)),
        }
        for k in topk_values:
            hits = sum(1 for r in pos_ranks if r <= k)
            row[f"hit@{k}"] = 1.0 if hits else 0.0
            row[f"recall@{k}"] = hits / len(pos_ranks)
        rows.append(row)

    if not rows:
        return {"num_evaluable_queries": 0.0}
    out: Dict[str, float] = {"num_evaluable_queries": float(len(rows))}
    for key in sorted({k for row in rows for k in row}):
        out[key] = sum(row.get(key, 0.0) for row in rows) / len(rows)
    return out


def _random_expectation(num_candidates: int, pos_counts: Sequence[int], topk_values: Sequence[int]) -> Dict[str, float]:
    positive_counts = [p for p in pos_counts if p > 0]
    if not positive_counts or num_candidates <= 0:
        return {"num_evaluable_queries": 0.0}
    out: Dict[str, float] = {"num_evaluable_queries": float(len(positive_counts))}
    for k in topk_values:
        kk = min(k, num_candidates)
        recalls = [kk / num_candidates for _ in positive_counts]
        hits = [1.0 - math.exp(math.lgamma(num_candidates - p + 1) + math.lgamma(num_candidates - kk + 1) - math.lgamma(num_candidates - p - kk + 1) - math.lgamma(num_candidates + 1)) if num_candidates - p >= kk else 1.0 for p in positive_counts]
        out[f"expected_recall@{k}"] = sum(recalls) / len(recalls)
        out[f"expected_hit@{k}"] = sum(hits) / len(hits)
    return out


def _recommendation(
    total_proteins: int,
    pos_query_coverage: float,
    discr_query_coverage: float,
    pop_diag: Dict[str, float],
) -> Dict[str, str]:
    hit10 = float(pop_diag.get("hit@10", 0.0))
    hit50 = float(pop_diag.get("hit@50", 0.0))
    if total_proteins <= 15:
        mode = "no_go"
        rationale = "protein candidate pool is not materially larger than strict reverse"
    elif pos_query_coverage >= 0.10 and hit10 < 0.80:
        mode = "materialize_candidate_generation"
        rationale = "positive-query coverage is broad, protein pool is large, and top-10 popularity does not fully solve the task"
    elif pos_query_coverage >= 0.10:
        mode = "materialize_with_popularity_warning"
        rationale = "positive-query coverage is broad, but popularity is strong enough to require degree-stratified reporting"
    else:
        mode = "audit_only"
        rationale = "positive-query coverage is too sparse for a main benchmark track"

    if discr_query_coverage < 0.05:
        discriminative = "secondary_or_appendix_only"
    else:
        discriminative = "auxiliary_diagnostic"
    return {"candidate_generation_decision": mode, "rationale": rationale, "discriminative_mode": discriminative, "popularity_hit10": f"{hit10:.4f}", "popularity_hit50": f"{hit50:.4f}"}


def _write_markdown(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    stats = payload["reverse_graph"]
    cov = payload["query_coverage"]
    pop = payload["protein_popularity"]
    diag = payload["popularity_diagnostic"]
    rec = payload["recommendation"]

    lines = [
        "# UPIR Open Reverse Audit",
        "",
        f"- source dataset: `{payload['source']['dataset_name']}`",
        f"- source path: `{payload['source']['dataset_path']}`",
        f"- audit timestamp: `{payload['audit_timestamp']}`",
        "",
        "## Reverse Graph",
        "",
        f"- ligand queries in pool: `{stats['total_ligand_queries']}`",
        f"- protein candidates in pool: `{stats['total_protein_candidates']}`",
        f"- known labeled pairs: `{stats['known_pairs']}`",
        f"- positive pairs: `{stats['positive_pairs']}`",
        f"- negative pairs: `{stats['negative_pairs']}`",
        "",
        "## Query Coverage",
        "",
        "| Rule | Queries | Coverage Over All Ligands | Coverage Over Labeled Ligands |",
        "|---|---:|---:|---:|",
        f"| `>=1` known label | {cov['with_any_known_label']} | {cov['with_any_known_label_rate_all']:.4f} | 1.0000 |",
        f"| `1+/0-` candidate generation | {cov['with_ge1_positive']} | {cov['with_ge1_positive_rate_all']:.4f} | {cov['with_ge1_positive_rate_labeled']:.4f} |",
        f"| `1+/1-` discriminative | {cov['with_ge1_positive_ge1_negative']} | {cov['with_ge1_positive_ge1_negative_rate_all']:.4f} | {cov['with_ge1_positive_ge1_negative_rate_labeled']:.4f} |",
        f"| `1+/5-` stricter discriminative | {cov['with_ge1_positive_ge5_negative']} | {cov['with_ge1_positive_ge5_negative_rate_all']:.4f} | {cov['with_ge1_positive_ge5_negative_rate_labeled']:.4f} |",
        f"| `2+/0-` multi-target positives | {cov['with_ge2_positive']} | {cov['with_ge2_positive_rate_all']:.4f} | {cov['with_ge2_positive_rate_labeled']:.4f} |",
        "",
        "## Positive Degree",
        "",
        f"- positives per positive ligand query: median `{stats['positive_count_stats_positive_queries']['median']:.1f}`, mean `{stats['positive_count_stats_positive_queries']['mean']:.3f}`, max `{stats['positive_count_stats_positive_queries']['max']:.0f}`",
        f"- negatives per labeled ligand query: median `{stats['negative_count_stats_labeled_queries']['median']:.1f}`, mean `{stats['negative_count_stats_labeled_queries']['mean']:.3f}`, max `{stats['negative_count_stats_labeled_queries']['max']:.0f}`",
        "",
        "## Protein Popularity",
        "",
        f"- positive-degree Gini: `{pop['positive_degree_gini']:.4f}`",
        f"- top-1 protein positive-pair share: `{pop['top1_positive_pair_share']:.4f}`",
        f"- top-5 protein positive-pair share: `{pop['top5_positive_pair_share']:.4f}`",
        f"- top-10 protein positive-pair share: `{pop['top10_positive_pair_share']:.4f}`",
        "",
        "Top proteins by positive reverse degree:",
        "",
        "| Rank | Protein | Positives | Share |",
        "|---:|---|---:|---:|",
    ]
    for row in pop["top_positive_proteins"][:10]:
        lines.append(f"| {row['rank']} | `{row['protein_id']}` | {row['positive_degree']} | {row['positive_pair_share']:.4f} |")

    lines.extend(
        [
            "",
            "## Popularity Diagnostic",
            "",
            "This is a global graph diagnostic, not an official train/test baseline.",
            "",
            "| Diagnostic | Hit@1 | Hit@5 | Hit@10 | Hit@50 | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Best Pos Rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| protein popularity | {h1:.4f} | {h5:.4f} | {h10:.4f} | {h50:.4f} | {r1:.4f} | {r5:.4f} | {r10:.4f} | {r50:.4f} | {br:.2f} |".format(
                h1=diag.get("hit@1", 0.0),
                h5=diag.get("hit@5", 0.0),
                h10=diag.get("hit@10", 0.0),
                h50=diag.get("hit@50", 0.0),
                r1=diag.get("recall@1", 0.0),
                r5=diag.get("recall@5", 0.0),
                r10=diag.get("recall@10", 0.0),
                r50=diag.get("recall@50", 0.0),
                br=diag.get("best_positive_rank", 0.0),
            ),
            "",
            "## Recommendation",
            "",
            f"- candidate-generation decision: `{rec['candidate_generation_decision']}`",
            f"- discriminative mode: `{rec['discriminative_mode']}`",
            f"- rationale: {rec['rationale']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topk_values = _parse_ints(args.topk)
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    stamped_json = out_json.with_name(f"{out_json.stem}_{stamp}{out_json.suffix}")
    stamped_md = out_md.with_name(f"{out_md.stem}_{stamp}{out_md.suffix}")

    dataset_path = Path(args.dataset_path)
    protocol_path = Path(args.protocol_path)
    bundle = load_dataset(dataset_path)

    protein_ids = [p.pid for p in bundle.pockets]
    ligand_ids = [l.lid for l in bundle.ligands]
    positives_by_ligand: Dict[str, set] = defaultdict(set)
    negatives_by_ligand: Dict[str, set] = defaultdict(set)
    pos_degree_by_protein: Counter = Counter()
    neg_degree_by_protein: Counter = Counter()

    known_pairs = 0
    positive_pairs = 0
    negative_pairs = 0
    for pid, label_row in bundle.labels.items():
        for lid, label in label_row.items():
            known_pairs += 1
            if int(label) == 1:
                positives_by_ligand[lid].add(pid)
                pos_degree_by_protein[pid] += 1
                positive_pairs += 1
            else:
                negatives_by_ligand[lid].add(pid)
                neg_degree_by_protein[pid] += 1
                negative_pairs += 1

    pos_counts_all = [len(positives_by_ligand.get(lid, set())) for lid in ligand_ids]
    neg_counts_all = [len(negatives_by_ligand.get(lid, set())) for lid in ligand_ids]
    known_counts_all = [p + n for p, n in zip(pos_counts_all, neg_counts_all)]
    labeled_ligands = [lid for lid, known in zip(ligand_ids, known_counts_all) if known > 0]
    positive_ligands = [lid for lid in ligand_ids if len(positives_by_ligand.get(lid, set())) >= 1]

    with_any_known = len(labeled_ligands)
    with_ge1_pos = len(positive_ligands)
    with_ge1_pos_ge1_neg = sum(1 for lid in ligand_ids if len(positives_by_ligand.get(lid, set())) >= 1 and len(negatives_by_ligand.get(lid, set())) >= 1)
    with_ge1_pos_ge5_neg = sum(1 for lid in ligand_ids if len(positives_by_ligand.get(lid, set())) >= 1 and len(negatives_by_ligand.get(lid, set())) >= 5)
    with_ge2_pos = sum(1 for lid in ligand_ids if len(positives_by_ligand.get(lid, set())) >= 2)

    total_pos_pairs = max(1, positive_pairs)
    top_order = _popularity_order(pos_degree_by_protein, protein_ids)
    top_rows = [
        {
            "rank": idx + 1,
            "protein_id": pid,
            "positive_degree": int(pos_degree_by_protein.get(pid, 0)),
            "negative_degree": int(neg_degree_by_protein.get(pid, 0)),
            "positive_pair_share": _safe_rate(pos_degree_by_protein.get(pid, 0), total_pos_pairs),
        }
        for idx, pid in enumerate(top_order[: args.top_popularities])
    ]

    pop_diag = _query_popularity_metrics(positives_by_ligand, top_order, topk_values)
    random_diag = _random_expectation(len(protein_ids), [len(positives_by_ligand.get(lid, set())) for lid in positive_ligands], topk_values)
    recommendation = _recommendation(
        total_proteins=len(protein_ids),
        pos_query_coverage=_safe_rate(with_ge1_pos, len(ligand_ids)),
        discr_query_coverage=_safe_rate(with_ge1_pos_ge1_neg, len(ligand_ids)),
        pop_diag=pop_diag,
    )

    payload = {
        "name": "UPIR_open_bigbind_reverse_audit",
        "audit_timestamp": stamp,
        "source": {
            "dataset_name": bundle.name,
            "dataset_path": str(dataset_path),
            "protocol_path": str(protocol_path),
            "protocol_exists": protocol_path.exists(),
        },
        "reverse_graph": {
            "total_ligand_queries": len(ligand_ids),
            "total_protein_candidates": len(protein_ids),
            "known_pairs": known_pairs,
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "positive_count_stats_all_ligands": _summary(pos_counts_all),
            "negative_count_stats_all_ligands": _summary(neg_counts_all),
            "known_count_stats_all_ligands": _summary(known_counts_all),
            "positive_count_stats_positive_queries": _summary([len(positives_by_ligand[lid]) for lid in positive_ligands]),
            "negative_count_stats_labeled_queries": _summary([len(negatives_by_ligand.get(lid, set())) for lid in labeled_ligands]),
            "positive_count_thresholds_all_ligands": _histogram(pos_counts_all, [1, 2, 3, 5, 10, 20, 50]),
            "negative_count_thresholds_all_ligands": _histogram(neg_counts_all, [1, 2, 5, 10, 20, 50]),
        },
        "query_coverage": {
            "with_any_known_label": with_any_known,
            "with_any_known_label_rate_all": _safe_rate(with_any_known, len(ligand_ids)),
            "with_ge1_positive": with_ge1_pos,
            "with_ge1_positive_rate_all": _safe_rate(with_ge1_pos, len(ligand_ids)),
            "with_ge1_positive_rate_labeled": _safe_rate(with_ge1_pos, with_any_known),
            "with_ge1_positive_ge1_negative": with_ge1_pos_ge1_neg,
            "with_ge1_positive_ge1_negative_rate_all": _safe_rate(with_ge1_pos_ge1_neg, len(ligand_ids)),
            "with_ge1_positive_ge1_negative_rate_labeled": _safe_rate(with_ge1_pos_ge1_neg, with_any_known),
            "with_ge1_positive_ge5_negative": with_ge1_pos_ge5_neg,
            "with_ge1_positive_ge5_negative_rate_all": _safe_rate(with_ge1_pos_ge5_neg, len(ligand_ids)),
            "with_ge1_positive_ge5_negative_rate_labeled": _safe_rate(with_ge1_pos_ge5_neg, with_any_known),
            "with_ge2_positive": with_ge2_pos,
            "with_ge2_positive_rate_all": _safe_rate(with_ge2_pos, len(ligand_ids)),
            "with_ge2_positive_rate_labeled": _safe_rate(with_ge2_pos, with_any_known),
        },
        "protein_popularity": {
            "positive_degree_stats": _summary([int(pos_degree_by_protein.get(pid, 0)) for pid in protein_ids]),
            "negative_degree_stats": _summary([int(neg_degree_by_protein.get(pid, 0)) for pid in protein_ids]),
            "positive_degree_gini": _gini([int(pos_degree_by_protein.get(pid, 0)) for pid in protein_ids]),
            "top1_positive_pair_share": sum(r["positive_degree"] for r in top_rows[:1]) / total_pos_pairs,
            "top5_positive_pair_share": sum(r["positive_degree"] for r in top_rows[:5]) / total_pos_pairs,
            "top10_positive_pair_share": sum(r["positive_degree"] for r in top_rows[:10]) / total_pos_pairs,
            "top_positive_proteins": top_rows,
        },
        "popularity_diagnostic": pop_diag,
        "random_expectation_diagnostic": random_diag,
        "recommendation": recommendation,
        "notes": [
            "This audit transposes dataset ground-truth labels from the forward BigBind open graph.",
            "Popularity diagnostic uses the full graph only to estimate dominance risk; it is not an official train/test result.",
            "Candidate-generation mode uses ligand queries with >=1 known positive protein.",
            "Discriminative mode additionally requires known negative protein labels.",
        ],
    }

    save_json(stamped_json, payload)
    _write_markdown(stamped_md, payload)
    ensure_dir(out_json.parent)
    shutil.copyfile(stamped_json, out_json)
    shutil.copyfile(stamped_md, out_md)
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "timestamped_json": str(stamped_json), "timestamped_markdown": str(stamped_md)}, indent=2))


if __name__ == "__main__":
    main()
