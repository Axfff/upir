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
    p = argparse.ArgumentParser(description="Analyze open reverse candidate-generation performance by protein popularity degree.")
    p.add_argument("--dataset-path", default="data/real_benchmarks/upir/UPIR_open_bigbind_reverse.json")
    p.add_argument("--out-dir", default="results/upir_open_bigbind_reverse_degree_slices_v1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--topk", default="1,5,10,50")
    return p.parse_args()


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


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


def _bucket(best_pop_rank: int, num_pos: int) -> str:
    if num_pos >= 2:
        return "multi_positive"
    if best_pop_rank <= 10:
        return "single_pos_head10"
    if best_pop_rank <= 50:
        return "single_pos_head50"
    if best_pop_rank <= 200:
        return "single_pos_mid200"
    return "single_pos_tail"


def _row_metrics(pos_ranks: Sequence[int], topk_values: Sequence[int]) -> Dict[str, float]:
    out = {
        "best_positive_rank": float(min(pos_ranks)),
        "num_pos": float(len(pos_ranks)),
    }
    for k in topk_values:
        hits = sum(1 for r in pos_ranks if r <= k)
        out[f"hit@{k}"] = 1.0 if hits else 0.0
        out[f"recall@{k}"] = hits / len(pos_ranks)
    return out


def _aggregate(rows: Sequence[Dict[str, float]], topk_values: Sequence[int]) -> Dict[str, float]:
    if not rows:
        out = {"num_queries": 0.0}
        for k in topk_values:
            out[f"hit@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
        out["best_positive_rank"] = 0.0
        out["num_pos"] = 0.0
        return out
    keys = sorted({k for row in rows for k in row})
    out = {"num_queries": float(len(rows))}
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
        "# UPIR Open Reverse Degree-Slice Analysis",
        "",
        f"- dataset: `{payload['dataset_name']}`",
        "- split: `standard` pair-hash 5-fold",
        "- baseline: reverse protein positive-degree popularity",
        "- slice definition: bucket each evaluable ligand by the popularity rank of its held-out positive protein(s)",
        "",
        "## Aggregate By Slice",
        "",
        "| Slice | AvgQ/Fold | Hit@1 | Hit@5 | Hit@10 | Hit@50 | Recall@10 | BestPosRank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in payload["slice_order"]:
        row = payload["summary_by_slice"][name]
        lines.append(
            f"| `{name}` | {row.get('num_queries', 0.0):.0f} | {row.get('hit@1', 0.0):.4f} | {row.get('hit@5', 0.0):.4f} | "
            f"{row.get('hit@10', 0.0):.4f} | {row.get('hit@50', 0.0):.4f} | {row.get('recall@10', 0.0):.4f} | {row.get('best_positive_rank', 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Head-target queries are easy for the popularity baseline by construction.",
            "- Tail-target queries remain difficult, so open reverse is not only a popularity lookup problem.",
            "- Main-paper reporting should include this table or a shortened version to make the popularity sensitivity explicit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    topk_values = _parse_ints(args.topk)
    bundle = load_dataset(Path(args.dataset_path))
    candidate_ids = [l.lid for l in bundle.ligands]
    slice_order = ["single_pos_head10", "single_pos_head50", "single_pos_mid200", "single_pos_tail", "multi_positive"]
    per_slice_folds = {name: [] for name in slice_order}
    fold_payload = []

    for fold in range(args.folds):
        train_pairs, eval_labels = _split_pairs(bundle.labels, fold, args.folds)
        order = _popularity_order(train_pairs, candidate_ids)
        rank = {cid: idx + 1 for idx, cid in enumerate(order)}
        rows_by_slice = defaultdict(list)
        for qid, eval_row in eval_labels.items():
            pos_ids = [cid for cid, label in eval_row.items() if int(label) == 1]
            if not pos_ids:
                continue
            pos_ranks = sorted(rank[cid] for cid in pos_ids if cid in rank)
            if not pos_ranks:
                continue
            name = _bucket(best_pop_rank=min(pos_ranks), num_pos=len(pos_ranks))
            rows_by_slice[name].append(_row_metrics(pos_ranks, topk_values))
        fold_entry = {"fold": fold, "slices": {}}
        for name in slice_order:
            agg = _aggregate(rows_by_slice[name], topk_values)
            fold_entry["slices"][name] = agg
            per_slice_folds[name].append(agg)
        fold_payload.append(fold_entry)

    payload = {
        "dataset_name": bundle.name,
        "dataset_path": args.dataset_path,
        "evaluation_mode": "candidate_generation_retrieval",
        "baseline": "reverse_protein_positive_degree_popularity",
        "slice_order": slice_order,
        "summary_by_slice": {name: _average(per_slice_folds[name]) for name in slice_order},
        "folds": fold_payload,
    }
    save_json(out_dir / "summary.json", payload)
    _write_md(out_dir / "SUMMARY.md", payload)
    print(json.dumps({"output": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
