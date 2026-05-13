#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.data import build_split_view, load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit overlap and leakage-related statistics for a UPIR benchmark split.")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--split", choices=["standard", "scaffold", "target", "joint_ood"], default="target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _scaffold_map(bundle) -> Dict[str, str]:
    return {l.lid: l.scaffold for l in bundle.ligands}


def _train_query_ids(train_pairs: List) -> Set[str]:
    return {pid for pid, _, _ in train_pairs}


def _train_candidate_ids(train_pairs: List) -> Set[str]:
    return {lid for _, lid, _ in train_pairs}


def _eval_pair_stats(bundle, query_ids: Set[str], candidate_ids: Set[str]) -> Dict[str, int]:
    num_known_pairs = 0
    num_pos = 0
    for pid in query_ids:
        for lid, label in bundle.labels.get(pid, {}).items():
            if candidate_ids and lid not in candidate_ids:
                continue
            num_known_pairs += 1
            if int(label) == 1:
                num_pos += 1
    return {
        "num_known_pairs": num_known_pairs,
        "num_pos": num_pos,
        "num_neg": num_known_pairs - num_pos,
    }


def main() -> None:
    args = parse_args()
    bundle = load_dataset(Path(args.dataset_path))
    split = build_split_view(bundle, split_name=args.split, seed=args.seed)
    scaffold_by_lid = _scaffold_map(bundle)

    train_query_ids = _train_query_ids(split.train_pairs)
    eval_query_ids = set(split.eval_pocket_ids)
    train_candidate_ids = _train_candidate_ids(split.train_pairs)
    eval_candidate_ids = set(split.candidate_ligand_ids)

    train_scaffolds = {scaffold_by_lid[lid] for lid in train_candidate_ids if lid in scaffold_by_lid}
    eval_scaffolds = {scaffold_by_lid[lid] for lid in eval_candidate_ids if lid in scaffold_by_lid}

    train_pos = sum(1 for _, _, label in split.train_pairs if label == 1)
    train_neg = len(split.train_pairs) - train_pos
    eval_stats = _eval_pair_stats(bundle, eval_query_ids, eval_candidate_ids)

    report = {
        "dataset_path": args.dataset_path,
        "dataset_name": bundle.name,
        "split": args.split,
        "seed": args.seed,
        "split_notes": split.notes,
        "train": {
            "num_queries": len(train_query_ids),
            "num_candidates_seen": len(train_candidate_ids),
            "num_pairs": len(split.train_pairs),
            "num_pos": train_pos,
            "num_neg": train_neg,
        },
        "eval": {
            "num_queries": len(eval_query_ids),
            "num_candidates": len(eval_candidate_ids),
            "num_known_pairs": eval_stats["num_known_pairs"],
            "num_pos": eval_stats["num_pos"],
            "num_neg": eval_stats["num_neg"],
        },
        "overlap": {
            "query_id_overlap": len(train_query_ids & eval_query_ids),
            "candidate_id_overlap": len(train_candidate_ids & eval_candidate_ids),
            "scaffold_overlap": len(train_scaffolds & eval_scaffolds),
            "train_scaffolds": len(train_scaffolds),
            "eval_scaffolds": len(eval_scaffolds),
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
