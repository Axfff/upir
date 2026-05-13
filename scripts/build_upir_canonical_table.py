#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment_bridge.text_tokens import parse_smi, scaffold_from_smiles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical UPIR edge table from local LIT-PCBA data.")
    p.add_argument("--litpcba-root", default="data/external_raw/LIT-PCBA_full")
    p.add_argument("--out-parquet", default="data/real_benchmarks/upir/upir_canonical_edges.parquet")
    p.add_argument("--summary-json", default="data/real_benchmarks/upir/upir_canonical_edges_summary.json")
    p.add_argument("--min-actives", type=int, default=1)
    p.add_argument("--max-pocket-residues", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.litpcba_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing LIT-PCBA root: {root}")

    columns: Dict[str, List] = {
        "query_type": [],
        "query_id": [],
        "candidate_type": [],
        "candidate_id": [],
        "label": [],
        "label_regime": [],
        "source": [],
        "assay_id": [],
        "activity_value": [],
        "activity_type": [],
        "timestamp": [],
        "protein_cluster": [],
        "scaffold": [],
        "smiles": [],
    }
    per_target: Dict[str, Dict[str, int]] = {}

    for target_dir in sorted(root.iterdir()):
        if not target_dir.is_dir():
            continue
        actives_path = target_dir / "actives.smi"
        inactives_path = target_dir / "inactives.smi"
        if not actives_path.exists() or not inactives_path.exists():
            continue

        actives = parse_smi(actives_path)
        inactives = parse_smi(inactives_path)
        if len(actives) < args.min_actives:
            continue

        query_id = f"LIT::{target_dir.name}"
        cluster_id = f"LIT_CLUSTER::{target_dir.name}"

        pos_count = 0
        neg_count = 0
        for label, items in ((1, actives), (0, inactives)):
            for smiles, cid in items:
                columns["query_type"].append("protein_pocket")
                columns["query_id"].append(query_id)
                columns["candidate_type"].append("ligand")
                columns["candidate_id"].append(f"CID::{cid}")
                columns["label"].append(label)
                columns["label_regime"].append("strict")
                columns["source"].append("LIT-PCBA")
                columns["assay_id"].append(target_dir.name)
                columns["activity_value"].append(None)
                columns["activity_type"].append("confirmed_binary")
                columns["timestamp"].append(None)
                columns["protein_cluster"].append(cluster_id)
                columns["scaffold"].append(scaffold_from_smiles(smiles))
                columns["smiles"].append(smiles)
                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1

        per_target[query_id] = {"num_pos": pos_count, "num_neg": neg_count}

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, out)

    summary = {
        "out_parquet": str(out),
        "num_rows": len(columns["label"]),
        "num_targets": len(per_target),
        "num_pos": sum(v["num_pos"] for v in per_target.values()),
        "num_neg": sum(v["num_neg"] for v in per_target.values()),
        "per_target": per_target,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
