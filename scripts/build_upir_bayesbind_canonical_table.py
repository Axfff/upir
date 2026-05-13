#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a UPIR canonical edge table from a local BayesBind tar archive.")
    p.add_argument("--bayesbind-tar", default="data/bigbind/bayesbind/BayesBindV1.5.tar.gz")
    p.add_argument("--out-parquet", default="data/real_benchmarks/upir/bayesbind_canonical_edges.parquet")
    p.add_argument("--summary-json", default="data/real_benchmarks/upir/bayesbind_canonical_edges_summary.json")
    return p.parse_args()


def _ligand_id(smiles: str) -> str:
    key = (smiles or "").strip()
    return "BAYLIG::" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _proxy_scaffold(smiles: str, lig_cluster: str) -> str:
    if lig_cluster:
        return f"LIG_CLUSTER::{lig_cluster}"
    core = "".join(ch for ch in smiles if ch.isalnum())[:16] or "UNK"
    return f"S::{core}"


def _merge_label(prev: Dict, label: int, label_source: str, row: Dict) -> Dict:
    if prev is None:
        merged = dict(row)
        merged["label"] = int(label)
        merged["label_source"] = label_source
        return merged
    prev["label"] = max(int(prev["label"]), int(label))
    if prev.get("label_source") != label_source and label == 1:
        prev["label_source"] = label_source
    return prev


def main() -> None:
    args = parse_args()
    tar_path = Path(args.bayesbind_tar)
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing BayesBind tar archive: {tar_path}")

    merged: Dict[Tuple[str, str], Dict] = {}
    target_stats = defaultdict(lambda: {"pos": 0, "assay_neg": 0, "decoy_neg": 0})

    with tarfile.open(tar_path) as tf:
        members = [m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(("actives.csv", "random.csv"))]
        for name in sorted(members):
            parts = name.split("/")
            if len(parts) != 4:
                continue
            _, split, target, fname = parts
            query_id = f"BAYES::{target}"
            with tf.extractfile(name) as fh:
                reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
                for row in reader:
                    smiles = (row.get("lig_smiles") or "").strip()
                    if not smiles:
                        continue
                    candidate_id = _ligand_id(smiles)
                    lig_cluster = (row.get("lig_cluster") or "").strip()
                    base_row = {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "smiles": smiles,
                        "scaffold": _proxy_scaffold(smiles, lig_cluster),
                        "source": "BayesBindV1.5",
                        "query_split": split,
                        "query_target": target,
                        "query_uniprot": (row.get("uniprot") or "").strip(),
                        "query_rec_cluster": (row.get("rec_cluster") or "").strip(),
                        "candidate_source_id": (row.get("lig_file") or "").strip(),
                        "lig_cluster": lig_cluster,
                        "standard_type": (row.get("standard_type") or "").strip(),
                        "standard_relation": (row.get("standard_relation") or "").strip(),
                        "standard_value": (row.get("standard_value") or "").strip(),
                        "standard_units": (row.get("standard_units") or "").strip(),
                        "pchembl_value": (row.get("pchembl_value") or "").strip(),
                    }
                    key = (query_id, candidate_id)
                    if fname == "actives.csv":
                        is_active = (row.get("active") or "").strip().lower() == "true"
                        label = 1 if is_active else 0
                        label_source = "assay_active" if is_active else "assay_inactive"
                        merged[key] = _merge_label(merged.get(key), label, label_source, base_row)
                    else:
                        label = 0
                        label_source = "random_decoy"
                        merged[key] = _merge_label(merged.get(key), label, label_source, base_row)

    rows = []
    for (query_id, _), row in sorted(merged.items()):
        rows.append(row)
        if int(row["label"]) == 1:
            target_stats[query_id]["pos"] += 1
        elif row.get("label_source") == "assay_inactive":
            target_stats[query_id]["assay_neg"] += 1
        else:
            target_stats[query_id]["decoy_neg"] += 1

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)

    summary = {
        "out_parquet": str(out),
        "num_edges": len(rows),
        "num_queries": len(target_stats),
        "num_candidates": len({row['candidate_id'] for row in rows}),
        "num_pos": sum(v["pos"] for v in target_stats.values()),
        "num_assay_neg": sum(v["assay_neg"] for v in target_stats.values()),
        "num_decoy_neg": sum(v["decoy_neg"] for v in target_stats.values()),
        "label_regime": "mixed_assay_plus_random_decoy",
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
