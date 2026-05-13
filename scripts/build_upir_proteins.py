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

from experiment_bridge.text_tokens import parse_residue_sequence_from_mol2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build UPIR protein/pocket entity table from local LIT-PCBA data.")
    p.add_argument("--litpcba-root", default="data/external_raw/LIT-PCBA_full")
    p.add_argument("--out-parquet", default="data/real_benchmarks/upir/upir_proteins.parquet")
    p.add_argument("--summary-json", default="data/real_benchmarks/upir/upir_proteins_summary.json")
    p.add_argument("--max-pocket-residues", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.litpcba_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing LIT-PCBA root: {root}")

    rows: List[Dict] = []
    for target_dir in sorted(root.iterdir()):
        if not target_dir.is_dir():
            continue
        actives_path = target_dir / "actives.smi"
        inactives_path = target_dir / "inactives.smi"
        if not actives_path.exists() or not inactives_path.exists():
            continue

        protein_files = sorted(target_dir.glob("*_protein.mol2"))
        protein_file = protein_files[0].name if protein_files else ""
        if protein_files:
            text = parse_residue_sequence_from_mol2(protein_files[0], max_residues=args.max_pocket_residues)
        else:
            text = target_dir.name
        if not text.strip():
            text = target_dir.name

        rows.append(
            {
                "protein_id": f"LIT::{target_dir.name}",
                "query_type": "protein_pocket",
                "source": "LIT-PCBA",
                "source_target": target_dir.name,
                "protein_cluster": f"LIT_CLUSTER::{target_dir.name}",
                "text": text,
                "text_kind": "protein_residue",
                "representative_protein_file": protein_file,
                "num_structure_files": len(protein_files),
            }
        )

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)

    summary = {"out_parquet": str(out), "num_proteins": len(rows)}
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
