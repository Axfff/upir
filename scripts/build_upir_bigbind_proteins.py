#!/usr/bin/env python3
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build UPIR protein/pocket entities from a full BigBind tar archive.")
    p.add_argument("--bigbind-tar", default="data/bigbind/BigBindV1.5.tar.gz")
    p.add_argument("--out-parquet", default="data/real_benchmarks/upir/bigbind_proteins.parquet")
    p.add_argument("--summary-json", default="data/real_benchmarks/upir/bigbind_proteins_summary.json")
    p.add_argument("--max-pocket-residues", type=int, default=128)
    return p.parse_args()


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _parse_residue_text_from_pdb_bytes(raw: bytes, max_residues: int) -> str:
    residues: List[str] = []
    seen = set()
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip()
        resid = line[22:26].strip()
        key = (chain, resid, resname)
        if key in seen:
            continue
        seen.add(key)
        residues.append(THREE_TO_ONE.get(resname, resname))
        if len(residues) >= max_residues:
            break
    return " ".join(residues)


def main() -> None:
    args = parse_args()
    tar_path = Path(args.bigbind_tar)
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing BigBind tar archive: {tar_path}")

    grouped: Dict[str, Dict] = {}
    with tarfile.open(tar_path) as tf:
        members = {m.name for m in tf.getmembers() if m.isfile()}
        with tf.extractfile("BigBindV1.5/activities_all.csv") as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            for row in reader:
                pocket = (row.get("pocket") or "").strip()
                if not pocket:
                    continue
                query_id = f"BIGBIND::{pocket}"
                if query_id in grouped:
                    continue
                pocket_pdb = f"BigBindV1.5/{(row.get('ex_rec_pocket_file') or '').strip()}"
                if pocket_pdb in members:
                    with tf.extractfile(pocket_pdb) as pdb_fh:
                        text = _parse_residue_text_from_pdb_bytes(pdb_fh.read(), max_residues=args.max_pocket_residues)
                else:
                    text = ""
                if not text.strip():
                    text = pocket
                grouped[query_id] = {
                    "protein_id": query_id,
                    "query_type": "protein_pocket",
                    "source": "BigBindV1.5",
                    "source_target": pocket,
                    "source_uniprot": (row.get("uniprot") or "").strip(),
                    "protein_cluster": f"BIGBIND_REC_CLUSTER::{(row.get('rec_cluster') or '').strip() or pocket}",
                    "text": text,
                    "text_kind": "protein_residue",
                    "representative_pocket_file": pocket_pdb if pocket_pdb in members else "",
                    "representative_receptor_pdb": (row.get("ex_rec_pdb") or "").strip(),
                }

    rows = [grouped[k] for k in sorted(grouped.keys())]
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)

    summary = {"out_parquet": str(out), "num_proteins": len(rows)}
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
