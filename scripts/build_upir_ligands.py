#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build UPIR ligand entity table from canonical edge parquet.")
    p.add_argument("--canonical-parquet", default="data/real_benchmarks/upir/upir_canonical_edges.parquet")
    p.add_argument("--out-parquet", default="data/real_benchmarks/upir/upir_ligands.parquet")
    p.add_argument("--summary-json", default="data/real_benchmarks/upir/upir_ligands_summary.json")
    return p.parse_args()


def _canonical_smiles(raw_smiles: str) -> str:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return raw_smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _murcko_scaffold(raw_smiles: str) -> tuple[str, bool]:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return "INVALID_SMILES", False
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold, True
    return "NO_MURCKO", False


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.canonical_parquet)
    cols = table.to_pydict()
    grouped: Dict[str, Dict] = {}
    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    num_ring_scaffolds = 0
    num_no_murcko = 0
    num_invalid = 0

    for ligand_id, smiles, scaffold, label in zip(
        cols["candidate_id"], cols["smiles"], cols["scaffold"], cols["label"]
    ):
        if ligand_id not in grouped:
            canonical_smiles = _canonical_smiles(smiles)
            murcko, has_ring_scaffold = _murcko_scaffold(canonical_smiles)
            if murcko == "INVALID_SMILES":
                num_invalid += 1
            elif has_ring_scaffold:
                num_ring_scaffolds += 1
            else:
                num_no_murcko += 1
            grouped[ligand_id] = {
                "ligand_id": ligand_id,
                "source": "LIT-PCBA",
                "text": canonical_smiles,
                "text_kind": "smiles",
                "canonical_smiles": canonical_smiles,
                "scaffold": murcko,
                "scaffold_semantics": "bemis_murcko",
                "legacy_scaffold_proxy": scaffold,
            }
        if label == 1:
            pos_counts[ligand_id] += 1
        else:
            neg_counts[ligand_id] += 1

    rows: List[Dict] = []
    for ligand_id in sorted(grouped.keys()):
        row = dict(grouped[ligand_id])
        row["num_positive_edges"] = pos_counts[ligand_id]
        row["num_negative_edges"] = neg_counts[ligand_id]
        rows.append(row)

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)

    summary = {
        "out_parquet": str(out),
        "num_ligands": len(rows),
        "scaffold_semantics": "bemis_murcko",
        "num_ring_scaffold_ligands": num_ring_scaffolds,
        "num_no_murcko_ligands": num_no_murcko,
        "num_invalid_smiles": num_invalid,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
