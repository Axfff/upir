#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize UPIR-Strict forward/reverse JSON benchmarks from canonical tables.")
    p.add_argument("--canonical-parquet", default="data/real_benchmarks/upir/upir_canonical_edges.parquet")
    p.add_argument("--proteins-parquet", default="data/real_benchmarks/upir/upir_proteins.parquet")
    p.add_argument("--ligands-parquet", default="data/real_benchmarks/upir/upir_ligands.parquet")
    p.add_argument("--out-forward-json", default="data/real_benchmarks/upir/UPIR_strict_forward.json")
    p.add_argument("--out-reverse-json", default="data/real_benchmarks/upir/UPIR_strict_reverse.json")
    p.add_argument("--forward-protocol-path", default="UPIR_strict_forward_protocol.json")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--smiles-kmer", type=int, default=3)
    p.add_argument("--residue-kmer", type=int, default=2)
    p.add_argument("--max-smiles-tokens", type=int, default=64)
    p.add_argument("--max-pocket-residues", type=int, default=128)
    return p.parse_args()


def _read_pylist(path: str) -> List[Dict]:
    return pq.read_table(path).to_pylist()


def _write_json(path: str, payload: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))


def _token_meta(args: argparse.Namespace) -> Dict:
    return {
        "dim": args.dim,
        "kinds": {
            "protein_residue": {
                "k": args.residue_kmer,
                "max_tokens": args.max_pocket_residues,
                "base_seed": 17,
            },
            "smiles": {
                "k": args.smiles_kmer,
                "max_tokens": args.max_smiles_tokens,
                "base_seed": 701,
            },
        },
    }


def main() -> None:
    args = parse_args()
    protein_rows = _read_pylist(args.proteins_parquet)
    ligand_rows = _read_pylist(args.ligands_parquet)

    forward_labels: Dict[str, Dict[str, int]] = {}
    per_target: Dict[str, Dict[str, int]] = {}
    num_edges = 0
    num_pos = 0
    num_neg = 0
    for batch in pq.ParquetFile(args.canonical_parquet).iter_batches():
        cols = batch.to_pydict()
        for qid, cid, label in zip(cols["query_id"], cols["candidate_id"], cols["label"]):
            label = int(label)
            forward_labels.setdefault(qid, {})[cid] = label
            target_stats = per_target.setdefault(qid, {"actives": 0, "inactives": 0})
            if label == 1:
                target_stats["actives"] += 1
                num_pos += 1
            else:
                target_stats["inactives"] += 1
                num_neg += 1
            num_edges += 1

    tokenization = _token_meta(args)

    forward_payload = {
        "name": "UPIR_strict_forward",
        "dim": args.dim,
        "pockets": [
            {
                "pid": row["protein_id"],
                "text": row["text"],
                "text_kind": row["text_kind"],
            }
            for row in protein_rows
        ],
        "ligands": [
            {
                "lid": row["ligand_id"],
                "text": row["text"],
                "text_kind": row["text_kind"],
                "scaffold": row["scaffold"],
            }
            for row in ligand_rows
        ],
        "labels": forward_labels,
        "meta": {
            "benchmark_type": "upir_strict_forward",
            "source": "LIT-PCBA",
            "label_regime": "strict",
            "query_entity": "protein_pocket",
            "candidate_entity": "ligand",
            "tokenization": tokenization,
            "counts": {
                "num_queries": len(protein_rows),
                "num_candidates": len(ligand_rows),
                "num_known_pairs": num_edges,
                "num_pos": num_pos,
                "num_neg": num_neg,
            },
            "protocol_path": args.forward_protocol_path,
            "target_stats": per_target,
            "note": "Compact-text UPIR strict benchmark built from full local LIT-PCBA pools.",
        },
    }

    forward_counts = dict(forward_payload["meta"]["counts"])
    _write_json(args.out_forward_json, forward_payload)

    del forward_payload
    reverse_labels: Dict[str, Dict[str, int]] = {}
    for batch in pq.ParquetFile(args.canonical_parquet).iter_batches():
        cols = batch.to_pydict()
        for qid, cid, label in zip(cols["query_id"], cols["candidate_id"], cols["label"]):
            reverse_labels.setdefault(cid, {})[qid] = int(label)

    reverse_payload = {
        "name": "UPIR_strict_reverse",
        "dim": args.dim,
        "pockets": [
            {
                "pid": row["ligand_id"],
                "text": row["text"],
                "text_kind": row["text_kind"],
            }
            for row in ligand_rows
        ],
        "ligands": [
            {
                "lid": row["protein_id"],
                "text": row["text"],
                "text_kind": row["text_kind"],
                "scaffold": row["protein_cluster"],
            }
            for row in protein_rows
        ],
        "labels": reverse_labels,
        "meta": {
            "benchmark_type": "upir_strict_reverse",
            "source": "LIT-PCBA",
            "label_regime": "strict",
            "query_entity": "ligand",
            "candidate_entity": "protein_pocket",
            "tokenization": tokenization,
            "counts": {
                "num_queries": len(ligand_rows),
                "num_candidates": len(protein_rows),
                "num_known_pairs": num_edges,
                "num_pos": num_pos,
                "num_neg": num_neg,
            },
            "note": "Compact-text reverse UPIR strict benchmark. Reverse training/evaluation still requires role-aware modeling care.",
        },
    }

    _write_json(args.out_reverse_json, reverse_payload)
    reverse_counts = dict(reverse_payload["meta"]["counts"])

    print(
        json.dumps(
            {
                "out_forward_json": args.out_forward_json,
                "out_reverse_json": args.out_reverse_json,
                "forward_counts": forward_counts,
                "reverse_counts": reverse_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
