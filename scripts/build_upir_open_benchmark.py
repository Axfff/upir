#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a forward UPIR open/mixed-label benchmark from canonical tables.")
    p.add_argument("--canonical-parquet", required=True)
    p.add_argument("--proteins-parquet", required=True)
    p.add_argument("--ligands-parquet", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--protocol-path", default="")
    p.add_argument("--benchmark-name", default="UPIR_open_forward")
    p.add_argument("--source", default="BayesBindV1.5")
    p.add_argument("--label-regime", default="mixed_assay_plus_decoy")
    p.add_argument("--benchmark-type", default="upir_open_forward")
    p.add_argument("--note", default="Compact-text UPIR open benchmark built from BayesBind/BigBind benchmark bundles.")
    p.add_argument("--min-known-pos", type=int, default=1)
    p.add_argument("--min-known-neg", type=int, default=0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--smiles-kmer", type=int, default=3)
    p.add_argument("--residue-kmer", type=int, default=2)
    p.add_argument("--max-smiles-tokens", type=int, default=64)
    p.add_argument("--max-pocket-residues", type=int, default=128)
    return p.parse_args()


def _read_pylist(path: str) -> List[Dict]:
    return pq.read_table(path).to_pylist()


def _token_meta(args: argparse.Namespace) -> Dict:
    return {
        "dim": args.dim,
        "kinds": {
            "protein_residue": {"k": args.residue_kmer, "max_tokens": args.max_pocket_residues, "base_seed": 17},
            "smiles": {"k": args.smiles_kmer, "max_tokens": args.max_smiles_tokens, "base_seed": 701},
        },
    }


def main() -> None:
    args = parse_args()
    protein_rows = _read_pylist(args.proteins_parquet)
    ligand_rows = _read_pylist(args.ligands_parquet)

    labels: Dict[str, Dict[str, int]] = {}
    per_target: Dict[str, Dict[str, int]] = {}
    num_edges = 0
    num_pos = 0
    num_neg = 0
    num_assay_neg = 0
    num_decoy_neg = 0
    num_other_neg = 0

    for batch in pq.ParquetFile(args.canonical_parquet).iter_batches():
        cols = batch.to_pydict()
        for qid, cid, label, label_source in zip(cols["query_id"], cols["candidate_id"], cols["label"], cols["label_source"]):
            label = int(label)
            labels.setdefault(qid, {})[cid] = label
            stats = per_target.setdefault(
                qid,
                {"actives": 0, "inactives": 0, "assay_inactives": 0, "decoy_negatives": 0, "other_negatives": 0},
            )
            if label == 1:
                stats["actives"] += 1
                num_pos += 1
            else:
                stats["inactives"] += 1
                num_neg += 1
                if label_source == "assay_inactive":
                    stats["assay_inactives"] += 1
                    num_assay_neg += 1
                elif label_source == "random_decoy":
                    stats["decoy_negatives"] += 1
                    num_decoy_neg += 1
                else:
                    stats["other_negatives"] += 1
                    num_other_neg += 1
            num_edges += 1

    payload = {
        "name": args.benchmark_name,
        "dim": args.dim,
        "pockets": [{"pid": row["protein_id"], "text": row["text"], "text_kind": row["text_kind"]} for row in protein_rows],
        "ligands": [{"lid": row["ligand_id"], "text": row["text"], "text_kind": row["text_kind"], "scaffold": row["scaffold"]} for row in ligand_rows],
        "labels": labels,
        "meta": {
            "benchmark_type": args.benchmark_type,
            "source": args.source,
            "label_regime": args.label_regime,
            "query_entity": "protein_pocket",
            "candidate_entity": "ligand",
            "primary_evaluable_query_rule": {
                "min_known_pos": int(args.min_known_pos),
                "min_known_neg": int(args.min_known_neg),
            },
            "tokenization": _token_meta(args),
            "counts": {
                "num_queries": len(protein_rows),
                "num_candidates": len(ligand_rows),
                "num_known_pairs": num_edges,
                "num_pos": num_pos,
                "num_neg": num_neg,
                "num_assay_neg": num_assay_neg,
                "num_decoy_neg": num_decoy_neg,
                "num_other_neg": num_other_neg,
            },
            "protocol_path": args.protocol_path,
            "target_stats": per_target,
            "note": args.note,
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"out_json": str(out), "counts": payload["meta"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
