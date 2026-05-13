#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write source audit CSV for UPIR benchmark construction.")
    p.add_argument("--litpcba-root", default="data/external_raw/LIT-PCBA_full")
    p.add_argument("--bigbind-root", default="data/external_raw/BigBind")
    p.add_argument("--plinder-root", default="data/external_raw/PLINDER")
    p.add_argument("--scope-root", default="data/external_raw/SCOPE-DTI")
    p.add_argument("--out-csv", default="data/real_benchmark_sources/upir_source_audit.csv")
    return p.parse_args()


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _litpcba_row(root: Path) -> Dict[str, str]:
    if not root.exists():
        return {
            "source": "LIT-PCBA",
            "local_status": "missing",
            "root_path": str(root),
            "benchmark_role": "strict_forward_core",
            "explicit_inactives": "yes",
            "structure_available": "yes",
            "reverse_task_support": "indirect_only",
            "num_targets": "",
            "num_positive_edges": "",
            "num_negative_edges": "",
            "notes": "Local raw package missing.",
        }

    targets = 0
    pos = 0
    neg = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        actives = d / "actives.smi"
        inactives = d / "inactives.smi"
        if not actives.exists() or not inactives.exists():
            continue
        targets += 1
        pos += _count_lines(actives)
        neg += _count_lines(inactives)

    return {
        "source": "LIT-PCBA",
        "local_status": "available",
        "root_path": str(root),
        "benchmark_role": "strict_forward_core",
        "explicit_inactives": "yes",
        "structure_available": "yes",
        "reverse_task_support": "indirect_only",
        "num_targets": str(targets),
        "num_positive_edges": str(pos),
        "num_negative_edges": str(neg),
        "notes": "Best strict inactive-aware source for UPIR-Strict v0.",
    }


def _missing_row(source: str, root: Path, role: str, notes: str) -> Dict[str, str]:
    return {
        "source": source,
        "local_status": "available" if root.exists() else "missing",
        "root_path": str(root),
        "benchmark_role": role,
        "explicit_inactives": "unknown",
        "structure_available": "unknown",
        "reverse_task_support": "unknown",
        "num_targets": "",
        "num_positive_edges": "",
        "num_negative_edges": "",
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, str]] = [
        _litpcba_row(Path(args.litpcba_root)),
        _missing_row(
            source="BigBind",
            root=Path(args.bigbind_root),
            role="open_world_forward_scale",
            notes="Planned source for UPIR-Open; local data not integrated in this round.",
        ),
        _missing_row(
            source="PLINDER",
            root=Path(args.plinder_root),
            role="structure_split_control",
            notes="Planned source for leak-aware structure coverage; local data not integrated in this round.",
        ),
        _missing_row(
            source="SCOPE-DTI",
            root=Path(args.scope_root),
            role="reverse_direction_reference",
            notes="Planned reverse-direction reference source; local data not integrated in this round.",
        ),
    ]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print({"out_csv": str(out), "rows": len(rows)})


if __name__ == "__main__":
    main()
