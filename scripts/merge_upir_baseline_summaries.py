#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge split-specific UPIR baseline summaries into one combined summary.")
    p.add_argument("--inputs", required=True, help="Comma-separated summary.json paths")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    paths = [x.strip() for x in args.inputs.split(",") if x.strip()]
    merged = None
    for path in paths:
        obj = _load(path)
        if merged is None:
            merged = {
                "dataset_name": obj["dataset_name"],
                "dataset_path": obj["dataset_path"],
                "folds": obj["folds"],
                "models": [],
                "results": {},
                "splits": set(),
            }
        merged["models"] = sorted(set(merged["models"]) | set(obj["models"]))
        for split_name, split_payload in obj["results"].items():
            merged["results"][split_name] = split_payload
            merged["splits"].add(split_name)
    merged["splits"] = sorted(merged["splits"], key=lambda x: ["standard", "target", "scaffold", "joint_ood"].index(x))
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out)}, indent=2))


if __name__ == "__main__":
    main()
