#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a release-grade benchmark card for UPIR.")
    p.add_argument("--protocol-json", default="data/real_benchmarks/upir/UPIR_strict_forward_protocol.json")
    p.add_argument("--stats-json", default="data/real_benchmarks/upir/UPIR_strict_forward_stats.json")
    p.add_argument("--out-md", default="data/real_benchmarks/upir/UPIR_BENCHMARK_CARD.md")
    return p.parse_args()


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol_json)
    stats = _load_json(args.stats_json)

    counts = stats["counts"]
    qstats = stats["query_positive_count_stats"]
    rstats = stats["query_positive_rate_stats"]
    dstats = stats["ligand_positive_degree_stats"]
    health = stats["protocol_health"]
    scaffold_semantics = stats.get("scaffold_semantics", protocol["splits"]["scaffold"].get("scaffold_semantics", "proxy_string_scaffold"))
    scaffold_limitation = (
        "- The current scaffold field is chemistry-grade Bemis-Murcko, but acyclic ligands fall into a shared `NO_MURCKO` bucket and coverage should be reported explicitly."
        if scaffold_semantics == "bemis_murcko"
        else "- The current scaffold field is a proxy string scaffold, not a chemistry-grade Bemis-Murcko scaffold."
    )
    scaffold_claim_line = (
        "- Not valid yet: scaffold generalization is chemistry-grade and broadly validated."
        if scaffold_semantics == "bemis_murcko"
        else "- Not valid yet: scaffold generalization is chemistry-grade."
    )
    upgrade_path_line = (
        "- Strengthen scaffold reporting with coverage and per-fold diagnostics under Murcko semantics."
        if scaffold_semantics == "bemis_murcko"
        else "- Upgrade scaffold semantics with chemistry tooling."
    )

    lines = [
        "# UPIR Benchmark Card",
        "",
        "## Summary",
        "",
        "`UPIR` means `Unified Pool Interactor Retrieval`.",
        "",
        "`UPIR-Strict` is the strict-label track for the explicit task: given a protein pocket, retrieve interacting ligands from one large unified ligand pool. The current release is built from the full local `LIT-PCBA` package and packages the task as a benchmark rather than a per-target screening script.",
        "",
        "## Current Release",
        "",
        f"- benchmark artifact: `UPIR_strict_forward`",
        f"- protocol artifact: `{Path(args.protocol_json).name}`",
        f"- queries: `{counts['num_queries']}`",
        f"- candidates: `{counts['num_candidates']}`",
        f"- labeled pairs: `{counts['num_known_pairs']}`",
        f"- positives: `{counts['num_pos']}`",
        f"- negatives: `{counts['num_neg']}`",
        f"- unique scaffold ids: `{counts['num_unique_scaffolds']}`",
        "",
        "## Task Definition",
        "",
        "- query entity: protein pocket",
        "- candidate entity: ligand",
        "- retrieval pool: one unified ligand pool shared by all queries",
        "- label regime: strict binary labels using explicit actives and inactives only",
        "- scoring rule: evaluate ranking only on known labels",
        "",
        "## Split Protocol",
        "",
        "- `standard`: deterministic `5`-fold pair-hash split with all entities visible at evaluation",
        "- `target`: deterministic `5`-fold held-out-query split",
        "- `scaffold`: deterministic `5`-fold held-out-scaffold split",
        "- `joint_ood`: paired held-out-query plus held-out-scaffold split",
        "- recommended primary OOD table: `target`",
        "- recommended aggregation: macro over evaluation queries, then mean over folds",
        "",
        "## Benchmark Statistics",
        "",
        f"- positives per query: min `{qstats['min']:.0f}`, median `{qstats['median']:.1f}`, max `{qstats['max']:.0f}`, mean `{qstats['mean']:.1f}`",
        f"- positive rate per query: min `{_pct(rstats['min'])}`, median `{_pct(rstats['median'])}`, max `{_pct(rstats['max'])}`, mean `{_pct(rstats['mean'])}`",
        f"- positive proteins per ligand in the strict graph: min `{dstats['min']:.0f}`, median `{dstats['median']:.1f}`, max `{dstats['max']:.0f}`, mean `{dstats['mean']:.3f}`",
        f"- target-fold evaluation queries: `{health['target_fold_eval_queries']}`",
        f"- target-fold evaluation positives: `{health['target_fold_eval_pos']}`",
        f"- scaffold-fold candidate counts: `{health['scaffold_fold_candidates']}`",
        f"- scaffold-fold query coverage with at least one positive eval ligand: `{health['scaffold_fold_queries_with_positive_eval']}`",
        "",
        "## Recommended Metrics",
        "",
        "- `MRR`",
        "- `Recall@10`",
        "- `Recall@50`",
        "- `EF1`",
        "- `EF5`",
        "",
        "## What Makes This Benchmark Valuable",
        "",
        "- It matches the project’s actual retrieval claim instead of reducing the task to small per-target candidate lists.",
        "- It uses a unified candidate pool, which is the right operating regime for retrieval methods.",
        "- It ships fixed folds, so results become reproducible and comparable across methods.",
        "- It cleanly separates strict-label evaluation from future open-world extensions.",
        "",
        "## Current Limitations",
        "",
        "- The current strict release is forward-direction only for main claims. Reverse-direction artifacts exist, but they should not be sold as benchmark-ready yet.",
        "- The source still contains only `15` protein-pocket queries, so target-level uncertainty remains high.",
        scaffold_limitation,
        "- Protein leakage control is at the target-identity level, not yet at the family or binding-site-cluster level.",
        "",
        "## Reviewer-Safe Positioning",
        "",
        "- Valid claim now: `UPIR-Strict` is a release-grade unified-pool strict retrieval benchmark with fixed protocol and leakage-aware held-out splits.",
        "- Not valid yet: `UPIR` fully solves bidirectional retrieval benchmarking.",
        scaffold_claim_line,
        "- Not valid yet: reverse retrieval is large-pool and publication-ready.",
        "",
        "## Next Upgrade Path",
        "",
        upgrade_path_line,
        "- Expand reverse-direction protein pool with additional public sources.",
        "- Add `UPIR-Open` with masked unknowns rather than forced negatives.",
        "- Add stronger family/site leakage audits for proteins.",
        "",
    ]

    if stats.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in stats["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_md": str(out)}, indent=2))


if __name__ == "__main__":
    main()
