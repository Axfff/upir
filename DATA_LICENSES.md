# Data Licenses and Redistribution Policy

UPIR is released under a reconstruction-first policy. The code is MIT-licensed, but the benchmark artifacts are derived from upstream scientific datasets with their own terms.

## Recommended Release Mode

The public repository should include:

- source code;
- benchmark construction scripts;
- fixed protocol-generation logic;
- benchmark cards;
- paper source;
- result summaries and checksums where they do not expose restricted upstream content;
- Slurm/manual execution helpers.

The public repository should not include by default:

- raw upstream archives such as `LIT-PCBA_full.tar` or `BigBindV1.5.tar.gz`;
- full derived JSON benchmark artifacts containing upstream-derived labels, SMILES, structures, or sequences;
- large ranking CSVs and encoding caches.

Users should obtain upstream resources from their original sources and reconstruct local UPIR artifacts with the scripts in this repository.

## Upstream Components

| UPIR component | Upstream source | Public release policy |
|---|---|---|
| `UPIR-Strict` forward/reverse | LIT-PCBA / PubChem-derived assays | Reconstruction scripts and summaries only by default; verify source-level redistribution before shipping derived artifacts |
| `UPIR-Open-BigBind` forward | BigBind/BayesBind, derived from ChEMBL/CrossDocked-style resources | Reconstruction scripts and summaries only by default; respect upstream attribution/share-alike terms |
| `UPIR-Open-BigBind` reverse | Transposed open BigBind graph | Same policy as open BigBind forward |
| Protocol metadata | This work, derived from benchmark construction | Releasable with attribution; use conservative metadata licensing if it exposes upstream-derived IDs/labels |
| Code and Slurm scripts | This work | MIT license |

## Practical Guidance

If full derived artifacts are later redistributed, use a separate data archive and include:

- exact upstream versions;
- checksums of upstream archives and generated artifacts;
- source citations;
- attribution text;
- license compatibility notes;
- a statement of changes made by UPIR;
- share-alike terms where required by upstream data.

Until that verification is complete, prefer reconstruction over redistribution.

## Suggested Paper Wording

> We release the UPIR code, protocol definitions, benchmark cards, result summaries, and deterministic reconstruction scripts. Because UPIR derives from upstream resources with their own licenses, the public release follows a reconstruction-first policy: users obtain LIT-PCBA and BigBind/BayesBind from their original sources, then run the provided scripts to regenerate the benchmark artifacts. Derived protocol metadata, checksums, and reference results are archived with the code release.

