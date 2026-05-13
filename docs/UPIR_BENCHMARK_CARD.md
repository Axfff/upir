# UPIR Benchmark Card

## Summary

`UPIR` means `Unified Pool Interactor Retrieval`.

The benchmark family currently has two materialized forward tracks and one reverse expansion path:

- `UPIR-Strict forward`: strict inactive-aware semantics from `LIT-PCBA`, small but clean.
- `UPIR_open_bigbind_forward`: benchmark-scale breadth from full `BigBind`, large and structure-aware but open/mixed-label.
- `UPIR_open_bigbind_reverse`: materialized ligand-query / protein-candidate track for target-proposal candidate generation.

This split is intentional. The strict track carries label-semantic credibility; the BigBind track carries scale and breadth.
The BigBind track should now be treated primarily as a first-stage candidate-generation benchmark and secondarily as an open-world structure-aware discriminative stress test.

## Current Family

### `UPIR-Strict forward`

- benchmark artifact: `UPIR_strict_forward`
- protocol artifact: `UPIR_strict_forward_protocol.json`
- queries: `15`
- candidates: `407983`
- labeled pairs: `2808770`
- positives: `10033`
- negatives: `2798737`
- unique scaffold ids: `99332`

### `UPIR_open_bigbind_forward`

- benchmark artifact: `UPIR_open_bigbind_forward`
- protocol artifact: `UPIR_open_bigbind_forward_protocol.json`
- queries: `1107`
- candidates: `399090`
- labeled pairs: `582017`
- positives: `488877`
- negatives: `93140`
- unique scaffold ids: `146876`
- primary evaluable query rule: at least `1` known positive and `1` known negative within the evaluation pool

### `UPIR_open_bigbind_reverse`

- source audit artifact: `results/upir_open_bigbind_reverse_audit.json`
- benchmark artifact: `UPIR_open_bigbind_reverse`
- protocol artifact: `UPIR_open_bigbind_reverse_protocol.json`
- query entity: ligand
- candidate entity: protein pocket
- source graph: transposed `UPIR_open_bigbind_forward`
- ligand queries in pool: `399090`
- protein candidates: `1107`
- known labeled pairs: `582017`
- positives: `488877`
- negatives: `93140`
- candidate-generation query rule: at least `1` known positive protein
- candidate-generation evaluable queries: `347207`
- discriminative `1+/1-` evaluable queries: `21476`
- completed candidate-generation baselines: `results/upir_open_bigbind_reverse_candidate_generation_standard_v1/summary.json`
- completed degree-slice diagnostic: `results/upir_open_bigbind_reverse_degree_slices_v1/summary.json`
- completed discriminative diagnostic: `results/upir_open_bigbind_reverse_discriminative_standard_v1/summary.json`
- recommended status: use as candidate-generation track; keep discriminative reverse as auxiliary diagnostic

## Shared Task Definition

- forward query entity: protein pocket
- forward candidate entity: ligand
- reverse query entity: ligand
- reverse candidate entity: protein pocket
- retrieval pool: one unified candidate pool shared by all queries in the same direction
- scoring rule: evaluate ranking only on known labels

Track-specific label regime:
- `UPIR-Strict`: explicit actives and inactives only
- `UPIR_open_bigbind_forward`: BigBind-provided binary activity flags, not strict confirmatory negatives

## Two Evaluation Modes

`UPIR` now supports two valid evaluation modes for unified-pool retrieval.

### 1. `candidate_generation_retrieval`

- goal: measure whether a method can surface plausible positives early for downstream pruning
- query rule: at least `1` known positive in the evaluation pool; known negatives are not required
- primary metrics:
  - forward: `Hit@10`, `Hit@50`, `Hit@100`, `Recall@10`, `Recall@50`, `Recall@100`, `Recall@1000`, best positive rank
  - reverse: `Hit@1`, `Hit@5`, `Hit@10`, `Hit@50`, `Recall@1`, `Recall@5`, `Recall@10`, `Recall@50`, best positive rank
- use case: first-stage screening where false positives are acceptable if known positives are not missed early

### 2. `discriminative_retrieval`

- goal: measure whether a method can rank known positives ahead of known negatives
- query rule: at least `1` known positive and at least `1` known negative in the evaluation pool
- primary metrics:
  - strict track: `MRR`, `Recall@10`, `Recall@50`, `EF1`, `EF5`
  - open track: `EF1`, `EF5`, `Recall@10`, `Recall@50`
- use case: benchmark-style ranking claims, shortcut resistance, and negative-aware evaluation

Recommended interpretation:
- candidate-generation retrieval is the primary mode for the paper's real-world first-stage screening story.
- `UPIR-Strict` should remain the main inactive-aware discriminative anchor.
- `UPIR_open_bigbind_forward` is the natural current place to report both modes.
- `UPIR_open_bigbind_reverse` should be introduced first as reverse candidate generation and only secondarily as discriminative reverse retrieval.
- candidate-generation mode should not be sold as a strict specificity benchmark.

Completed official table on `UPIR_open_bigbind_forward`, `target_rec_cluster`, `5` folds:
- query rule `1+/0-` raises mean fold coverage from about `0.828` in discriminative mode to `0.961` in candidate-generation mode
- `RANDOM`: `Hit@10 = 0.0177`, `Hit@50 = 0.0498`, `Recall@10 < 0.0001`, `Recall@50 = 0.0001`
- `POPULARITY`: `Hit@10 = 0.1773`, `Hit@50 = 0.1874`, `Recall@10 = 0.0125`, `Recall@50 = 0.0231`
- `B1`: `Hit@10 = 0.0072`, `Hit@50 = 0.0333`, `Recall@10 < 0.0001`, `Recall@50 = 0.0002`
- interpretation: the mode is real and higher-coverage, but early recovery over the full shared pool is still nontrivial and should not be misread as a strict specificity benchmark

## Split Protocol

- `standard`: deterministic `5`-fold pair-hash split with all entities visible at evaluation
- `target`: deterministic `5`-fold held-out-query split
- `target_rec_cluster`: deterministic `5`-fold held-out-receptor-cluster split on the BigBind track
- `scaffold`: deterministic `5`-fold held-out-scaffold split
- `joint_ood`: paired held-out-query plus held-out-scaffold split
- recommended strict-track primary OOD table: `target`
- recommended open BigBind primary protein-OOD table: `target_rec_cluster`
- recommended open BigBind primary headline slice: `target_rec_cluster` plus a stricter evaluable-query threshold such as `>=10` known negatives per query
- recommended aggregation: macro over evaluation queries, then mean over folds

## Benchmark Statistics

### Strict Track

- positives per query: min `13`, median `102.0`, max `7168`, mean `668.9`
- positive rate per query: min `0.01%`, median `0.23%`, max `4.94%`, mean `0.74%`
- positive proteins per ligand in the strict graph: min `0`, median `0.0`, max `3`, mean `0.025`
- target-fold evaluation queries: `[3, 3, 3, 3, 3]`
- target-fold evaluation positives: `[7198, 935, 664, 632, 604]`
- scaffold-fold candidate counts: `[81597, 81596, 81597, 81597, 81596]`
- scaffold-fold query coverage with at least one positive eval ligand: `[15, 15, 15, 15, 15]`

### Open BigBind Track

- positives per query: min `0`, median `72.0`, max `8268`, mean `441.6`
- positive rate per query: min `0.00%`, median `87.50%`, max `100.00%`, mean `76.08%`
- target-fold evaluation queries: `[223, 221, 221, 221, 221]`
- target-fold evaluable queries: `[181, 178, 185, 184, 181]`
- target-rec-cluster-fold evaluation queries: `[268, 246, 192, 216, 185]`
- target-rec-cluster-fold evaluable queries: `[185, 216, 166, 182, 160]`
- scaffold-fold candidate counts: `[79818, 79818, 79818, 79818, 79818]`
- scaffold-fold evaluable queries: `[701, 680, 669, 683, 670]`
- `target_rec_cluster` `5`-fold official baseline means:
  - `RANDOM`: `MRR=0.8702`, `EF1=1.0365`, `EF5=1.0264`
  - `POPULARITY`: `MRR=0.8508`, `EF1=1.0355`, `EF5=1.0465`
  - `PROT_KNN_POP`: `MRR=0.8394`, `EF1=1.0260`, `EF5=1.0703`
  - `PROT_CHEM_CENTROID`: `MRR=0.8414`, `EF1=1.0219`, `EF5=1.0337`
- hard-slice coverage on `target_rec_cluster`:
  - `>=10` known negatives/query: `0.5561`
  - `>=20` known negatives/query: `0.4174`
  - `>=50` known negatives/query: `0.2415`

## Recommended Metrics

- strict track:
  - discriminative mode: `MRR`, `Recall@10`, `Recall@50`, `EF1`, `EF5`
  - candidate-generation mode: `Recall@10`, `Recall@50`, `Hit@10`, `Hit@50`
- open BigBind track:
  - discriminative mode primary: `EF1`, `EF5`, `Recall@10`, `Recall@50`
  - discriminative mode secondary / diagnostic: `MRR`
  - candidate-generation mode primary: `Recall@10`, `Recall@50`, `Hit@10`, `Hit@50`
  - recommended discriminative headline reporting should prefer a harder query slice, not the raw full open-track table

## What Makes This Benchmark Valuable

- It matches the project’s actual retrieval claim instead of reducing the task to small per-target candidate lists.
- It uses a unified candidate pool, which is the right operating regime for retrieval methods.
- It ships fixed folds, so results become reproducible and comparable across methods.
- It now separates strict semantics from benchmark-scale breadth instead of pretending one source can do both perfectly.
- It can now support an explicit first-stage screening story: the benchmark asks whether known interactors are recovered early enough for downstream docking, affinity prediction, or expert triage.
- It can still support a negative-aware stress-test story, but that story is auxiliary to candidate generation.

## Current Limitations

- The current benchmark family now has a materialized open reverse track, but only lightweight heuristic baselines are complete; learned reverse baselines remain future work.
- The strict anchor still contains only `15` queries, so it cannot carry broad generalization claims alone.
- The BigBind track has benchmark-scale breadth, but its label geometry is highly positive-heavy; raw full-table `MRR` is partly saturated and should not be a headline metric.
- The learned BigBind results on `target_rec_cluster` are negative for the current method story: `B1` beats `M1` on discriminative completed folds, but full-pool candidate-generation `B1` remains below simple open-track heuristics on early-hit probability.
- The current full-table `scaffold` setting is close to random and should be treated as a diagnostic negative result, not as a headline generalization result.
- The current scaffold field is chemistry-grade Bemis-Murcko, but acyclic ligands fall into a shared `NO_MURCKO` bucket and coverage should be reported explicitly.
- Protein leakage control is now stronger on the BigBind track via `target_rec_cluster`, but still not site-aware.

## Reviewer-Safe Positioning

- Valid claim now: `UPIR` is a two-mode benchmark family with a strict semantic anchor and a large-scale open BigBind forward track.
- Valid claim now: `UPIR_open_bigbind_forward` solves the old small-query bottleneck and supports large-scale unified-pool evaluation as an open-world structure-aware stress test.
- Valid claim now: candidate-generation retrieval is the primary first-stage screening mode; discriminative retrieval is a separate specificity/calibration mode.
- Valid claim now: `UPIR` can support both discriminative retrieval evaluation and candidate-generation retrieval evaluation, provided the two modes are reported separately.
- Valid claim now: transposing the open BigBind graph supports reverse candidate generation, with `347207` full-graph positive ligand queries and `1107` protein candidates.
- Valid claim now: the materialized open reverse track has a completed lightweight `5`-fold candidate-generation baseline table and degree-sliced popularity diagnostic.
- Not valid yet: `UPIR` fully solves bidirectional retrieval benchmarking with learned methods, because open reverse currently has heuristic baselines only.
- Not valid yet: the BigBind track is a strict confirmatory-negative benchmark.
- Not valid yet: the raw full BigBind table is a clean, highly discriminative ranking benchmark across standard metrics.
- Not valid yet: the raw BigBind scaffold split supports a strong scaffold-generalization story.
- Not valid yet: the large-pool reverse track is learned-baseline complete or release-final; it currently supports heuristic candidate-generation baselines and degree-sliced diagnostics.

## Next Upgrade Path

- Keep degree-stratified reporting for `UPIR_open_bigbind_reverse` because reverse protein popularity is strong.
- Add at least one learned reverse candidate-generation anchor if compute budget permits.
- Treat the completed learned `B1` candidate-generation anchor as a negative reference point; stronger learned candidate-generation methods are future work.
- Use reverse discriminative diagnostics only as appendix/specificity calibration.
- Keep scaffold and hard-negative slices as diagnostics unless a stricter evaluable-query slice becomes more discriminative.
- Add stronger family/site leakage audits for proteins.

## Warnings

- `UPIR-Strict` target count is still small; report fold-level uncertainty and avoid over-claiming held-out-target generalization.
- `UPIR_open_bigbind_forward` primary ranking tables should report only evaluable queries with both known positives and known negatives in the evaluation pool.
- BigBind positive rates remain highly imbalanced across queries; macro averaging is required.
- For the BigBind track, raw full-table `MRR` should be treated as diagnostic rather than headline because random and simple heuristics already achieve high values.
- Acyclic ligands fall into the `NO_MURCKO` bucket under chemistry-grade scaffold extraction; coverage should be reported explicitly.
