# Character classifier tuning loop - 2026-08-12

Status: fixed-split two-epoch tuning is complete. The selected checkpoint is a
research pilot only; it is not product adopted and it does not score formulas.

The selected checkpoint is the external-only base for the later real
project-symbol calibration in `PROJECT_SYMBOL_HEAD_CALIBRATION_20260812.md`.
The `=` limitation below applies to this base checkpoint and cache, not to the
separate writer-disjoint project calibration.

## Controlled protocol

- Every candidate used the same seed (`20260812`), two epochs, 128 x 5 tensor,
  372-class math head, batch size 64, and the fixed 18,723-row external
  holdout. Project-owned ownership data (211 groups / 47 formulas) remained
  evaluation-only.
- Formula-exact CROHME2019 and the ten canonical project replays remain
  unscored because grouping, relation prediction, and a formula decoder are
  outside this box-local classifier.
- Trial outputs and caches remain local under the ignored
  `artifacts/tuning_lr_20260812/` path. No raw dataset was changed.

## Results

| Candidate | External Top-1 / Top-5 | Direct ownership Top-1 / Top-5 | Decision |
|---|---:|---:|---|
| `3e-4`, sampler + loss weights (control) | 42.58% / 82.49% | 25.12% / 43.60% | historical control |
| `1e-4`, sampler + loss weights | 21.98% / 57.18% | 27.01% / 45.50% | too slow in two epochs |
| `1e-3`, sampler + loss weights | 0.09% / 0.51% | 3.79% / 3.79% | reject: optimization collapse |
| `3e-4`, sampler only | 57.93% / 89.50% | 33.18% / 52.61% | removes double balancing |
| `5e-4`, sampler only | 57.83% / 89.41% | 37.91% / 57.35% | not selected |
| `3e-4`, sampler only + UJI parentheses to math head | **59.70% / 90.17%** | **36.49% / 54.50%** | selected pilot |
| selected candidate + synthetic `=` from HWRT `-` | 47.94% / 82.66% | 30.81% / 48.34% | reject and remove |

The paired `3e-4` versus `5e-4` sampler-only comparison was not decisive:
McNemar two-sided p = 0.776855 on the external holdout (2,257 vs 2,237
discordant wins) and p = 0.132498 on direct ownership (13 vs 23). The lower
learning rate is therefore retained because it has the slightly higher fixed
external result; this is not evidence that it is universally better.

## Diagnosed issues and applied changes

1. The original sampler and capped inverse-frequency loss weights both
   corrected class frequency. This double correction overemphasized rare HWRT
   labels and suppressed common labels. The default is now one correction:
   `--balance-mode sampler`, with unit cross-entropy weights.
2. `1e-4` had insufficient progress during the imposed two-epoch budget;
   `1e-3` did not remain useful. The selected default stays AdamW `3e-4`,
   weight decay `1e-2`, gradient clipping 1.0, no scheduler, and no warm-up.
3. UJI has commercially admitted isolated `(` and `)` examples. Those rows
   now train and evaluate the math head instead of a duplicate auxiliary head.
   The compatible cache has 151,005 math training rows (150,789 HWRT + 216
   UJI parentheses), 16,598 math evaluation rows (16,574 + 24), 19,496
   auxiliary training rows, and 2,125 auxiliary evaluation rows.
4. Train-only geometric synthesis of `=` from HWRT `-` was tested and rejected:
   it reduced both aggregate metrics and reached only 0/20 direct Top-1 and
   1/20 direct Top-5 for `=`. The synthetic path is absent from the final code.

## Remaining boundary

- `=` is the only untrained math-head label in the selected cache. It needs
  real, consented, writer-disjoint box-local handwriting rather than a
  fabricated substitute.
- On direct ownership, `(` is 0/6 Top-1 and Top-5, `)` is 0/6, and `=` is
  0/20. The UJI holdout parentheses are useful internal supervision evidence
  (`(` 11/12 Top-1, `)` 4/12 Top-1) but do not demonstrate transfer to the
  project-owned writers.
- The gain is sufficient to keep the pilot configuration for later classifier
  experiments, not to claim commercial readiness or formula recognition.

## Verification

- `train_character_classifier_v1.py --self-test`,
  `character_tensor_v1.py --self-test`, and
  `replay_evaluate_hwr_v1.py --self-test` passed.
- `prepare_cache` accepted the existing v2 cache under the final source
  contract without rebuilding it.
- `build_normalized_ink_v1.py --verify-only` reported 189,334 records,
  351,140 strokes, and 28,054,726 points.
- `verify_datasets.ps1` verified 21 D: artifacts (17 hashes and four audited
  large-file lengths), five ZIP catalogs, six GZIP containers, one TAR, one
  HWRT manifest, and nine Parquet files in quick mode.
