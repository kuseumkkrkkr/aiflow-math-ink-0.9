# Project symbol-head calibration - 2026-08-12

Status: accepted as a research classifier candidate; `product_adopted` remains
false. This is a box-local output-head calibration, not a formula recognizer.

## Scope and split

- Base: the fixed two-epoch external-only checkpoint selected in
  `CHARACTER_CLASSIFIER_TUNING_20260812.md` (372 math outputs and 95 auxiliary
  outputs).
- Project data: 211 manually ownership-grounded symbol trajectories from 47
  formulas, across three already hashed writer groups. It contains 25 observed
  math labels, including 6 `(`, 6 `)`, and 20 `=` samples.
- Validation: leave one project writer group out. The other groups calibrate;
  the held group is evaluated once. The fixed 18,723-row external holdout is
  never calibration input.
- Finalization trains on all real project symbol rows only after the
  writer-disjoint result passes. It deliberately does not score those same
  project rows afterward.

## Method

The script is `scripts/calibrate_project_punctuation_v1.py`. It keeps the
encoder, auxiliary head, and every non-project math output row byte-identical.

1. Calibrate only `(`, `)`, and `=` for 250 class-balanced steps (batch 16).
2. Calibrate the remaining observed project math outputs for 250
   class-balanced steps (batch 32), while leaving the first-stage rows fixed.
3. Each step also uses external **training** rehearsal: 216 UJI parenthesis
   rows plus a deterministic 8,192-row non-punctuation math sample. The first
   stage weights rehearsal loss 8:1; the second weights it 16:1.

No raw archive, canonical derivative, formula grouping, ownership grouping,
or fixed holdout row is modified or used as calibration input.

## Writer-disjoint result

| Metric | External-only base | Two-stage calibration |
|---|---:|---:|
| Project direct Top-1 (211 pooled held-writer rows) | 36.49% | **64.93%** |
| Project direct Top-5 | 54.50% | **87.20%** |
| Direct `(`, `)`, `=` Top-1 | 0/32 | **28/32** |
| Direct `(`, `)`, `=` Top-5 | 0/32 | **32/32** |
| Other direct symbols Top-1 | 77/179 | **109/179** |
| Other direct symbols Top-5 | 115/179 | **152/179** |

The direct Top-1 gain is 28.44 percentage points. In paired writer-disjoint
comparison, the base alone was correct on 9 rows and calibration alone on 69;
the exact two-sided McNemar p-value is `1.381e-12`.

The weakest calibration fold still improved external math Top-1 from 57.64%
to 57.99%; all three external full-holdout folds were 60.01%-60.15% Top-1,
above the 59.70% base. The planned gates therefore pass: direct Top-1 gain is
at least 5 points and no fold loses more than 1 external-math point.

## Final research checkpoint

`artifacts/project_punctuation_calibration_v1/final_symbol_head_20260812/`
contains the local-only final checkpoint. Its external fixed-holdout result is
60.03% Top-1 / 90.34% Top-5, versus 59.70% / 90.17% for the base.

Only 25 math-head weight rows and the corresponding 25 bias rows changed;
the encoder, auxiliary head, and every other math output row are bitwise
unchanged. Direct ownership performance is intentionally marked `not_scored`
for that final checkpoint because all 211 project rows supplied calibration
labels.

## Remaining gate

This is not commercial acceptance: there are only three writer groups, the
same small project collection selected the two-stage recipe, and no independent
new project-writer holdout remains after finalization. Collect new writers for
a post-finalization regression set, then separately implement and validate
stroke grouping, spatial relations, and formula decoding.

## Verification

- The calibration script self-test, classifier self-test, tensor self-test,
  and replay-evaluator self-test pass.
- The final checkpoint loads into the 372/95 architecture. A byte comparison
  confirms that only its 25 declared math-head rows and biases differ from the
  external-only base.
- `build_normalized_ink_v1.py --verify-only` reports 189,334 records,
  351,140 strokes, and 28,054,726 points.
- `verify_datasets.ps1` verifies 21 D: artifacts (17 hashes and four audited
  large-file lengths), five ZIP catalogs, six GZIP containers, one TAR, one
  HWRT manifest, and nine Parquet files in quick mode.
