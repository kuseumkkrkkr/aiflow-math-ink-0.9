# Project symbol-head calibration - 2026-08-12

Status: accepted as a research classifier candidate. `product_adopted` remains
false. This is a box-local output-head calibration, not a formula recognizer.

## Scope and split

- Base: the controlled three-epoch external-only checkpoint in
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
  rows afterward.

## Method

`scripts/calibrate_project_punctuation_v1.py` keeps the encoder, auxiliary
head, and every non-project math output row byte-identical.

1. Calibrate only `(`, `)`, and `=` for 250 class-balanced steps (batch 16).
2. Calibrate the remaining observed project math outputs for 250
   class-balanced steps (batch 32), leaving first-stage rows fixed.
3. Each step uses external training rehearsal: 216 UJI parenthesis rows plus a
   deterministic 8,192-row non-punctuation math sample. Rehearsal loss weights
   are 8:1 for the first stage and 16:1 for the second.

The runtime exposes the calibration learning rate to make the selected setting
reproducible. No raw archive, canonical derivative, formula grouping,
ownership grouping, or fixed holdout row is modified or used as calibration
input.

## Writer-disjoint selection

| Candidate | External base Top-1 / Top-5 | Project direct LOO Top-1 / Top-5 | Decision |
|---|---:|---:|---|
| Former 2-epoch base + calibration `1e-3` | 59.70% / 90.17% | 64.93% / 87.20% | historical baseline |
| 3-epoch base + calibration `1e-3` | 66.43% / 94.25% | 64.93% / 85.78% | reject: weaker Top-5 |
| **3-epoch base + calibration `5e-4`** | **66.43% / 94.25%** | **64.93% / 88.15%** | **selected** |

The selected candidate preserves the 137/211 writer-disjoint Top-1 hits and
raises Top-5 from 184/211 to 186/211 versus the former calibrated baseline.
It improves raw external-only direct performance from 77/211 to 81/211 Top-1
plus a better rank distribution (115/211 to 123/211 Top-5); the calibrated
Top-1 gain over its own 3-epoch base is 26.54 points. The weakest calibration
fold loses only 0.24 external-math Top-1 points, within the 1-point gate; all
three writer-disjoint folds pass the direct gain gate.

## Final research checkpoint

`artifacts/training_length_20260812/project_symbol_3ep_final_lr5e-4/`
contains the current local-only final checkpoint. On the fixed external
holdout it reaches 66.67% Top-1 / 94.02% Top-5, versus 66.43% / 94.25% for the
uncalibrated three-epoch base. Math punctuation is 24/24 Top-1 and Top-5.

Only 25 math-head weight rows and their bias rows can change; the encoder,
auxiliary head, and every other math output row are asserted byte-identical.
Direct ownership is intentionally `not_scored` for this final checkpoint
because all 211 rows supplied calibration labels.

## Remaining gate

This is not commercial acceptance: the three project writer groups selected
the calibration setting and no independent new project-writer holdout remains.
`a` and `f` have only one writer group (four rows total). Collect new writers
for a post-finalization regression set, then separately implement and validate
stroke grouping, spatial relations, and formula decoding.

## Verification

- Both calibration modes complete with the frozen external cache and no raw
  dataset mutation; the selected LOO report passes both research gates.
- The calibration, classifier, tensor, and replay evaluator self-tests pass.
- Dataset normalization and archive verification are re-run in verify-only
  mode after the code change.
