# Character classifier tuning and training-length follow-up - 2026-08-12

Status: the fixed-split external classifier candidate is now the controlled
three-epoch run. It remains a research-only box-local classifier:
`product_adopted` is false and no formula-exact metric is claimed.

## Scope and fixed protocol

- Architecture and input are unchanged: 128 x 5 online-ink tensor, four
  Transformer blocks, hidden size 128, 372-class math head, and 95-class
  auxiliary head.
- All comparisons use seed `20260812`, AdamW, batch size 64, gradient clip 1,
  weight decay `1e-2`, sampler-only balancing, learning rate `3e-4`, and the
  fixed 18,723-row external holdout. Project data is not external-model input.
- Formula-exact CROHME2019 and canonical replay remain unscored because stroke
  grouping, relations, and a formula decoder are outside this classifier.
- Raw archives and canonical derivatives were not changed. Trial outputs stay
  under ignored `artifacts/` paths.

## Fixed two-epoch tuning (historical)

| Candidate | External Top-1 / Top-5 | Direct ownership Top-1 / Top-5 | Decision |
|---|---:|---:|---|
| `3e-4`, sampler + loss weights (control) | 42.58% / 82.49% | 25.12% / 43.60% | historical control |
| `1e-4`, sampler + loss weights | 21.98% / 57.18% | 27.01% / 45.50% | too slow in two epochs |
| `1e-3`, sampler + loss weights | 0.09% / 0.51% | 3.79% / 3.79% | reject: optimization collapse |
| `3e-4`, sampler only | 57.93% / 89.50% | 33.18% / 52.61% | removes double balancing |
| `5e-4`, sampler only | 57.83% / 89.41% | 37.91% / 57.35% | not selected |
| `3e-4`, sampler only + UJI parentheses to math head | 59.70% / 90.17% | 36.49% / 54.50% | former two-epoch base |
| former base + synthetic `=` from HWRT `-` | 47.94% / 82.66% | 30.81% / 48.34% | reject and remove |

The paired two-epoch `3e-4` versus `5e-4` sampler-only comparison was not
decisive (external McNemar two-sided p = `0.776855`; direct p = `0.132498`).
The synthetic `=` path remains absent: real, consented project ink is used
only by the separate writer-disjoint symbol-head calibration.

## Training-length follow-up (current selection)

The former two-epoch run ended with math loss `0.9871` and auxiliary loss
`1.0211`; this did not establish convergence. One controlled extension changed
only epoch count from two to three. The third epoch reduced those losses to
`0.7348` and `0.7030`.

| External-only checkpoint | External Top-1 / Top-5 | Raw project ownership Top-1 / Top-5 |
|---|---:|---:|
| Two epochs (former base) | 59.70% / 90.17% | 36.49% / 54.50% |
| **Three epochs (current base)** | **66.43% / 94.25%** | **38.39% / 58.29%** |

The 3-epoch result has zero missing predictions across the fixed external
holdout. Its per-source external Top-1 / Top-5 is HWRT 65.04% / 93.82%, UJI
75.41% / 96.72%, ISGL 73.88% / 98.03%, and UCI 92.81% / 100.00%.

The current base checkpoint is local-only at
`artifacts/training_length_20260812/external_3ep_sampler_bg/classifier_checkpoint.pt`.
It is paired with the separately validated project-symbol output-head
calibration documented in `PROJECT_SYMBOL_HEAD_CALIBRATION_20260812.md`.

## Remaining boundary

- The external base does not learn `=` from fabricated data. Real project
  symbols are admitted only through writer-disjoint output-head calibration.
- The direct collection has only three writer groups. `a` and `f` have one
  writer group each (four rows total), so they need new-writer collection
  before any robust generalization claim.
- The current result is not a commercial or formula-recognition claim. New
  writer holdout, stroke grouping, spatial relations, and decoding remain
  separate gates.

## Verification

- `train_character_classifier_v1.py --self-test` passes with both the legacy
  two-epoch checkpoint contract and the generic external-training contract.
- The 3-epoch report is `completed`, records the fixed split, and uses schema
  `aiflow-character-classifier-external-training/v1`.
- Tensor, calibration, and replay evaluator self-tests are run after the
  selection. Dataset normalization and archive verification are rechecked
  without mutating raw data.
