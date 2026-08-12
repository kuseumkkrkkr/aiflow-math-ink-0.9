# Character classifier tuning and epoch selection - 2026-08-13

Status: the current external-only base is the selected five-epoch run. It is a
research-only, box-local classifier: `product_adopted` is false and no
formula-exact score is claimed.

## Fixed scope

- Architecture is unchanged: 128 x 5 online-ink tensor, four Transformer
  blocks, hidden size 128, 372-class math head, and 95-class auxiliary head.
- Optimizer settings are fixed: seed `20260812`, AdamW, batch size 64,
  sampler-only balancing, learning rate `3e-4`, weight decay `1e-2`, and
  gradient clipping 1.
- External corpora train only the box-local representation. Project data is
  not external-model input; formula grouping, relations, decoding, CROHME, and
  canonical formula replay remain outside scope.
- Raw archives and canonical derivatives are unchanged. All trial artifacts
  remain local under ignored `artifacts/` paths.

## Historical two-epoch learning-rate decision

| Candidate | External Top-1 / Top-5 | Decision |
|---|---:|---|
| `3e-4`, sampler + loss weights | 42.58% / 82.49% | reject: double frequency correction |
| `1e-4`, sampler + loss weights | 21.98% / 57.18% | reject: too slow in two epochs |
| `1e-3`, sampler + loss weights | 0.09% / 0.51% | reject: optimization collapse |
| `3e-4`, sampler only + UJI parentheses to math head | 59.70% / 90.17% | former two-epoch base |
| former base + synthetic `=` from HWRT `-` | 47.94% / 82.66% | reject and remove |

The synthetic `=` path remains absent. Real project symbols are admitted only
through the separate writer-disjoint output-head calibration.

## Leakage-controlled epoch selection

The earlier 3-epoch run still had falling losses, so epoch count was selected
without reusing the fixed external holdout. A deterministic 10% source-label
slice was removed from the existing external training cache:

- math: 136,074 training and 14,931 selection rows;
  selection-index SHA-256 `d589699119714f002c91b9b8f5866d1c531407265f617e4d1ef7cf3c85ee7883`;
- auxiliary: 17,664 training and 1,832 selection rows;
  selection-index SHA-256 `5d8bb9e0a250f975dbc8ea5656170cc274216b54e33975947a38c9a0498617c3`;
- seed `20260813`, retaining at least one row from every source+label group
  in the fitting split.

The fixed 18,723-row external holdout and project ownership data were both
`not_scored` during this selection. The metric is all-glyph Top-1, then Top-5,
with the earliest epoch winning an exact tie.

| Epoch | Internal selection Top-1 / Top-5 |
|---:|---:|
| 1 | 29.64% / 64.08% |
| 2 | 60.08% / 91.77% |
| 3 | 68.29% / 95.10% |
| 4 | 72.77% / 96.15% |
| **5** | **74.75% / 96.87%** |

Five epochs is therefore selected before the final full-data retrain. The
selection run is a decision artifact only; its checkpoint is not a final
evaluation model.

## One-time final evaluation after selection

The selected five epochs were retrained once using all external training rows,
then evaluated on the untouched fixed holdout.

| External-only checkpoint | Fixed external Top-1 / Top-5 | Raw project ownership Top-1 / Top-5 |
|---|---:|---:|
| Three epochs (former base) | 66.43% / 94.25% | 38.39% / 58.29% |
| **Five epochs (current base)** | **74.10% / 96.57%** | 35.07% / 55.45% |

The external Top-1 gain is 7.67 points and Top-5 gain is 2.32 points. Raw
project ownership decreases before calibration, so it is not used as an
external-training selection metric; the independent writer-LOO calibration
result is documented separately. The current base has zero missing external
predictions, with HWRT 73.41% / 96.31%, UJI 78.43% / 98.02%, ISGL 74.86% /
98.74%, and UCI 94.96% / 100.00%.

The local-only base checkpoint is
`artifacts/training_length_20260812/external_5ep_selected_full/classifier_checkpoint.pt`.
Its five math/auxiliary training losses are 2.9012/2.3419, 0.9871/1.0211,
0.7348/0.7030, 0.6313/0.5729, and 0.5623/0.5128.

## Remaining boundary

- `a` and `f` have one project writer group only (four rows total); collect
  new-writer examples before making a robust direct-generalization claim.
- The final external-only base is not a commercial or formula-recognition
  claim. New writer holdout, stroke grouping, spatial relations, and decoding
  remain independent gates.

## Verification

- The selection runner records
  `aiflow-character-classifier-selection/v1`; it never scores the fixed
  external holdout or project ownership set.
- The full-data result records
  `aiflow-character-classifier-external-training/v1` and has `status` of
  `completed`.
- Classifier, calibration, tensor, and replay self-tests are re-run after the
  code change. Normalization and archive verification are verify-only.
