# Character classifier tuning and epoch selection - 2026-08-13

Status: the current external-only base is the selected seven-epoch run. It is
a research-only, box-local classifier: `product_adopted` is false and no
formula-exact score is claimed. The earlier five-epoch all-writer calibration
remains the current local final because the new seven-epoch writer-LOO result
does not surpass it.

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
| 5 | 74.75% / 96.87% |
| 6 | 75.33% / 97.17% |
| **7** | **75.99% / 97.36%** |
| 8 | 74.02% / 97.21% |

Seven epochs is therefore selected before the final full-data retrain. Epoch
8 loses 1.97 Top-1 points from epoch 7 on the same internal slice, so it is
not a candidate for continued training. The selection run is a decision
artifact only; its checkpoint is not a final evaluation model.

## One-time final evaluation after selection

The selected five epochs were retrained once using all external training rows,
then evaluated on the untouched fixed holdout.

| External-only checkpoint | Fixed external Top-1 / Top-5 | Raw project ownership Top-1 / Top-5 |
|---|---:|---:|
| Three epochs (former base) | 66.43% / 94.25% | 38.39% / 58.29% |
| Five epochs | 74.10% / 96.57% | 35.07% / 55.45% |
| **Seven epochs (current external-only base)** | **76.02% / 96.96%** | **45.02% / 65.88%** |

Against the five-epoch checkpoint, the seven-epoch external Top-1 gain is
1.61 points and Top-5 gain is 0.29 points; raw project ownership also rises
9.95 / 10.43 points before calibration. Project ownership is not used for
external-training selection. The new base has zero missing external
predictions, with HWRT 75.48% / 96.81%, UJI 79.03% / 97.58%, ISGL 75.98% /
98.17%, and UCI 96.04% / 100.00%.

The local-only seven-epoch base checkpoint is
`artifacts/training_length_20260813/external_7ep_selected_full/classifier_checkpoint.pt`.
Its math/auxiliary training losses are 2.9012/2.3419, 0.9871/1.0211,
0.7348/0.7030, 0.6313/0.5729, 0.5623/0.5128, 0.5168/0.4578, and
0.4814/0.4341.

## Data, imbalance, and character-correction audit

`character_classifier_data_audit_20260813/CHARACTER_CLASSIFIER_DATA_AUDIT.md`
records the current checkpoint diagnostics. All 128 x 5 cache tensors pass
the shape, finite-value, range, and stroke-boundary checks. UCI still has zero
remaining adjacent exact-XY duplicates after canonicalization.

- HWRT has 167,363 examples over 369 classes (range 51..3,512). The rarest
  math-support band reaches 75.25% fixed-holdout Top-1 versus 79.55% for the
  highest-support band. Retain sampler-only balancing; do not add loss weights
  or a learned rebalancer.
- 38 train and 3 evaluation math inputs spatially collapse, concentrated in
  point-like labels. The 13 exact tensor-collision groups include visually
  ambiguous `\\bullet`, `\\cdot`, `\\dotsc`, and `\\vdots`; preserve labels and
  route their ultimate distinction to grouping/context, rather than silently
  merging or deleting source rows.
- `=` remains absent from external training and no synthetic example is
  admitted. The only permitted correction is writer-disjoint, output-row
  calibration using real project symbols; fixed-holdout confusions never
  expand that correction set.

The seven-epoch writer-LOO calibration passes its research gates: direct
Top-1 rises 45.02% -> 64.93%, punctuation rises 0/32 -> 22/32 Top-1, and the
worst external-math Top-1 regression is 0.20 points. It is still below the
five-epoch calibrated 66.35% / 91.00% result, so it is recorded as a passing
research candidate only and does not replace the current five-epoch final
calibrated checkpoint.

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
