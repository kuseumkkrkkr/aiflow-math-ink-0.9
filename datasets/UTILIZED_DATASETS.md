# AIFlow Math Ink 1.0 utilized datasets

Last updated: 2026-08-10

## Current checkpoint evidence

| Dataset | Current use | Evidence boundary |
|---|---|---|
| AIFlow project-owned public ink | Current 0.9 calibrator training and replay evaluation | 110 valid formulas are eligible; the ownership training subset is 47 formulas / 211 symbols. Pending 4 and rejected 5 are excluded. |
| CROHME2019 valid | Noncommercial regression evaluation only | 985 parseable formulas; never product training or product-performance evidence. |

The current 0.9 checkpoint has not yet been retrained on the newly approved external pool below.

## Approved 1.0 training pool

| Dataset | Available scope | Planned model role | Explicit limit |
|---|---|---|---|
| UJI Pen Characters v2 | 11,640 isolated characters, 60 writers, two sessions | Box-local online character trajectory pretraining | No time, pressure, formula relation, or layout labels |
| ISGL | 7,985 characters plus 3,790 words from 64 source writer IDs | English uppercase/lowercase, digits, and word-level trajectory expansion | Online data only; no per-point time; inherited derivative has 7,414 character rows and is training-only |
| UCI Character Trajectories | 2,858 samples, 20 lowercase Latin labels, one writer | Single-stroke representation pretraining | Treat as one-writer training data; never validation or model selection |
| HWRT / Detexify curated | 168,027 samples, 369 math-symbol labels, 317,996 strokes, 25,393,124 points | Box-local mathematical-symbol candidate pretraining | 193 reversing samples and 13 normalized duplicates removed; source user IDs are not reliable writer identities, so no model selection/final evaluation |
| BDSHWA | 1,348 raw trajectory CSV files across 29 participant folders | Future non-formula online HWR and English/Bengali text expansion | Exclude all demographic, identity, age, gender, and forensic-biometric targets and metadata |

## Not used for product training

- Three Hugging Face `newbienewbie` trajectory datasets: rejected for missing licence/provenance and split leakage.
- CROHME and MathWriting-class corpora: noncommercial research evaluation only.
- BDSHWA processed biometric feature tables and participant metadata: excluded even though raw trajectories are approved for the restricted HWR role.

## Layer boundary

Within the current Math Ink model, external isolated-character or text corpora may improve only the box-local stroke encoder and character Top-k candidates. Formula grouping, stroke ownership, spatial relations, and final decisions remain supervised by project-owned formula data and validated through the existing holdout/regression gates.

BDSHWA and ISGL word-level material may later support a separate general text-HWR branch. That future branch must keep its sequence/text objectives separate from mathematical formula layout and decision responsibilities.
