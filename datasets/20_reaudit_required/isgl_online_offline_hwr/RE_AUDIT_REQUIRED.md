# ISGL re-audit result: remain blocked

Audited 2026-08-10 from the original Mendeley Data version 1 download. The archive and online trajectory files are readable, but this dataset is not admitted to AIFlow Math Ink 1.0 training.

## Source and integrity

- Source: ISGL Online and Offline Character Recognition Dataset
- DOI/source: https://doi.org/10.17632/n7kmd7t7yx.1
- Source-page licence: CC BY 4.0
- Local source: `raw/isgl_source.zip`
- Bytes: `974053582`
- SHA-256: `A94F2473246222F9470D4B93B68CFBC756ECB4865742DB8164788359FE511693`
- Outer ZIP CRC: passed for all 12 RAR payloads.
- The four online metadata RAR files extracted successfully with `bsdtar -m`.

The old interrupted downloads and range parts were removed after the complete source passed validation.

## Observed online structure

| Subset | Records | Writers | Representation |
|---|---:|---:|---|
| Uppercase characters | 3,382 | 64 | stroke-separated X/Y text |
| Lowercase characters | 3,314 | 64 | stroke-separated X/Y text |
| Digits | 1,289 | 64 | stroke-separated X/Y text |
| Words | 3,790 | 64 | stroke-separated X/Y text |

- Character total: 7,985 records, 62 labels, 13,739 declared strokes, and 388,148 points.
- Writer IDs are 1-60 and 62-65; ID 61 is absent, yielding the claimed 64 writers.
- Each stroke has X/Y points and an aggregate `TimeTaken`; there is no per-point timestamp. The source therefore does not support true temporal resampling.
- Source-quality findings: 553 character files contain at least one empty stroke, 225 contain at least one missing stroke duration, and 38 contain no non-empty stroke.

## Comparison with inherited normalized data

`migrated/isgl_online.jsonl.gz` contains 7,414 character records. For included records, labels and non-empty-stroke coordinates match the original after the documented empty-stroke removal. It is not a lossless copy:

- 571 original character records are absent.
- Every retained row is assigned to `train`; there is no writer-disjoint validation or test split.
- Every row embeds `eligible_for_training=true`, even though its track says registry approval is pending and the file is located in the blocked tier.
- Timestamps are null and `timestamp_mode=canonical`; only aggregate stroke durations survive.

The embedded eligibility value is stale and must not override the folder/registry decision.

## Blocking reasons

1. The source page does not document participant consent, privacy handling, or commercial biometric-model use beyond the dataset copyright licence.
2. The migrated artifact omits 571 rows and does not record a reproducible exclusion manifest.
3. It has no writer-disjoint model-selection split and currently carries a contradictory eligibility flag.
4. It supplies isolated English characters/words, not mathematical layout, stroke ownership, or relation labels.

## Release gate

- Obtain a written privacy/consent and commercial-use assessment.
- Rebuild from the original archive with a deterministic rejection manifest and `eligible_for_training=false` by default.
- Create writer-disjoint train/validation/test splits.
- Limit any approved role to box-local character candidate pretraining and pass project-owned writer-holdout plus CROHME regression gates.

Decision: **BLOCKED - no training, validation, model selection, or release use.**
