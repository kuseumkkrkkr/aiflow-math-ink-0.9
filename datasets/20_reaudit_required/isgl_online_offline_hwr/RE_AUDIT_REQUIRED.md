# ISGL re-audit required

## Source claim

- Source: ISGL Online and Offline Character Recognition Dataset
- URL: https://data.mendeley.com/datasets/n7kmd7t7yx/1
- Claimed license: CC BY 4.0
- Claimed fields: 64 writers; number of strokes, pen up/down, X/Y, and write time.

## Why this is blocked

1. The inherited artifact is a normalized JSONL derivative, not a verified original source archive.
2. Its rows mark `missing_timestamp=true`; the stored timing is canonicalized rather than verified per-point source timing.
3. Existing 0.9 commercial allowlist excluded this source, so there is no prior product-use approval to inherit.
4. CC BY 4.0 needs source-file provenance, attribution text, contributor-consent/biometric-risk review, and a check for third-party content before product training.
5. The dataset is English character/word HWR, not formula ownership or relation supervision.

## Required release gate

- Retrieve and hash the original download successfully.
- Compare original sample counts and fields with the migrated JSONL.
- Record attribution and any contributor/privacy restrictions.
- Approve its specific training role in the 1.0 commercial registry.
- Keep it out of train/validation/model-selection manifests until every gate above is signed off.

## Retrieval note

The public source's `Download All` endpoint issued a temporary signed URL that failed to complete within six minutes on 2026-08-10. Its incomplete 14.8 MB response is named `raw/isgl_source.zip.partial-invalid` and must not be consumed. The migrated artifact is retained only for inspection; it is not an approval substitute.
