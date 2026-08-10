# HWRT / Detexify re-audit result: valid source, remain blocked

Audited 2026-08-10 from the official Zenodo record. Archive integrity and row parsing pass, but the published split, time representation, privacy fields, and ODbL operating requirements prevent immediate product-training approval.

## Source and integrity

- Source: HWRT / Detexify handwritten mathematical symbols, 2015-01-28 release
- Zenodo: https://zenodo.org/records/50022
- Licence: ODbL 1.0
- Local source: `raw/2015-01-28-data.tar`
- Bytes: `140790596`
- MD5: `2BF1D089CE65C0A39E57064516F1BD1C` (matches the Zenodo file record)
- SHA-256: `B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A`
- TAR members: `symbols.csv`, `train-data.csv`, and `test-data.csv`; all members read successfully.

## Observed structure

- 369 unique mathematical-symbol IDs.
- 151,159 train rows and 17,074 test rows; every row parses as `symbol_id;user_id;strokes;user_agent`.
- Each stroke stores ordered `{x, y, time}` points. Samples contain 1-43 strokes and 1-1,475 points.
- Symbol-level train/test counts exactly match `symbols.csv`.
- Train contains 463 user IDs; test contains 118; 103 user IDs occur in both splits.
- Eight exact duplicate rows occur inside train, and two exact records occur in both train and test.
- Time bases are mixed: 1,379 train and 161 test samples use relative millisecond-like values, while the rest use epoch-millisecond-like values.
- 1,186 point-to-point time reversals were observed across the full corpus.
- `user_id` and full browser `user_agent` strings are present.

## Blocking reasons

1. The official split is not writer-disjoint and contains exact overlap.
2. Mixed time bases and time reversals require a documented canonicalization policy.
3. User IDs and user-agent fingerprints require privacy minimization and review.
4. ODbL attribution/share-alike obligations need a written derivative-database and model-release operating plan.
5. The corpus supplies isolated symbols, not formula grouping, stroke ownership, or spatial-relation supervision.

## Release gate

- Strip unnecessary user-agent data and replace user IDs with audit-safe writer keys.
- Normalize time with an explicit per-sample policy and rejection log.
- Rebuild writer-disjoint splits with exact-duplicate isolation.
- Approve the ODbL attribution/share-alike plan before any model training or release.
- Limit any approved use to the box-local mathematical-symbol candidate model and pass all adoption gates.

Decision: **BLOCKED - technically readable and mathematically relevant, but not admitted to 1.0 training.**
