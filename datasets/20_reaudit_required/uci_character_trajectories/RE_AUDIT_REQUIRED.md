# UCI Character Trajectories re-audit result: structurally valid, role blocked

Audited 2026-08-10 from the official UCI download. Provenance, archive integrity, and numeric structure pass, but the dataset remains outside product training until a narrowly scoped experiment passes the project adoption gates.

## Source and integrity

- Source: UCI Machine Learning Repository, dataset 175
- URL: https://archive.ics.uci.edu/dataset/175/character+trajectories
- Licence: CC BY 4.0
- Official local source: `raw/character+trajectories.official.zip`
- Bytes: `7915950`
- SHA-256: `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52`
- ZIP CRC: passed.
- The official download is byte-identical to the inherited `raw/character_trajectories.zip`.

## Observed structure

- `mixoutALL_shifted.mat`: 2,858 arrays, each shaped `3 x T` for X, Y, and pen-tip force.
- `T` ranges from 109 to 205; all values are finite.
- 20 lowercase Latin classes: `a b c d e g h l m n o p q r s u v w y z`.
- All samples come from one writer and retain only one pen-down segment.
- Sampling interval is documented as 0.005 seconds (200 Hz).
- The released values were numerically differentiated, Gaussian-smoothed, normalized, and time-shifted. They are not raw device coordinates.

## Blocking reasons

1. One writer cannot support writer-disjoint training or model selection.
2. Single pen-down samples cannot supervise multi-stroke grouping, ownership, or formula layout.
3. The transformed signals do not match AIFlow's raw ordered-stroke contract.
4. Scientific usefulness is limited to an optional representation-pretraining ablation; usefulness is not established by the permissive licence alone.

## Release gate

- Register CC BY 4.0 attribution.
- Use only in an isolated single-stroke representation experiment.
- Evaluate on a held-out project-owned writer split and retain only if all character, ownership, relation, and end-to-end formula gates are non-regressive.
- Never use it to train ownership, grouping, relation, or final-decision components.

Decision: **BLOCKED FROM 1.0 TRAINING - archive valid, optional ablation only after explicit approval.**
