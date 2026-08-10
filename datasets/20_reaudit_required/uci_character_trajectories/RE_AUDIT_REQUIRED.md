# UCI Character Trajectories re-audit required

## Source claim

- Source: UCI Machine Learning Repository, dataset 175
- URL: https://archive.ics.uci.edu/dataset/175/character%2Btrajectories
- License: CC BY 4.0
- Structure: 2,858 single-character samples from one writer; Wacom-derived X/Y/pen-force trajectories captured at 200 Hz.

## Why this is blocked

1. The source data was differentiated, Gaussian-smoothed, normalized, and shifted before release; it is not raw device ink.
2. Only one pen-down segment is retained per character. It cannot supervise multi-stroke grouping, ownership, or formula layout.
3. It contains a single writer, so it cannot support writer-disjoint model selection.
4. Existing 0.9 commercial allowlist excluded this source; 1.0 needs an explicit scope approval despite CC BY 4.0.

## Required release gate

- Verify the ZIP contents and documented preprocessing against the source page.
- Register attribution and a narrowly scoped role: optional single-stroke representation pretraining only.
- Demonstrate no regression against a held-out project-owned writer split before use.
- Do not use it for ownership, relation, formula-level, or final-decision training.
