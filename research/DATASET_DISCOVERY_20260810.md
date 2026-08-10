# AIFlow Math Ink 1.0 online-HWR dataset discovery

Searched 2026-08-10 through Kaggle CLI, Google-indexed primary dataset pages, UCI, Mendeley Data, Zenodo/Hugging Face result cards, CASIA/TUAT pages, and AI Hub.

## Immediate 1.0 candidates

| Dataset | Ink fields / writers | Rights state | 1.0 action |
|---|---|---|---|
| UJI Pen Characters v2 | X/Y strokes; 60 writers; two sessions | CC BY 4.0 | Staged as approved isolated-character pretraining data. |
| Project-owned Math Ink | Original points, order, sensor fields, target cells/relations | Project-owned subject to consent | Primary online HWR, grouping, ownership, and formula-relation source. |

## Re-audit candidates

| Dataset | Why relevant | Blocking condition |
|---|---|---|
| ISGL Online/Offline HWR | 64 writers; pen up/down, X/Y, aggregate stroke duration | Original was verified, but privacy/consent scope, 571-row derivative loss, stale eligibility, and writer split remain blocked. |
| UCI Character Trajectories | X/Y/force time series; CC BY 4.0 | Preprocessed single-stroke data from one writer; representation-only role needs approval. |
| HWRT / Detexify | X/Y/time; user ID and user-agent; mathematical symbols | ODbL operations, privacy minimization, mixed time bases, duplicate isolation, and writer-disjoint split required. |
| BDSHWA | Wacom digital ink; 29 writers; structured English/Bengali tasks; CC BY 4.0 | Audit found demographic/biometric risk, participant-ID collision, modality/schema drift, and no mathematical supervision. |
| CASIA online handwriting | Very large online XY/stroke corpus with writer coverage | Commercial use requires direct licensing/permission; do not download or train before written terms. |
| TUAT Nakagawa Lab online DB | Licensed commercial-use route exists | Paid license and explicit procurement/usage agreement required. |

## Rejected for 1.0 product training

| Source | Reason |
|---|---|
| CROHME / Kaggle mirrors | Online InkML is structurally ideal but license is noncommercial. |
| MathWriting | Online formula data but CC BY-NC-SA; research-only. |
| AI Hub handwriting/OCR data | Found data is image/OCR oriented rather than online ink, and AI Hub policy requires separate agreement for commercial use; do not ingest as 1.0 HWR data. |
| Kaggle ArabicMath2LaTeX | CC BY claim but raster images only; outside online-HWR core path. |

## Data-gap decision

No externally found dataset can replace project-owned ink for formula grouping, stroke ownership, or spatial relations. Use external corpora only for the explicitly listed box-local character representation role. Fill operator, multi-stroke, and formula-layout gaps with consented project-owned online ink and transformations that preserve stroke order and ownership.

## 2026-08-10 re-audit disposition

- Original ISGL, UCI Character Trajectories, HWRT, and BDSHWA artifacts were downloaded to D: and independently opened, hashed, and structurally audited.
- Only UJI Pen Characters v2 remains approved external training data.
- ISGL, UCI Character Trajectories, HWRT, and BDSHWA remain blocked for the reasons recorded in each dataset folder.
- Three additional Hugging Face trajectory repositories were downloaded and rejected because they lack licence/provenance and have severe split leakage.
- Representative 2022-2025 online-HWR/HMER papers did not reveal a new downloadable, commercially clear data class. Paper searching stopped at that saturation point; see `HWR_PAPER_DATASET_REVIEW_20260810.md`.

## Source links

- UCI UJI Pen Characters v2: https://archive.ics.uci.edu/dataset/177/uji%2Bpen%2Bcharacters%2Bversion%2B2
- UCI Character Trajectories: https://archive.ics.uci.edu/dataset/175/character%2Btrajectories
- ISGL: https://data.mendeley.com/datasets/n7kmd7t7yx/1
- HWRT: https://www.martin-thoma.de/write-math/data/
- BDSHWA: https://data.mendeley.com/datasets/99t9jhvksv/1
- CASIA online handwriting: https://nlpr.ia.ac.cn/databases/handwriting/home.html
- TUAT online handwriting terms: https://web.tuat.ac.jp/~nakagawa/database/en/kondate_proc.html
- AI Hub policy: https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
