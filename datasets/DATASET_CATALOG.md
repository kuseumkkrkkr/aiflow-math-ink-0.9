# AIFlow Math Ink 1.0 dataset catalog

Last verified: 2026-08-10

| Folder | Dataset | Status | Permitted role | Local artifact |
|---|---|---|---|---|
| `00_project_owned/` | AIFlow consented public ink | Project-owned | Primary online HWR, ownership, and formula-layout training after label/consent gate | Existing release data remains in `hf-dataset/data/`. |
| `10_approved_external/uji_pen_characters_v2/` | UJI Pen Characters v2 | Approved external | Isolated-character online HWR pretraining only | Source ZIP and curated derivative |
| `20_reaudit_required/isgl_online_offline_hwr/` | ISGL | Blocked | None | Original ZIP and inherited normalized JSONL |
| `20_reaudit_required/uci_character_trajectories/` | UCI Character Trajectories | Blocked | Optional single-stroke ablation only after explicit gate | Official source ZIP |
| `20_reaudit_required/hwrt/` | HWRT / Detexify | Blocked | None until ODbL, privacy, time, duplicate, and writer-split gates pass | Official Zenodo TAR |
| `20_reaudit_required/bdshwa/` | BDSHWA | Blocked/high risk | None; demographic or identity inference prohibited | Original Mendeley ZIP |
| `20_reaudit_required/hf_cli_unlicensed/` | Three `newbienewbie` trajectory repos | Rejected | Audit evidence only | D-pinned Hugging Face CLI downloads |
| `30_noncommercial_evaluation/` | CROHME/MathWriting-class data | Research-only | Frozen external regression evaluation only | Never copy into product-training folders |

## Integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `10_approved_external/uji_pen_characters_v2/raw/uji_pen_v2.zip` | - | `0881B522911B99D9922820289441B50FD3D307F71CD7F9CC70E86872424A5F90` |
| `10_approved_external/uji_pen_characters_v2/derived/uji_math_curated.jsonl.gz` | - | `CDA2FE17C213BC8C90ED93EC8998658DDAF9D7871C75ADF521798E6E3BA3029B` |
| `20_reaudit_required/isgl_online_offline_hwr/raw/isgl_source.zip` | 974,053,582 | `A94F2473246222F9470D4B93B68CFBC756ECB4865742DB8164788359FE511693` |
| `20_reaudit_required/isgl_online_offline_hwr/migrated/isgl_online.jsonl.gz` | - | `0A4BD4FE37CD7656B2FD3586FED4D5C886FEBA1D8E5120803B60A0F04044988D` |
| `20_reaudit_required/uci_character_trajectories/raw/character_trajectories.zip` | 7,915,950 | `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52` |
| `20_reaudit_required/uci_character_trajectories/raw/character+trajectories.official.zip` | 7,915,950 | `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52` |
| `20_reaudit_required/hwrt/raw/2015-01-28-data.tar` | 140,790,596 | `B96FEAFD71B01F1623997DFF3CC8AC4D18628D128CEE1B3DF1880518BBA3EA4A` |
| `20_reaudit_required/bdshwa/raw/bdshwa_v1.zip` | 1,318,392,171 | `45FBBDCD1A9C4353F5C93BF8096FBEE9723931185F7BBFED3A86E21BA4CF20EA` |

## Admission rules

1. A readable archive is not a training approval.
2. Only `10_approved_external` and explicitly consented `00_project_owned` data may enter a training manifest.
3. External isolated-character corpora may train only the box-local character candidate model. They cannot train stroke ownership, grouping, spatial relations, or the final decision layer without labels and a separate approval.
4. Every candidate must pass provenance/licence, privacy/consent, schema/integrity, writer-disjoint evaluation, project holdout, and CROHME regression gates.
