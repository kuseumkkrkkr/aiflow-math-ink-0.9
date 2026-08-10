# AIFlow Math Ink 1.0 dataset catalog

Last verified: 2026-08-10

| Folder | Dataset | Status | Permitted role | Local artifact |
|---|---|---|---|---|
| `00_project_owned/` | AIFlow consented public ink | project-owned | Primary online HWR and formula-layout training after label/consent gate | Existing release data remains in `hf-dataset/data/`; it is not duplicated here. |
| `10_approved_external/uji_pen_characters_v2/` | UJI Pen Characters v2 | Approved external | Isolated-character online HWR pretraining only | `raw/uji_pen_v2.zip`, migrated curated subset |
| `20_reaudit_required/isgl_online_offline_hwr/` | ISGL Online/Offline Character Recognition | Blocked pending re-audit | None before written approval | Migrated online JSONL; see `RE_AUDIT_REQUIRED.md` |
| `20_reaudit_required/uci_character_trajectories/` | UCI Character Trajectories | Blocked pending re-audit | None before written approval | Source ZIP; see `RE_AUDIT_REQUIRED.md` |
| `30_noncommercial_evaluation/` | CROHME/MathWriting-class data | Research-only | Frozen external regression evaluation only | Do not copy into product-training folders. |

## Integrity

| Artifact | SHA-256 |
|---|---|
| `10_approved_external/uji_pen_characters_v2/raw/uji_pen_v2.zip` | `0881B522911B99D9922820289441B50FD3D307F71CD7F9CC70E86872424A5F90` |
| `10_approved_external/uji_pen_characters_v2/derived/uji_math_curated.jsonl.gz` | `CDA2FE17C213BC8C90ED93EC8998658DDAF9D7871C75ADF521798E6E3BA3029B` |
| `20_reaudit_required/isgl_online_offline_hwr/migrated/isgl_online.jsonl.gz` | `0A4BD4FE37CD7656B2FD3586FED4D5C886FEBA1D8E5120803B60A0F04044988D` |
| `20_reaudit_required/uci_character_trajectories/raw/character_trajectories.zip` | `5D2DB017EF0D8CF0E65ED060C9E90399F78EB9F1E3CB63E22CA8C3EF4BA67D52` |

## Layout-contract rule

External isolated-character corpora may train only the box-local character candidate model. They must not train stroke ownership, formula grouping, spatial-relation, or final-decision components unless the source supplies and passes audit for those labels.
