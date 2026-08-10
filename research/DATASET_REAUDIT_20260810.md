# AIFlow Math Ink 1.0 dataset re-audit - 2026-08-10

## Outcome

All working files, source archives, Hugging Face downloads, caches, and audit dependencies were kept on `D:`. The four source datasets were downloaded and opened successfully, but none of the re-audit candidates was promoted into the commercial training tier.

| Dataset | Integrity | HWR fit | Commercial/privacy gate | Decision |
|---|---|---|---|---|
| ISGL | Outer ZIP and online RAR extraction pass | X/Y strokes; no per-point time; isolated English chars/words | CC BY 4.0, but consent/privacy scope absent and inherited derivative is incomplete | Block |
| UCI Character Trajectories | Official ZIP CRC and all 2,858 arrays pass | One writer, one pen-down, transformed signals | CC BY 4.0; role/evaluation mismatch remains | Block pending ablation gate |
| HWRT | Zenodo MD5, TAR, 168,233 JSON rows, and symbol counts pass | Mathematical symbol strokes with X/Y/time | ODbL plan, user-agent minimization, split leakage, mixed time bases | Block |
| BDSHWA | Outer and both nested ZIP CRCs pass; stratified trajectory sample passes | Rich online ink but Bengali/English forensic tasks, not math | CC BY 4.0 plus stated consent, but biometric/demographic and metadata risks need independent review | Block/high risk |

## Hugging Face CLI repeat search

Environment:

```powershell
$env:HF_HOME='D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0\research\hf_home'
hf version
hf datasets list --search 'handwriting_strokes' --limit 20 --format quiet
hf datasets list --search 'MathWriting' --limit 20 --format quiet
hf datasets list --search 'CROHME' --limit 20 --format quiet
```

- CLI version: 1.24.0.
- Exact searches for `online handwriting trajectory`, `digital ink handwriting`, `handwriting strokes coordinates`, and `InkML handwriting math` returned no repositories.
- Name searches found three small coordinate-sequence repositories and MathWriting/CROHME mirrors.
- MathWriting and CROHME remain noncommercial/research-only sources; mirrors cannot broaden upstream rights.
- The three coordinate repositories were downloaded to `datasets/20_reaudit_required/hf_cli_unlicensed/`, verified against their Hub revisions, and rejected for missing licence/provenance plus split leakage. See that folder's audit note.

## D-only control

- The Hugging Face project cache is `research/hf_home/` on D:.
- DuckDB and Python audit dependencies are under `research/python_deps/`, with D: pip and bytecode caches.
- Three initially created project-specific Hub cache directories under `C:\Users\user\.cache\huggingface` were moved to the D: project cache; they were not copied or left behind.
- Download and audit temporary material used `D:\AIFlow-Workspace\Temp\aiflow-math-ink-audit-20260810` and was removed after the audit.

## Reproducibility

- `scripts/download_range_file.ps1` performs resumable D-only range downloads with segment-length and final-length checks.
- `scripts/verify_datasets.ps1` verifies artifact expectations and container readability. Quick mode checks audited large-file lengths; `-Full` recomputes every SHA-256.
- Source binaries remain Git-ignored; hashes, revisions, counts, and decisions are versioned.
