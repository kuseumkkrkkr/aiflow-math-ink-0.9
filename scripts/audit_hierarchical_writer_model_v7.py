#!/usr/bin/env python3
"""v7 raw-writer catalog와 known writer-disjoint replay를 독립 재검증한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_hierarchical_writer_model_v5 import (
    _formula_audit,
    _read_json,
    _read_jsonl,
    _sha256,
)
from build_normalized_ink_v1 import SourceSample, _canonicalize
from character_tensor_v1 import tensorize
from train_character_classifier_v1 import InkClassifierV1


SCHEMA = "aiflow-hierarchical-writer-model-v7-independent-audit/v2"
RAW_DOMAIN = "project_owned_legacy_raw_observed"
EXTERNAL_DOMAIN = "approved_external_cleanroom"


def _source_metadata_path(path: Path) -> Path:
    """외부 NPZ와 짝을 이루는 압축 metadata 경로를 계산한다."""

    return path.with_suffix(".metadata.jsonl.gz")


def _raw_stroke_hash(strokes: list[dict[str, Any]], indices: tuple[int, ...]) -> str:
    """선택된 원시 stroke 본문의 안정 해시를 다시 계산한다."""

    selected = [strokes[index] for index in indices]
    value = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rebuild_legacy_catalog(
    frozen_root: Path,
    report: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], dict[str, Any]]:
    """동결 원본에서 fresh/unowned를 제외하고 legacy glyph를 독립 재구성한다."""

    formulas_path = frozen_root / "frozen_dataset/data/formulas_valid.jsonl"
    ownership_path = frozen_root / "frozen_dataset/data/ownership_train.jsonl"
    fresh_path = frozen_root / "ownership_fresh_acceptance.jsonl"
    info_path = frozen_root / "frozen_dataset/dataset_info.json"
    actual_hashes = {
        "formulas": _sha256(formulas_path),
        "ownership": _sha256(ownership_path),
        "fresh_id_exclusion": _sha256(fresh_path),
        "dataset_info": _sha256(info_path),
    }
    expected_hashes = report["partition"]["hashes"]
    hash_mismatches = {
        key: {"actual": value, "reported": expected_hashes.get(key)}
        for key, value in actual_hashes.items()
        if value != expected_hashes.get(key)
    }

    fresh_ids = {str(row["sample_id"]) for row in _read_jsonl(fresh_path)}
    all_ownership = _read_jsonl(ownership_path)
    ownership = [row for row in all_ownership if str(row["sample_id"]) not in fresh_ids]
    legacy_ids = {str(row["sample_id"]) for row in ownership}
    all_formulas = _read_jsonl(formulas_path)
    formulas = {
        str(row["sample_id"]): row
        for row in all_formulas
        if str(row["sample_id"]) in legacy_ids
    }
    unowned = [
        row for row in all_formulas
        if str(row["sample_id"]) not in fresh_ids and str(row["sample_id"]) not in legacy_ids
    ]

    features: list[np.ndarray] = []
    tokens: list[str] = []
    provenance: list[dict[str, Any]] = []
    for annotation in ownership:
        sample_id = str(annotation["sample_id"])
        source = formulas.get(sample_id)
        if not annotation.get("accepted") or source is None:
            raise ValueError(f"invalid legacy ownership/formula join: {sample_id}")
        if str(source["writer_id"]) != str(annotation["writer_id"]):
            raise ValueError(f"legacy writer mismatch: {sample_id}")
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        if [int(row["order"]) for row in strokes] != list(range(len(strokes))):
            raise ValueError(f"non-contiguous legacy stroke order: {sample_id}")
        groups = [tuple(int(value) for value in group) for group in annotation["groups"]]
        labels = [str(value) for value in annotation["labels"]]
        targets = [str(cell["token"]) for cell in source.get("target_cells") or []]
        assigned = [index for group in groups for index in group]
        if labels != targets or sorted(assigned) != list(range(len(strokes))) or len(assigned) != len(set(assigned)):
            raise ValueError(f"legacy ownership contract mismatch: {sample_id}")
        for group_index, (indices, token) in enumerate(zip(groups, labels, strict=True)):
            raw_strokes = [
                [(float(point["x"]), float(point["y"]), float(point["t_ms"])) for point in strokes[index]["points"]]
                for index in indices
            ]
            canonical = _canonicalize(SourceSample(
                "project_owned_legacy_raw", f"{sample_id}:{group_index}", token,
                "project_owned_legacy_writer_catalog", "raw_observed_writer_evidence", raw_strokes,
            ))
            features.append(tensorize(canonical))
            tokens.append(token)
            provenance.append({
                "sample_id": sample_id,
                "writer_id": str(annotation["writer_id"]),
                "session_group": str(source.get("session_group", "")),
                "group_index": group_index,
                "stroke_indices": list(indices),
                "raw_stroke_sha256": _raw_stroke_hash(strokes, indices),
                "canonical_record_id": canonical["record_id"],
                "duration_ms": canonical["transform"]["duration_ms"],
                "source_point_count": canonical["point_count"],
            })
    return np.stack(features).astype(np.float32), tokens, provenance, {
        "actual_hashes": actual_hashes,
        "hash_mismatch_count": len(hash_mismatches),
        "hash_mismatches": hash_mismatches,
        "fresh_ids": len(fresh_ids),
        "legacy_ownership_rows": len(ownership),
        "fresh_ownership_rows": len(all_ownership) - len(ownership),
        "legacy_formulae": len(formulas),
        "fresh_formulae": sum(str(row["sample_id"]) in fresh_ids for row in all_formulas),
        "unowned_legacy_formulae": len(unowned),
    }


def _load_raw_catalog(run_dir: Path, report: dict[str, Any]) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    """v7 raw catalog의 tensor·token·provenance·fresh 격리를 검증한다."""

    npz_path = run_dir / "raw_legacy_writer_catalog.npz"
    metadata_path = run_dir / "raw_legacy_writer_catalog.metadata.jsonl"
    with np.load(npz_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        tokens = np.asarray(payload["tokens"]).astype(str)
    metadata = _read_jsonl(metadata_path)
    if features.shape != (393, 128, 5) or len(tokens) != 393 or len(metadata) != 393:
        raise ValueError("v7 raw catalog shape or metadata coverage mismatch")
    frozen_root = Path(report["inputs"]["frozen_root"])
    rebuilt_features, rebuilt_tokens, rebuilt_provenance, source_partition = _rebuild_legacy_catalog(frozen_root, report)
    expected_formulas = report["partition"]["hashes"]["formulas"]
    expected_ownership = report["partition"]["hashes"]["ownership"]
    label_indices = []
    failures: list[str] = []
    for index, (token, row, feature) in enumerate(zip(tokens, metadata, features, strict=True)):
        label_indices.append(int(row["label_index"]))
        if int(row["source_row"]) != index or str(row["token"]) != token:
            failures.append(f"row {index}: token/source_row mismatch")
        if row.get("source_domain") != RAW_DOMAIN or "raw_observed" not in str(row.get("evidence_tier", "")):
            failures.append(f"row {index}: raw evidence tier mismatch")
        if row.get("formulas_sha256") != expected_formulas or row.get("ownership_sha256") != expected_ownership:
            failures.append(f"row {index}: parent hash mismatch")
        if int(np.count_nonzero(feature[:, 3] > 0.5)) != int(row["stroke_count"]):
            failures.append(f"row {index}: stroke topology mismatch")
        expected_row = rebuilt_provenance[index]
        for field in (
            "sample_id", "writer_id", "session_group", "group_index", "stroke_indices",
            "raw_stroke_sha256", "canonical_record_id", "duration_ms", "source_point_count",
        ):
            if row.get(field) != expected_row[field]:
                failures.append(f"row {index}: rebuilt {field} mismatch")
    fresh_ids = {str(row["sample_id"]) for row in _read_jsonl(frozen_root / "ownership_fresh_acceptance.jsonl")}
    raw_ids = {str(row["sample_id"]) for row in metadata}
    writer_ids = {str(row["writer_id"]) for row in metadata}
    t_rows = [row for row in metadata if row["token"] == "t"]
    if raw_ids & fresh_ids:
        failures.append("raw legacy catalog contains a known replay sample")
    if len(raw_ids) != 96 or len(writer_ids) != 7 or len({row["token"] for row in metadata}) != 28:
        failures.append("raw legacy formula/writer/class cardinality mismatch")
    if len(t_rows) != 1 or int(t_rows[0]["label_index"]) != -1:
        failures.append("lowercase t evidence boundary mismatch")
    tensor_max_abs_error = float(np.max(np.abs(features - rebuilt_features)))
    tensor_exact_match = bool(np.array_equal(features, rebuilt_features))
    token_exact_match = bool(np.array_equal(tokens, np.asarray(rebuilt_tokens)))
    if not tensor_exact_match:
        failures.append(f"raw tensor rebuild mismatch: max_abs_error={tensor_max_abs_error}")
    if not token_exact_match:
        failures.append("raw token rebuild mismatch")
    expected_partition = {
        "fresh_ids": 53,
        "legacy_ownership_rows": 96,
        "fresh_ownership_rows": 53,
        "legacy_formulae": 96,
        "fresh_formulae": 53,
        "unowned_legacy_formulae": 10,
    }
    for key, expected in expected_partition.items():
        if source_partition[key] != expected:
            failures.append(f"source partition {key}: {source_partition[key]} != {expected}")
    if source_partition["hash_mismatch_count"]:
        failures.append("frozen source hash mismatch")
    return {
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256(npz_path),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "glyphs": len(metadata),
        "formulae": len(raw_ids),
        "writers": len(writer_ids),
        "classes": len({row["token"] for row in metadata}),
        "t_evidence_rows": len(t_rows),
        "fresh_sample_overlap": len(raw_ids & fresh_ids),
        "source_partition": source_partition,
        "tensor_rebuild_exact_match": tensor_exact_match,
        "tensor_rebuild_max_abs_error": tensor_max_abs_error,
        "token_rebuild_exact_match": token_exact_match,
        "finite": bool(np.isfinite(features).all()),
        "failure_count": len(failures),
        "failures": failures[:50],
    }, (features, np.asarray(label_indices, dtype=np.int64), metadata)


def _load_external(report: dict[str, Any]) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    """v7 external fallback bank의 hash와 행별 합성 provenance를 검증한다."""

    input_row = report["inputs"]["external"]
    path = Path(input_row["path"]).resolve()
    if _sha256(path) != input_row["sha256"]:
        raise ValueError("v7 external fallback hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = _read_jsonl(_source_metadata_path(path))
    if features.shape != (4096, 128, 5) or len(labels) != len(metadata):
        raise ValueError("v7 external fallback shape or metadata mismatch")
    synthetic_rows = sum(bool(row.get("synthetic_id") or row.get("parents")) for row in metadata)
    return {
        "path": str(path),
        "sha256": input_row["sha256"],
        "records": len(labels),
        "classes": len(np.unique(labels)),
        "synthetic_rows": synthetic_rows,
        "finite": bool(np.isfinite(features).all()),
    }, (features, labels, metadata)


def _formula_sources(
    run_dir: Path,
    raw_source: tuple[np.ndarray, np.ndarray, list[dict[str, Any]]],
    external_source: tuple[np.ndarray, np.ndarray, list[dict[str, Any]]],
    report: dict[str, Any],
) -> dict[str, Any]:
    """생성 수식의 source join과 생성 작가별 고정 raw 부모를 검증한다."""

    rows = _read_jsonl(run_dir / "formula_generation.jsonl")
    formula = _formula_audit(rows, {RAW_DOMAIN: raw_source, EXTERNAL_DOMAIN: external_source})
    domain_counts: Counter[str] = Counter()
    parent_by_generated: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        generated_writer = str(row["writer_id"])
        for allograph in row["allographs"]:
            plan = allograph["plan"]
            domain = str(plan["source_domain"])
            domain_counts[domain] += 1
            if domain == RAW_DOMAIN:
                parent_by_generated[generated_writer].add(str(plan["writer_id"]))
    expected_map = report["formula_generation"]["fixed_legacy_base_writer"]
    parent_mismatches = {
        writer: sorted(parents)
        for writer, parents in parent_by_generated.items()
        if parents != {str(expected_map[writer])}
    }
    expected_counts = report["formula_generation"]["selected_source_domains"]
    return {
        **formula,
        "source_domain_counts": dict(sorted(domain_counts.items())),
        "source_domain_counts_match_report": dict(domain_counts) == expected_counts,
        "raw_parent_writers": {key: sorted(value) for key, value in sorted(parent_by_generated.items())},
        "raw_parent_mismatch_count": len(parent_mismatches),
        "raw_parent_mismatches": parent_mismatches,
        "maximum_raw_parent_writers_per_generated_writer": max(map(len, parent_by_generated.values())),
    }


def _fresh_replay_rows(frozen_root: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """known replay 53수식의 186 glyph를 원시 ownership에서 독립 tensorize한다."""

    formulas_path = frozen_root / "frozen_dataset/data/formulas_valid.jsonl"
    ownership_path = frozen_root / "ownership_fresh_acceptance.jsonl"
    ownership = _read_jsonl(ownership_path)
    identifiers = {str(row["sample_id"]) for row in ownership}
    formulas = {
        str(row["sample_id"]): row
        for row in _read_jsonl(formulas_path)
        if str(row["sample_id"]) in identifiers
    }
    label_to_index = {token: index for index, token in enumerate(labels)}
    features: list[np.ndarray] = []
    truth: list[int] = []
    metadata: list[dict[str, Any]] = []
    for annotation in ownership:
        sample_id = str(annotation["sample_id"])
        source = formulas.get(sample_id)
        if source is None or str(source["writer_id"]) != str(annotation["writer_id"]):
            raise ValueError(f"known replay source/writer mismatch: {sample_id}")
        strokes = sorted(source["strokes"], key=lambda row: int(row["order"]))
        groups = [tuple(int(value) for value in group) for group in annotation["groups"]]
        tokens = [str(value) for value in annotation["labels"]]
        assigned = [index for group in groups for index in group]
        targets = [str(cell["token"]) for cell in source.get("target_cells") or []]
        if tokens != targets or sorted(assigned) != list(range(len(strokes))) or len(assigned) != len(set(assigned)):
            raise ValueError(f"known replay ownership contract mismatch: {sample_id}")
        for group_index, (indices, token) in enumerate(zip(groups, tokens, strict=True)):
            raw_strokes = [
                [(float(point["x"]), float(point["y"]), float(point["t_ms"])) for point in strokes[index]["points"]]
                for index in indices
            ]
            canonical = _canonicalize(SourceSample(
                "project_owned_known_writer_replay", f"{sample_id}:{group_index}", token,
                "known_writer_disjoint_replay", "diagnostic_only", raw_strokes,
            ))
            features.append(tensorize(canonical))
            truth.append(label_to_index.get(token, -1))
            metadata.append({"sample_id": sample_id, "writer_id": str(annotation["writer_id"]), "token": token})
    return np.stack(features).astype(np.float32), np.asarray(truth, dtype=np.int64), metadata


def _recompute_replay(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """동결 HWR로 known replay Top-1/Top-5/formula exact를 독립 재계산한다."""

    receipt_path = run_dir / "known_writer_disjoint_replay_diagnostic/report.json"
    receipt = _read_json(receipt_path)
    checkpoint = Path(report["inputs"]["checkpoint"]["path"]).resolve()
    if _sha256(checkpoint) != report["inputs"]["checkpoint"]["sha256"]:
        raise ValueError("v7 checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = [str(value) for value in payload["math_labels"]]
    features, truth, metadata = _fresh_replay_rows(Path(report["inputs"]["frozen_root"]), labels)
    supported = np.flatnonzero(truth >= 0)
    model = InkClassifierV1(len(labels), None)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        rank = model(torch.from_numpy(features[supported]), "math").argsort(dim=1, descending=True)[:, :5].cpu().numpy()
    expected = truth[supported]
    per_formula: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for index, target, predicted in zip(supported, expected, rank, strict=True):
        per_formula[metadata[int(index)]["sample_id"]].append((int(target), predicted))
    metrics = {
        "supported_glyphs": int(len(supported)),
        "unsupported_glyphs": int(len(truth) - len(supported)),
        "top1": float(np.mean(rank[:, 0] == expected)),
        "top5": float(np.mean(np.any(rank == expected[:, None], axis=1))),
        "formula_top1_exact": float(np.mean([
            all(int(predicted[0]) == target for target, predicted in rows)
            for rows in per_formula.values()
        ])),
        "formula_top5_oracle": float(np.mean([
            all(target in predicted for target, predicted in rows)
            for rows in per_formula.values()
        ])),
    }
    receipt_metrics = receipt["metrics"]
    mismatches = {
        key: {"recomputed": value, "receipt": receipt_metrics.get(key)}
        for key, value in metrics.items()
        if abs(float(value) - float(receipt_metrics.get(key, float("nan")))) > 1.0e-12
    }
    legacy_writers = set(report["raw_catalog"]["writer_ids"])
    replay_writers = {row["writer_id"] for row in metadata}
    return {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "status": receipt.get("status"),
        "formulae": len(per_formula),
        "writers": len(replay_writers),
        "glyphs": len(metadata),
        "writer_overlap_with_legacy": len(legacy_writers & replay_writers),
        "metrics": metrics,
        "metric_mismatch_count": len(mismatches),
        "metric_mismatches": mismatches,
        "untouched_acceptance": bool(receipt["gates"].get("untouched_acceptance")),
        "promotion_evidence": bool(receipt["gates"].get("promotion_evidence")),
        "promotion_ready": bool(receipt["gates"].get("promotion_ready")),
    }


def audit(run_dir: Path) -> dict[str, Any]:
    """v7 build와 replay의 독립 검증 결과 및 승격 결정을 생성한다."""

    run_dir = run_dir.resolve()
    report = _read_json(run_dir / "report.json")
    if report.get("schema") != "aiflow-hierarchical-writer-model/v7":
        raise ValueError("independent v7 audit requires a v7 build")
    raw, raw_source = _load_raw_catalog(run_dir, report)
    external, external_source = _load_external(report)
    formula = _formula_sources(run_dir, raw_source, external_source, report)
    replay = _recompute_replay(run_dir, report)
    structural = bool(
        raw["failure_count"] == 0 and raw["finite"] and external["synthetic_rows"] == external["records"]
        and formula["provenance_failure_count"] == 0
        and formula["source_domain_counts_match_report"]
        and formula["raw_parent_mismatch_count"] == 0
        and formula["maximum_raw_parent_writers_per_generated_writer"] == 1
        and replay["writer_overlap_with_legacy"] == 0
        and replay["metric_mismatch_count"] == 0
    )
    gates = {
        "structural_and_provenance_pass": structural,
        "raw_human_writer_evidence_present": bool(
            raw["glyphs"] == 393 and raw["tensor_rebuild_exact_match"] and raw["failure_count"] == 0
        ),
        "coherent_fixed_raw_parent_writer": formula["maximum_raw_parent_writers_per_generated_writer"] == 1,
        "known_writer_disjoint_replay_reproduced": replay["metric_mismatch_count"] == 0,
        "untouched_acceptance_present": False,
        "commercial_model_promotion_ready": False,
    }
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audited_run": str(run_dir),
        "raw_catalog": raw,
        "external_fallback": external,
        "formula_generation": formula,
        "known_writer_disjoint_replay": replay,
        "gates": gates,
        "decision": "RAW_WRITER_SHADOW_CANDIDATE" if structural else "REJECT_INDEPENDENT_AUDIT_FAILURE",
        "boundary": "known replay is diagnostic only; a newly collected untouched writer/formula set is required for promotion",
    }


def _markdown(report: dict[str, Any]) -> str:
    """v7 독립 감사 결과를 간결한 한국어 Markdown으로 변환한다."""

    raw = report["raw_catalog"]
    formula = report["formula_generation"]
    replay = report["known_writer_disjoint_replay"]
    return "\n".join([
        "# 계층형 작가모델 v7 독립 감사",
        "",
        "## 판정",
        "",
        f"- 최종 판정: **{report['decision']}**",
        f"- raw catalog: {raw['formulae']}수식 / {raw['writers']}작가 / {raw['glyphs']}glyph / {raw['classes']}클래스.",
        f"- 동결 원본 재구성: tensor exact={raw['tensor_rebuild_exact_match']}, token exact={raw['token_rebuild_exact_match']}, source hash mismatch={raw['source_partition']['hash_mismatch_count']}건.",
        f"- 생성식 원천: raw {formula['source_domain_counts'].get(RAW_DOMAIN, 0)} / external fallback {formula['source_domain_counts'].get(EXTERNAL_DOMAIN, 0)}.",
        f"- 생성 작가별 raw 부모 최대: {formula['maximum_raw_parent_writers_per_generated_writer']}명; provenance 실패 {formula['provenance_failure_count']}건.",
        f"- known replay 재계산: Top-1 {100*replay['metrics']['top1']:.2f}%, Top-5 {100*replay['metrics']['top5']:.2f}%, 수식 exact {100*replay['metrics']['formula_top1_exact']:.2f}%, Top-5 oracle {100*replay['metrics']['formula_top5_oracle']:.2f}%.",
        f"- replay 영수증 불일치: {replay['metric_mismatch_count']}건; legacy writer 중첩: {replay['writer_overlap_with_legacy']}명.",
        "",
        "## 경계",
        "",
        "- v7은 합성 전용 v6에서 실제 legacy raw writer 증거를 우선하는 구조로 진전했다.",
        "- replay 53수식은 writer-disjoint이지만 이미 알려진 자료이므로 튜닝·promotion 근거가 아니다.",
        "- 소문자 t는 raw 증거 1건만 있고 372-class 출력부에는 없어 evidence-only다.",
        "- 상용 승격에는 새로 수집하고 사전 동결한 작가/수식 acceptance가 필요하다.",
        "",
    ])


def main() -> int:
    """D:의 v7 산출물을 감사하고 새 JSON·Markdown 영수증을 작성한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    markdown = output.with_suffix(".md")
    if run_dir.drive.upper() != "D:" or output.drive.upper() != "D:":
        parser.error("run and output must remain on D:")
    if output.exists() or markdown.exists():
        parser.error("refusing to overwrite an existing v7 audit receipt")
    report = audit(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": report["decision"], "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
