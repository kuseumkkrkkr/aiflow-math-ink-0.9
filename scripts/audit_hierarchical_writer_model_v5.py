#!/usr/bin/env python3
"""계층형 작가모델 v5의 provenance와 승격 가능성을 독립 감사한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


SCHEMA = "aiflow-hierarchical-writer-model-independent-audit/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOMAIN_INPUTS = (
    ("approved_external_cleanroom", "external"),
    ("project_owned_cleanroom", "owned"),
    ("project_evaluation_cleanroom_smoke_only", "project_evaluation"),
)


def _sha256(path: Path) -> str:
    """파일을 스트리밍으로 읽어 SHA-256을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """UTF-8 JSON 객체를 읽고 최상위 자료형을 검증한다."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """UTF-8 JSONL 또는 JSONL.GZ를 순서대로 읽는다."""

    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _metadata_path(npz_path: Path) -> Path:
    """NPZ와 같은 stem을 쓰는 압축 provenance 경로를 계산한다."""

    return npz_path.with_suffix(".metadata.jsonl.gz")


def _input_path_and_hash(
    inputs: dict[str, Any], input_key: str,
) -> tuple[Path, str | None] | None:
    """v5 문자열 입력과 v6 role 객체 입력을 공통 경로·hash로 정규화한다."""

    value = inputs.get(input_key)
    if value is None:
        return None
    if isinstance(value, dict):
        if value.get("loaded") is False:
            return None
        path_value = value.get("path")
        if not path_value:
            return None
        return Path(path_value).resolve(), value.get("sha256")
    return Path(value).resolve(), inputs.get(f"{input_key}_sha256")


def _collect_sha256(value: Any) -> set[str]:
    """중첩 객체에 포함된 SHA-256 문자열을 재귀적으로 수집한다."""

    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(_collect_sha256(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            output.update(_collect_sha256(child))
    elif isinstance(value, str) and SHA256_RE.fullmatch(value):
        output.add(value)
    return output


def _is_raw_human_evidence(row: dict[str, Any]) -> bool:
    """metadata evidence tier가 합성물이 아닌 원시 인간 관측인지 판정한다."""

    tier = str(row.get("evidence_tier", row.get("record_kind", ""))).lower()
    return "raw_observed" in tier or tier == "human_observed"


def _load_checkpoint_report(path: Path) -> dict[str, Any]:
    """체크포인트에서 학습 계보 보고서만 CPU로 안전하게 읽는다."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    report = payload.get("report", {})
    if not isinstance(report, dict):
        raise ValueError("checkpoint report is missing or malformed")
    return report


def _source_audit(path: Path, expected_sha256: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """한 합성 은행의 텐서·행별 provenance·작가 연결성을 검증한다."""

    metadata_path = _metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"source or metadata is missing: {path}")
    actual_sha256 = _sha256(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"source hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=np.int64)
    metadata = _read_jsonl(metadata_path)
    if features.shape != (len(labels), 128, 5) or len(metadata) != len(labels):
        raise ValueError(f"tensor or metadata coverage mismatch: {path}")

    synthetic = [bool(row.get("synthetic_id") or row.get("parents")) for row in metadata]
    raw_observed = [_is_raw_human_evidence(row) for row in metadata]
    writer_fingerprints = {
        str(writer)
        for row in metadata
        for writer in row.get("parent_writer_fingerprints", [])
        if writer
    }
    observed_counts = np.sum(features[:, :, 4] > 0.5, axis=1)
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "records": int(len(labels)),
        "classes": int(len(np.unique(labels))),
        "metadata_records": len(metadata),
        "synthetic_rows": int(sum(synthetic)),
        "raw_human_observed_rows": int(sum(raw_observed)),
        "cross_writer_synthetic_rows": int(sum(bool(row.get("cross_writer")) for row in metadata)),
        "rows_with_parent_writer_fingerprints": int(sum(bool(row.get("parent_writer_fingerprints")) for row in metadata)),
        "unique_parent_writer_fingerprints": len(writer_fingerprints),
        "finite": bool(np.isfinite(features).all()),
        "observed_points": {
            "minimum": int(observed_counts.min()),
            "maximum": int(observed_counts.max()),
        },
    }, metadata


def _baseline_collisions(row: dict[str, Any]) -> list[dict[str, Any]]:
    """실제 생성 좌표에서 인접 baseline glyph의 강한 겹침 후보를 찾는다."""

    glyphs = {str(item["glyph_id"]): item for item in row.get("glyphs", [])}
    points_by_glyph: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in row.get("events", []):
        glyph_id = str(event.get("glyph_id", ""))
        points_by_glyph[glyph_id].extend(
            (float(point["x"]), float(point["y"])) for point in event.get("points", [])
        )
    baseline = sorted(
        (item for item in glyphs.values() if item.get("relation") == "baseline"),
        key=lambda item: float(item.get("x", 0.0)),
    )
    collisions: list[dict[str, Any]] = []
    for left, right in zip(baseline, baseline[1:]):
        left_id, right_id = str(left["glyph_id"]), str(right["glyph_id"])
        left_points, right_points = points_by_glyph.get(left_id, []), points_by_glyph.get(right_id, [])
        if not left_points or not right_points:
            continue
        left_x = [point[0] for point in left_points]; left_y = [point[1] for point in left_points]
        right_x = [point[0] for point in right_points]; right_y = [point[1] for point in right_points]
        overlap_x = min(max(left_x), max(right_x)) - max(min(left_x), min(right_x))
        overlap_y = min(max(left_y), max(right_y)) - max(min(left_y), min(right_y))
        width = max(min(max(left_x) - min(left_x), max(right_x) - min(right_x)), 1.0e-6)
        height = max(min(max(left_y) - min(left_y), max(right_y) - min(right_y)), 1.0e-6)
        x_ratio, y_ratio = overlap_x / width, overlap_y / height
        if x_ratio >= 0.15 and y_ratio >= 0.15:
            collisions.append({
                "left_glyph": left_id,
                "right_glyph": right_id,
                "left_token": str(left.get("token", "")),
                "right_token": str(right.get("token", "")),
                "x_overlap_ratio": float(x_ratio),
                "y_overlap_ratio": float(y_ratio),
            })
    return collisions


def _formula_audit(
    rows: Iterable[dict[str, Any]],
    sources: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """수식 생성 행의 glyph/stroke/time/provenance 완전성을 검증한다."""

    formula_rows = 0
    glyph_rows = 0
    event_rows = 0
    selected_synthetic = 0
    selected_raw_observed = 0
    selected_cross_writer = 0
    provenance_failures: list[str] = []
    collision_candidates: list[dict[str, Any]] = []
    formula_ids: set[str] = set()
    writer_ids: set[str] = set()
    for formula_rows, row in enumerate(rows, 1):
        formula_id = str(row.get("formula_id", ""))
        writer_id = str(row.get("writer_id", ""))
        formula_ids.add(formula_id)
        writer_ids.add(writer_id)
        glyphs = {str(item["glyph_id"]): item for item in row.get("glyphs", [])}
        order = [str(value) for value in row.get("production_order", [])]
        if len(order) != len(glyphs) or len(order) != len(set(order)) or set(order) != set(glyphs):
            provenance_failures.append(f"{formula_id}/{writer_id}: invalid glyph production partition")
        allographs = row.get("allographs", [])
        glyph_rows += len(allographs)
        expected_strokes = 0
        for item in allographs:
            glyph_id = str(item.get("glyph_id", ""))
            plan = item.get("plan", {})
            domain = str(plan.get("source_domain", ""))
            source_row = int(plan.get("source_row", -1))
            source = sources.get(domain)
            if glyph_id not in glyphs or source is None or not 0 <= source_row < len(source[1]):
                provenance_failures.append(f"{formula_id}/{writer_id}/{glyph_id}: unresolved source row")
                continue
            _features, labels, metadata = source
            if int(plan.get("label_index", -1)) != int(labels[source_row]):
                provenance_failures.append(f"{formula_id}/{writer_id}/{glyph_id}: source label mismatch")
            meta = metadata[source_row]
            is_synthetic = bool(meta.get("synthetic_id") or meta.get("parents"))
            selected_synthetic += int(is_synthetic)
            selected_raw_observed += int(_is_raw_human_evidence(meta))
            selected_cross_writer += int(bool(meta.get("cross_writer")))
            expected_strokes += int(plan.get("stroke_count", 0))
        events = row.get("events", [])
        event_rows += len(events)
        for collision in _baseline_collisions(row):
            collision_candidates.append({
                "formula_id": formula_id,
                "writer_id": writer_id,
                **collision,
            })
        if len(events) != expected_strokes:
            provenance_failures.append(f"{formula_id}/{writer_id}: stroke count mismatch")
        prior_time = -math.inf
        for event in events:
            if str(event.get("glyph_id", "")) not in glyphs:
                provenance_failures.append(f"{formula_id}/{writer_id}: event references unknown glyph")
            points = event.get("points", [])
            times = [float(point["t_ms"]) for point in points]
            coordinates = [float(point[axis]) for point in points for axis in ("x", "y", "t_ms")]
            if not points or not all(math.isfinite(value) for value in coordinates):
                provenance_failures.append(f"{formula_id}/{writer_id}: non-finite or empty event")
                continue
            if any(right <= left for left, right in zip(times, times[1:])) or times[0] <= prior_time:
                provenance_failures.append(f"{formula_id}/{writer_id}: non-monotonic event time")
            prior_time = times[-1]
    return {
        "formula_writer_rows": formula_rows,
        "formula_templates": len(formula_ids),
        "writers": len(writer_ids),
        "selected_glyphs": glyph_rows,
        "selected_strokes": event_rows,
        "selected_synthetic_glyphs": selected_synthetic,
        "selected_raw_human_observed_glyphs": selected_raw_observed,
        "selected_cross_writer_synthetic_glyphs": selected_cross_writer,
        "provenance_failure_count": len(provenance_failures),
        "provenance_failures": provenance_failures[:50],
        "baseline_collision_candidate_count": len(collision_candidates),
        "baseline_collision_candidates": collision_candidates[:50],
    }


def _assignment_audit(path: Path) -> dict[str, Any]:
    """작가별 문자 allograph 할당의 실제 source 다양성을 계산한다."""

    payload = _read_json(path)
    by_token: dict[str, set[tuple[str, int]]] = defaultdict(set)
    assignments = 0
    for values in payload.values():
        if not isinstance(values, dict):
            continue
        for token, plan in values.items():
            assignments += 1
            by_token[str(token)].add((str(plan["source_domain"]), int(plan["source_row"])))
    counts = Counter(len(values) for values in by_token.values())
    return {
        "writers": len(payload),
        "assignments": assignments,
        "tokens": len(by_token),
        "tokens_with_multiple_source_rows": int(sum(count for variants, count in counts.items() if variants > 1)),
        "source_variants_per_token": {str(key): value for key, value in sorted(counts.items())},
    }


def _checkpoint_lineage(checkpoint: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    """분류기 학습 보고서와 생성원 manifest 사이의 계보 중첩을 보수적으로 판정한다."""

    report = _load_checkpoint_report(checkpoint)
    checkpoint_sources = set(report.get("clean_room_contract", {}).get("admitted_parent_sources", []))
    source_counts = source_manifest.get("clean_room_contract", {}).get("admitted_external_sources", {})
    source_names = set(source_counts) if isinstance(source_counts, dict) else set(source_counts)
    sha_overlap = _collect_sha256(report) & _collect_sha256(source_manifest)
    source_overlap = sorted(checkpoint_sources & source_names)
    return {
        "checkpoint_schema": report.get("schema"),
        "checkpoint_status": report.get("status"),
        "admitted_parent_source_overlap": source_overlap,
        "sha256_lineage_overlap_count": len(sha_overlap),
        "independent_from_generator_sources": not source_overlap and not sha_overlap,
        "note": "계보 중첩은 행 단위 누수를 뜻하지 않지만 독립 작가 일반화 검증으로 사용할 수 없음을 뜻한다.",
    }


def _evaluation_boundary(source_manifest: dict[str, Any], project_bank_used: bool) -> dict[str, Any]:
    """생성은행 부모가 평가 전용 원시 행인지 확인해 학습 유입을 차단한다."""

    inputs = source_manifest.get("inputs", {})
    raw_path_value = inputs.get("project_owned_rows")
    if not raw_path_value:
        return {
            "project_parent_path_present": False,
            "records": 0,
            "writers": 0,
            "training_roles": {},
            "splits": {},
            "declared_sha256_matches": False,
            "evaluation_only_parent_lineage": False,
            "evaluation_only_parent_lineage_used": False,
        }
    raw_path = Path(raw_path_value).resolve()
    rows = _read_jsonl(raw_path)
    roles = Counter(str(row.get("training_role", "")) for row in rows)
    splits = Counter(str(row.get("split", "")) for row in rows)
    writers = {str(row.get("writer_group", "")) for row in rows if row.get("writer_group")}
    expected_sha = inputs.get("project_owned_rows_sha256")
    actual_sha = _sha256(raw_path)
    evaluation_only = bool(rows) and all(
        "evaluation" in str(row.get("training_role", "")).lower()
        or "evaluation" in str(row.get("split", "")).lower()
        for row in rows
    )
    return {
        "project_parent_path_present": True,
        "records": len(rows),
        "writers": len(writers),
        "training_roles": dict(sorted(roles.items())),
        "splits": dict(sorted(splits.items())),
        "declared_sha256_matches": bool(expected_sha and expected_sha == actual_sha),
        "evaluation_only_parent_lineage": evaluation_only,
        "evaluation_only_parent_lineage_used": bool(project_bank_used and evaluation_only),
        "policy": "평가 전용 부모의 파생 행은 구조 스모크에는 쓸 수 있지만 학습·증강 선택·승격에는 사용할 수 없다.",
    }


def audit(run_dir: Path) -> dict[str, Any]:
    """v5 산출물 전체를 읽어 구조·증거·승격 게이트를 계산한다."""

    run_dir = run_dir.resolve()
    report = _read_json(run_dir / "report.json")
    inputs = report.get("inputs", {})
    source_records: dict[str, dict[str, Any]] = {}
    source_values: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    source_manifests: list[dict[str, Any]] = []
    for domain, input_key in DOMAIN_INPUTS:
        normalized_input = _input_path_and_hash(inputs, input_key)
        if normalized_input is None:
            continue
        path, expected_sha = normalized_input
        source_record, metadata = _source_audit(path, expected_sha)
        with np.load(path, allow_pickle=False) as payload:
            features = np.asarray(payload["features"], dtype=np.float32)
            labels = np.asarray(payload["labels"], dtype=np.int64)
        source_records[domain] = source_record
        source_values[domain] = (features, labels, metadata)
        manifest_path = path.parent / "manifest.json"
        if manifest_path.is_file():
            source_manifests.append(_read_json(manifest_path))
    if not source_manifests:
        raise FileNotFoundError("source dataset manifest is missing")

    formulas = _read_jsonl(run_dir / "formula_generation.jsonl")
    formula = _formula_audit(formulas, source_values)
    assignments = _assignment_audit(run_dir / "writer_allograph_assignments.json")
    checkpoint_input = _input_path_and_hash(inputs, "checkpoint")
    if checkpoint_input is None:
        raise ValueError("checkpoint input is missing")
    checkpoint, checkpoint_sha = checkpoint_input
    if _sha256(checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint hash mismatch")
    lineage = _checkpoint_lineage(checkpoint, source_manifests[0])
    evaluation_boundary = _evaluation_boundary(
        source_manifests[0], any(domain != "approved_external_cleanroom" for domain in source_records),
    )

    preservation = report.get("frozen_hwr_alphanumeric_preservation", {})
    top5_values = [float(row["top5"]) for row in preservation.values() if "top5" in row]
    top1_values = [float(row["top1"]) for row in preservation.values() if "top1" in row]
    minimum_top5 = min(top5_values) if top5_values else 0.0
    source_raw_rows = sum(row["raw_human_observed_rows"] for row in source_records.values())
    source_synthetic_rows = sum(row["synthetic_rows"] for row in source_records.values())
    structural = formula["provenance_failure_count"] == 0 and all(
        row["finite"] and row["records"] == row["metadata_records"]
        for row in source_records.values()
    )
    evidence = {
        "synthetic_plan_terminology_required": source_synthetic_rows > 0 and source_raw_rows == 0,
        "raw_human_allograph_evidence_present": source_raw_rows > 0,
        "raw_writer_disjoint_acceptance_present": False,
        "classifier_independent_acceptance_present": lineage["independent_from_generator_sources"],
        "evaluation_only_parent_lineage_used": evaluation_boundary["evaluation_only_parent_lineage_used"],
    }
    label_safe = bool(top5_values and minimum_top5 >= 0.90)
    augmentation_admission = bool(
        structural and label_safe and not evaluation_boundary["evaluation_only_parent_lineage_used"]
    )
    gates = {
        "structural_smoke_pass": structural,
        "augmentation_label_safety_top5_90pct": label_safe,
        "training_or_augmentation_admission_allowed": augmentation_admission,
        "human_allograph_claim_allowed": evidence["raw_human_allograph_evidence_present"],
        "writer_generalization_claim_allowed": bool(
            evidence["raw_writer_disjoint_acceptance_present"]
            and evidence["classifier_independent_acceptance_present"]
        ),
        "commercial_model_promotion_ready": False,
    }
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audited_run": str(run_dir),
        "audited_run_schema": report.get("schema"),
        "sources": source_records,
        "formula_generation": formula,
        "allograph_assignments": assignments,
        "frozen_hwr_preservation": {
            "writers": len(top5_values),
            "minimum_top1": min(top1_values) if top1_values else 0.0,
            "maximum_top1": max(top1_values) if top1_values else 0.0,
            "minimum_top5": minimum_top5,
            "maximum_top5": max(top5_values) if top5_values else 0.0,
        },
        "checkpoint_lineage": lineage,
        "evaluation_boundary": evaluation_boundary,
        "evidence": evidence,
        "gates": gates,
        "decision": (
            "AUGMENTATION_SHADOW_CANDIDATE"
            if augmentation_admission else (
                "STRUCTURAL_SMOKE_ONLY" if structural else "REJECT_STRUCTURAL_FAILURE"
            )
        ),
        "required_next_evidence": [
            "평가 전용 387행과 겹치지 않는 프로젝트 소유 training/calibration writer 세트",
            "원시 인간 writer/session ID가 보존된 숫자·A-Z·a-z·수학기호 온라인 궤적",
            "동결 생성기와 동결 HWR 선택이 끝난 뒤의 미접촉 작가 분리 평가",
            "누락 소문자 t 및 t/f/+ 동형군의 프로젝트 소유 다작가 표본",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    """감사 JSON을 사람이 검토하기 쉬운 한국어 Markdown으로 변환한다."""

    sources = report["sources"]
    hwr = report["frozen_hwr_preservation"]
    formula = report["formula_generation"]
    lines = [
        f"# 계층형 작가모델 독립 감사 ({report['audited_run_schema']})",
        "",
        "## 판정",
        "",
        f"- 최종 판정: **{report['decision']}**",
        f"- 구조·provenance 스모크: **{'통과' if report['gates']['structural_smoke_pass'] else '실패'}**",
        f"- 작가별 동결 HWR Top-5 최저: **{100*hwr['minimum_top5']:.2f}%**",
        "- 현재 입력은 모두 DTW/physics 합성은행이므로 `관측 allograph`가 아니라 `합성 allograph plan`으로만 표현해야 한다.",
        "- 원시 인간 writer/session의 미접촉 작가 분리 검증이 없어 상용 작가모델 승격 근거는 없다.",
        (
            "- project bank의 부모 387행은 `evaluation_only`이며 현재 run이 이 bank를 사용하므로 학습·증강 선택에 입장시킬 수 없다."
            if report["evaluation_boundary"]["evaluation_only_parent_lineage_used"]
            else "- 평가 전용 project bank는 현재 run의 catalog 입력에서 격리됐다."
        ),
        "",
        "## 입력 증거",
        "",
        "| 구분 | 행 | 합성 행 | 원시 인간 관측 | cross-writer 합성 | 고유 부모 작가 지문 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, row in sources.items():
        lines.append(
            f"| {domain} | {row['records']} | {row['synthetic_rows']} | "
            f"{row['raw_human_observed_rows']} | {row['cross_writer_synthetic_rows']} | "
            f"{row['unique_parent_writer_fingerprints']} |"
        )
    lines.extend([
        "",
        "## 생성 구조",
        "",
        f"- 수식×작가 행: {formula['formula_writer_rows']}; glyph: {formula['selected_glyphs']}; stroke: {formula['selected_strokes']}.",
        f"- source join/partition/time 실패: {formula['provenance_failure_count']}건.",
        f"- 인접 baseline glyph 겹침 후보: {formula['baseline_collision_candidate_count']}건(진단 지표, 자동 실패 gate 아님).",
        f"- 선택 glyph 중 합성 원천: {formula['selected_synthetic_glyphs']}건; 원시 인간 관측 원천: {formula['selected_raw_human_observed_glyphs']}건.",
        "",
        "## 왜 Top-5 게이트만으로 부족한가",
        "",
        "- 생성 원천과 동결 HWR이 같은 승인 데이터 계보를 공유한다. 이는 행 단위 누수로 단정할 수 없지만 독립 일반화 평가는 아니다.",
        "- HWR Top-5로 allograph를 고르면 문자 정체성 보존에는 도움이 되지만, 분류기가 이미 아는 필체만 남기는 순환 선택이 된다.",
        "- 따라서 이 값은 증강 안전성 지표로만 사용하고 인간 필체 다양성 또는 새 작가 성능으로 해석하지 않는다.",
        "",
        "## 다음 승격 조건",
        "",
    ])
    lines.extend(f"- {item}" for item in report["required_next_evidence"])
    lines.extend([
        "- 위 조건 전에는 기존 HWR/runtime/checkpoint를 변경하지 않고 shadow 구조 스모크로 유지한다.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    """CLI 인수를 검증하고 JSON과 Markdown 감사 영수증을 새로 생성한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="새 JSON 감사 보고서 경로")
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    markdown = output.with_suffix(".md")
    if run_dir.drive.upper() != "D:" or output.drive.upper() != "D:":
        parser.error("run and output must remain on D:")
    if output.exists() or markdown.exists():
        parser.error("refusing to overwrite an existing audit receipt")
    report = audit(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": report["decision"], "gates": report["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
