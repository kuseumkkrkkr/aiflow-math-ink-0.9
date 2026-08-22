#!/usr/bin/env python3
"""Independently audit the v8 shadow writer-adaptation experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_character_classifier_v1 import InkClassifierV1  # noqa: E402


SEED = 20260823
TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def deterministic_split(writer: str, sample_ids: list[str]) -> dict[str, Any]:
    ordered = sorted(
        set(sample_ids),
        key=lambda value: hashlib.sha256(f"{SEED}:{writer}:{value}".encode()).hexdigest(),
    )
    require(len(ordered) >= 4, f"writer {writer} has fewer than four formulae")
    count = min(6, max(2, len(ordered) // 4))
    calibration = ordered[:count]
    evaluation = ordered[count:]
    require(not (set(calibration) & set(evaluation)), f"formula overlap for {writer}")
    return {
        "writer_id": writer,
        "calibration_formulae": calibration,
        "evaluation_formulae": evaluation,
    }


def splits_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_writer: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_writer[str(row["writer_id"])].append(str(row["sample_id"]))
    return {
        writer: deterministic_split(writer, sample_ids)
        for writer, sample_ids in sorted(by_writer.items())
    }


def top5(logits: np.ndarray) -> np.ndarray:
    return np.argsort(logits, axis=1)[:, -5:][:, ::-1]


def frozen_outputs(features: np.ndarray, checkpoint: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = [str(value) for value in payload["math_labels"]]
    model = InkClassifierV1(len(labels), None)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            tensor = torch.from_numpy(features[start : start + 256])
            embedding = model.encode(tensor)
            embeddings.append(embedding.cpu().numpy())
            logits.append(model.math_head(embedding).cpu().numpy())
    return labels, np.concatenate(embeddings), np.concatenate(logits)


def prototype_200(
    calibration_embeddings: np.ndarray,
    calibration_truth: np.ndarray,
    evaluation_embeddings: np.ndarray,
    evaluation_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_rank = top5(evaluation_logits)
    normalized_calibration = calibration_embeddings / np.maximum(
        np.linalg.norm(calibration_embeddings, axis=1, keepdims=True), 1.0e-8
    )
    normalized_evaluation = evaluation_embeddings / np.maximum(
        np.linalg.norm(evaluation_embeddings, axis=1, keepdims=True), 1.0e-8
    )
    adapted = evaluation_logits.copy()
    for label in sorted(set(calibration_truth.tolist())):
        centroid = normalized_calibration[calibration_truth == label].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1.0e-8)
        adapted[:, label] += 2.0 * (normalized_evaluation @ centroid)
    masked = np.full_like(adapted, -1.0e9)
    local_rows = np.arange(len(adapted))[:, None]
    masked[local_rows, baseline_rank] = adapted[local_rows, baseline_rank]
    return baseline_rank, top5(masked)


def score_ranked_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(rows), "cannot score an empty row set")
    formulae: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        formulae[str(row["sample_id"])].append(row)

    def metrics(key: str) -> dict[str, Any]:
        return {
            "glyphs": len(rows),
            "top1": sum(row[key][0] == row["truth"] for row in rows) / len(rows),
            "top5": sum(row["truth"] in row[key] for row in rows) / len(rows),
            "formulae": len(formulae),
            "formula_exact": sum(
                all(row[key][0] == row["truth"] for row in members)
                for members in formulae.values()
            )
            / len(formulae),
            "formula_top5_oracle": sum(
                all(row["truth"] in row[key] for row in members)
                for members in formulae.values()
            )
            / len(formulae),
        }

    return {"baseline": metrics("baseline_top5"), "adapted": metrics("adapted_top5")}


def aggregate(per_writer: list[dict[str, Any]]) -> dict[str, Any]:
    glyphs = sum(row["baseline"]["glyphs"] for row in per_writer)
    formulae = sum(row["baseline"]["formulae"] for row in per_writer)
    output: dict[str, Any] = {
        "writers": len(per_writer),
        "glyphs": glyphs,
        "formulae": formulae,
        "candidate_violations": sum(row.get("candidate_violations", 0) for row in per_writer),
    }
    for version in ("baseline", "adapted"):
        output[version] = {
            "top1": sum(row[version]["top1"] * row[version]["glyphs"] for row in per_writer) / glyphs,
            "top5": sum(row[version]["top5"] * row[version]["glyphs"] for row in per_writer) / glyphs,
            "formula_exact": sum(
                row[version]["formula_exact"] * row[version]["formulae"] for row in per_writer
            )
            / formulae,
            "formula_top5_oracle": sum(
                row[version]["formula_top5_oracle"] * row[version]["formulae"]
                for row in per_writer
            )
            / formulae,
        }
    return output


def compare_metrics(actual: dict[str, Any], expected: dict[str, Any], context: str) -> None:
    for count in ("glyphs", "formulae"):
        require(int(actual[count]) == int(expected[count]), f"{context}.{count} mismatch")
    for metric in ("top1", "top5", "formula_exact", "formula_top5_oracle"):
        require_close(actual[metric], expected[metric], f"{context}.{metric}")


def writer_regressions(per_writer: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for row in per_writer:
        top1_delta = row["adapted"]["top1"] - row["baseline"]["top1"]
        formula_delta = row["adapted"]["formula_exact"] - row["baseline"]["formula_exact"]
        if row.get("candidate_violations", 0) or top1_delta < 0 or formula_delta < 0:
            regressions.append(
                {
                    "writer_id": row["writer_id"],
                    "top1_delta": top1_delta,
                    "formula_exact_delta": formula_delta,
                }
            )
    return regressions


def paired_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved = sum(
        row["baseline_top5"][0] != row["truth"] and row["adapted_top5"][0] == row["truth"]
        for row in rows
    )
    regressed = sum(
        row["baseline_top5"][0] == row["truth"] and row["adapted_top5"][0] != row["truth"]
        for row in rows
    )
    discordant = improved + regressed
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(improved, regressed) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "changed_top1": discordant,
        "improved": improved,
        "regressed": regressed,
        "mcnemar_exact_two_sided_p": p_value,
    }


def formula_paired_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_formula: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_formula[(str(row["writer_id"]), str(row["sample_id"]))].append(row)
    formula_rows = []
    for (writer_id, sample_id), members in by_formula.items():
        baseline_correct = all(row["baseline_top5"][0] == row["truth"] for row in members)
        adapted_correct = all(row["adapted_top5"][0] == row["truth"] for row in members)
        formula_rows.append(
            {
                "writer_id": writer_id,
                "sample_id": sample_id,
                "truth": True,
                "baseline_top5": [baseline_correct],
                "adapted_top5": [adapted_correct],
            }
        )
    return paired_counts(formula_rows)


def legacy_reproduction(
    v7_dir: Path,
    checkpoint: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with np.load(v7_dir / "raw_legacy_writer_catalog.npz", allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
    metadata = read_jsonl(v7_dir / "raw_legacy_writer_catalog.metadata.jsonl")
    require(features.shape == (393, 128, 5), "legacy tensor shape changed")
    require(len(metadata) == len(features), "legacy metadata/tensor row mismatch")
    labels, embeddings, logits = frozen_outputs(features, checkpoint)
    expected = report["selection"]["candidate_grid"]["prototype_200"]
    per_writer: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for writer, split in manifest["legacy"].items():
        calibration_ids = set(split["calibration_formulae"])
        evaluation_ids = set(split["evaluation_formulae"])
        calibration = [
            index
            for index, row in enumerate(metadata)
            if row["writer_id"] == writer
            and row["sample_id"] in calibration_ids
            and int(row["label_index"]) >= 0
        ]
        evaluation = [
            index
            for index, row in enumerate(metadata)
            if row["writer_id"] == writer
            and row["sample_id"] in evaluation_ids
            and int(row["label_index"]) >= 0
        ]
        calibration_truth = np.asarray([metadata[index]["label_index"] for index in calibration], dtype=np.int64)
        baseline_rank, adapted_rank = prototype_200(
            embeddings[calibration], calibration_truth, embeddings[evaluation], logits[evaluation]
        )
        local_rows: list[dict[str, Any]] = []
        for local, index in enumerate(evaluation):
            row = {
                "sample_id": str(metadata[index]["sample_id"]),
                "writer_id": writer,
                "group_index": int(metadata[index]["group_index"]),
                "truth": labels[int(metadata[index]["label_index"])],
                "baseline_top5": [labels[value] for value in baseline_rank[local]],
                "adapted_top5": [labels[value] for value in adapted_rank[local]],
            }
            require(set(row["baseline_top5"]) == set(row["adapted_top5"]), "legacy candidate set changed")
            local_rows.append(row)
            prediction_rows.append(row)
        scored = score_ranked_rows(local_rows)
        scored.update(
            {
                "writer_id": writer,
                "candidate_violations": 0,
                "calibration_glyphs": len(calibration),
                "evaluation_glyphs": len(evaluation),
            }
        )
        reported = next(row for row in expected["per_writer"] if row["writer_id"] == writer)
        compare_metrics(scored["baseline"], reported["baseline"], f"legacy.{writer}.baseline")
        compare_metrics(scored["adapted"], reported["adapted"], f"legacy.{writer}.adapted")
        require(scored["calibration_glyphs"] == reported["calibration_glyphs"], f"{writer} calibration glyph mismatch")
        per_writer.append(scored)
    reproduced = aggregate(per_writer)
    reported_aggregate = expected["aggregate"]
    require(reproduced["glyphs"] == reported_aggregate["glyphs"], "legacy aggregate glyph mismatch")
    require(reproduced["formulae"] == reported_aggregate["formulae"], "legacy aggregate formula mismatch")
    for version in ("baseline", "adapted"):
        compare_metrics(
            {**reproduced[version], "glyphs": reproduced["glyphs"], "formulae": reproduced["formulae"]},
            {**reported_aggregate[version], "glyphs": reported_aggregate["glyphs"], "formulae": reported_aggregate["formulae"]},
            f"legacy.aggregate.{version}",
        )
    return reproduced, per_writer, prediction_rows


def replay_audit(
    prediction_path: Path,
    ownership_path: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = read_jsonl(prediction_path)
    ownership = read_jsonl(ownership_path)
    truth_map: dict[tuple[str, int], tuple[str, str]] = {}
    for record in ownership:
        for group_index, token in enumerate(record["labels"]):
            truth_map[(str(record["sample_id"]), group_index)] = (str(record["writer_id"]), str(token))
    seen: set[tuple[str, int]] = set()
    by_writer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = (str(row["sample_id"]), int(row["group_index"]))
        require(key not in seen, f"duplicate replay prediction {key}")
        seen.add(key)
        require(key in truth_map, f"replay prediction has no ownership truth: {key}")
        writer, token = truth_map[key]
        require(str(row["writer_id"]) == writer, f"writer mismatch for {key}")
        require(str(row["truth"]) == token, f"truth mismatch for {key}")
        evaluation_ids = set(manifest["known_replay"][writer]["evaluation_formulae"])
        calibration_ids = set(manifest["known_replay"][writer]["calibration_formulae"])
        require(row["sample_id"] in evaluation_ids, f"non-evaluation replay row: {key}")
        require(row["sample_id"] not in calibration_ids, f"calibration leakage in replay ledger: {key}")
        require(len(row["baseline_top5"]) == 5 and len(set(row["baseline_top5"])) == 5, f"invalid baseline Top-5: {key}")
        require(len(row["adapted_top5"]) == 5 and len(set(row["adapted_top5"])) == 5, f"invalid adapted Top-5: {key}")
        require(set(row["baseline_top5"]) == set(row["adapted_top5"]), f"candidate set changed: {key}")
        by_writer[writer].append(row)

    per_writer: list[dict[str, Any]] = []
    for writer, rows in sorted(by_writer.items()):
        scored = score_ranked_rows(rows)
        scored.update({"writer_id": writer, "candidate_violations": 0})
        reported = next(row for row in report["known_replay"]["per_writer"] if row["writer_id"] == writer)
        compare_metrics(scored["baseline"], reported["baseline"], f"replay.{writer}.baseline")
        compare_metrics(scored["adapted"], reported["adapted"], f"replay.{writer}.adapted")
        per_writer.append(scored)
    reproduced = aggregate(per_writer)
    reported_aggregate = report["known_replay"]["aggregate"]
    require(reproduced["glyphs"] == reported_aggregate["glyphs"], "replay aggregate glyph mismatch")
    require(reproduced["formulae"] == reported_aggregate["formulae"], "replay aggregate formula mismatch")
    for version in ("baseline", "adapted"):
        compare_metrics(
            {**reproduced[version], "glyphs": reproduced["glyphs"], "formulae": reproduced["formulae"]},
            {**reported_aggregate[version], "glyphs": reported_aggregate["glyphs"], "formulae": reported_aggregate["formulae"]},
            f"replay.aggregate.{version}",
        )
    return reproduced, per_writer, predictions


def safe_positive(candidate: dict[str, Any]) -> bool:
    per_writer = candidate["per_writer"]
    summary = aggregate(per_writer)
    positive = (
        summary["adapted"]["top1"] > summary["baseline"]["top1"]
        or summary["adapted"]["formula_exact"] > summary["baseline"]["formula_exact"]
    )
    return not writer_regressions(per_writer) and positive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--v7-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--failclosed-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report_path = args.run_dir / "report.json"
    manifest_path = args.run_dir / "split_manifest.json"
    decision_path = args.run_dir / "adoption_decision.json"
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    decision = read_json(decision_path)
    v7_report_path = args.v7_dir / "report.json"
    v7_report = read_json(v7_report_path)
    failclosed_path = args.failclosed_dir / "pre_replay_rejection.json"
    failclosed = read_json(failclosed_path)

    hashes = {
        "base_report": sha256(report_path),
        "split_manifest": sha256(manifest_path),
        "implementation": sha256(args.implementation),
        "checkpoint": sha256(args.checkpoint),
        "v7_report": sha256(v7_report_path),
        "independent_failclosed_receipt": sha256(failclosed_path),
    }
    require(hashes["base_report"] == decision["base_report_sha256"], "base report hash mismatch")
    require(hashes["split_manifest"] == decision["split_manifest_sha256"], "split manifest hash mismatch")
    require(hashes["implementation"] == decision["script_sha256"], "implementation hash mismatch")
    require(hashes["checkpoint"] == report["inputs"]["checkpoint_sha256"], "report checkpoint hash mismatch")
    require(hashes["checkpoint"] == v7_report["inputs"]["checkpoint"]["sha256"], "v7 checkpoint hash mismatch")
    require(hashes["v7_report"] == manifest["source_hashes"]["v7_report"], "v7 report source hash mismatch")

    legacy_metadata = read_jsonl(args.v7_dir / "raw_legacy_writer_catalog.metadata.jsonl")
    ownership_path = args.frozen_dir / "ownership_fresh_acceptance.jsonl"
    ownership = read_jsonl(ownership_path)
    recomputed_legacy_split = splits_from_rows(legacy_metadata)
    recomputed_replay_split = splits_from_rows(ownership)
    require(recomputed_legacy_split == manifest["legacy"], "legacy split manifest is not reproducible")
    require(recomputed_replay_split == manifest["known_replay"], "known replay split manifest is not reproducible")

    legacy_aggregate, legacy_per_writer, legacy_predictions = legacy_reproduction(
        args.v7_dir, args.checkpoint, manifest, report
    )
    replay_aggregate, replay_per_writer, replay_predictions = replay_audit(
        args.run_dir / "known_replay_predictions.jsonl", ownership_path, manifest, report
    )
    legacy_regressions = writer_regressions(legacy_per_writer)
    replay_regressions = writer_regressions(replay_per_writer)
    legacy_paired = paired_counts(legacy_predictions)
    replay_paired = paired_counts(replay_predictions)
    replay_formula_paired = formula_paired_counts(replay_predictions)

    require([row["writer_id"] for row in legacy_regressions] == ["writer_004"], "unexpected legacy writer regressions")
    require(not replay_regressions, "known replay writer regression found")
    require(legacy_paired["improved"] == 8 and legacy_paired["regressed"] == 1, "legacy pairing mismatch")
    require(replay_paired["improved"] == 5 and replay_paired["regressed"] == 2, "replay pairing mismatch")
    require(replay_formula_paired["improved"] == 3 and replay_formula_paired["regressed"] == 2, "replay formula pairing mismatch")
    require_close(legacy_paired["mcnemar_exact_two_sided_p"], 0.0390625, "legacy McNemar")
    require_close(replay_paired["mcnemar_exact_two_sided_p"], 0.453125, "replay McNemar")
    require_close(replay_formula_paired["mcnemar_exact_two_sided_p"], 1.0, "replay formula McNemar")

    require(failclosed["status"] == "PRE_REPLAY_FAIL_CLOSED", "future run did not fail closed")
    require(failclosed["known_replay_opened"] is False, "future run opened known replay")
    require("identity" in failclosed["candidate_grid"], "identity candidate missing from future grid")
    eligible = [
        name
        for name, candidate in failclosed["candidate_grid"].items()
        if name != "identity" and safe_positive(candidate)
    ]
    require(not eligible, f"future grid unexpectedly has eligible candidates: {eligible}")
    forbidden_failclosed_outputs = [
        "known_replay_predictions.jsonl",
        "adapter_states.json",
        "split_manifest.json",
        "report.json",
    ]
    require(
        not any((args.failclosed_dir / name).exists() for name in forbidden_failclosed_outputs),
        "fail-closed directory contains post-replay outputs",
    )
    source_lines = args.implementation.read_text(encoding="utf-8").splitlines()
    gate_line = next(index for index, line in enumerate(source_lines, 1) if "if not globally_eligible" in line)
    replay_line = next(
        index
        for index, line in enumerate(source_lines, 1)
        if "replay_formulas, replay_ownership, replay_hashes = fresh_rows" in line
    )
    require(gate_line < replay_line, "known replay is opened before fail-closed gate")
    require(any("return 2" in line for line in source_lines[gate_line - 1 : replay_line - 1]), "gate has no pre-replay return")
    require(any("not _writer_regressions(rows) and positive" in line for line in source_lines), "safe-positive rule changed")

    require(decision["final_status"] == "AUTO_REJECTED", "decision is not AUTO_REJECTED")
    require(decision["product_promotion"] is False, "product promotion unexpectedly enabled")
    require(decision["legacy_all_writer_nonregression"] is False, "legacy non-regression decision mismatch")
    require(decision["known_replay_all_writer_nonregression"] is True, "replay non-regression decision mismatch")
    require(decision["candidate_violations"] == 0, "candidate violation decision mismatch")
    for key, expected in legacy_paired.items():
        require_close(expected, decision["legacy_paired_glyphs"][key], f"decision legacy paired {key}")
    for key, expected in replay_paired.items():
        if key in decision["known_replay_paired_glyphs"]:
            require_close(expected, decision["known_replay_paired_glyphs"][key], f"decision replay paired {key}")

    checks = {
        "receipt_hashes_match": True,
        "checkpoint_unchanged": True,
        "legacy_split_exactly_reproduced": True,
        "known_replay_split_exactly_reproduced": True,
        "legacy_prototype_200_reproduced_from_tensor_and_checkpoint": True,
        "known_replay_prediction_ledger_truth_and_metrics_reproduced": True,
        "candidate_top5_sets_preserved": True,
        "future_grid_contains_identity": True,
        "future_no_safe_positive_candidate": True,
        "future_known_replay_opened": False,
        "future_post_replay_outputs_absent": True,
        "source_failclosed_gate_precedes_replay_open": True,
        "promotion_ready": False,
    }
    output = {
        "schema": "aiflow-writer-adaptation-independent-audit/v8",
        "status": "AUTO_REJECT_CONFIRMED",
        "checks": checks,
        "hashes": hashes,
        "legacy": {
            "selected_config": report["selection"]["selected_config"],
            "aggregate": legacy_aggregate,
            "writer_regressions": legacy_regressions,
            "paired_glyphs": legacy_paired,
        },
        "known_replay": {
            "role": "diagnostic_not_promotion",
            "aggregate": replay_aggregate,
            "writer_regressions": replay_regressions,
            "paired_glyphs": replay_paired,
            "paired_formulae": replay_formula_paired,
        },
        "future_failclosed": {
            "receipt": str(failclosed_path.resolve()),
            "eligible_non_identity_candidates": eligible,
            "gate_line": gate_line,
            "known_replay_open_line": replay_line,
        },
        "decision": {
            "base_report_status": report["status"],
            "authoritative_final_status": decision["final_status"],
            "reason": decision["reason"],
            "product_promotion": False,
        },
        "limitations": [
            "The base report is retained as an immutable execution receipt; adoption_decision.json supersedes its historical PASS label.",
            "Known replay was already known and is diagnostic only, not untouched acceptance evidence.",
            "The legacy p-value follows a development-grid-selected configuration and is not confirmatory evidence.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy = legacy_aggregate
    replay = replay_aggregate
    markdown = f"""# Writer adaptation v8 독립 감사

## 판정

**AUTO_REJECT_CONFIRMED** — v8은 작가별 보정 가능성은 보였지만 제품 채택 조건은 충족하지 못했다. 원본 `report.json`은 당시 실행 영수증으로 보존되며, 최종 채택 판정은 `adoption_decision.json`의 `AUTO_REJECTED`가 우선한다.

| 구간 | Glyph Top-1 | Formula exact | Top-5 | 판정 |
|---|---:|---:|---:|---|
| Legacy baseline | {legacy['baseline']['top1']:.2%} | {legacy['baseline']['formula_exact']:.2%} | {legacy['baseline']['top5']:.2%} | 기준 |
| Legacy prototype_200 | {legacy['adapted']['top1']:.2%} | {legacy['adapted']['formula_exact']:.2%} | {legacy['adapted']['top5']:.2%} | writer_004 회귀로 거부 |
| Known replay baseline | {replay['baseline']['top1']:.2%} | {replay['baseline']['formula_exact']:.2%} | {replay['baseline']['top5']:.2%} | 진단 전용 |
| Known replay prototype_200 | {replay['adapted']['top1']:.2%} | {replay['adapted']['formula_exact']:.2%} | {replay['adapted']['top5']:.2%} | 승격 근거 아님 |

## 독립 확인

- v7 원시 텐서 393행과 동결 체크포인트에서 `prototype_200` 결과를 재계산했다.
- Legacy 300 glyph/73 formula의 집계와 작가별 수치가 원 보고서와 일치했다.
- writer_004는 Top-1 {legacy_regressions[0]['top1_delta'] * 100:+.2f}pp, formula exact {legacy_regressions[0]['formula_exact_delta'] * 100:+.2f}pp 회귀했다.
- Known replay 원장 163 glyph/45 formula의 ownership truth, evaluation-only 소속, 지표를 재계산했다.
- Replay glyph pairing은 5개 개선/2개 회귀, exact McNemar p={replay_paired['mcnemar_exact_two_sided_p']:.6f}; formula pairing은 3개 개선/2개 회귀, p={replay_formula_paired['mcnemar_exact_two_sided_p']:.1f}이다.
- 모든 adapted Top-5 집합은 frozen baseline Top-5와 동일하며 후보 위반은 0건이다.
- 현재 구현은 identity를 포함하고, Legacy 전 작가 비회귀와 양의 집계 개선을 함께 만족하는 후보가 없어 replay를 열기 전에 `PRE_REPLAY_FAIL_CLOSED`로 종료한다.
- HWR, 체크포인트, 제품 runtime, CROHME, MathWriting, v7은 변경되지 않았다.

## 해석 경계

- Legacy p={legacy_paired['mcnemar_exact_two_sided_p']:.7f}는 같은 개발 grid에서 선택된 설정의 값이므로 확증 통계로 사용할 수 없다.
- Known replay p={replay_paired['mcnemar_exact_two_sided_p']:.6f}는 약하고, 이미 알려진 writer replay라 promotion evidence가 아니다.
- 다음 확장은 새로운 untouched 작가/수식 acceptance set을 확보한 뒤 사전 고정된 단일 설정으로만 평가해야 한다.
"""
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output_json.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
