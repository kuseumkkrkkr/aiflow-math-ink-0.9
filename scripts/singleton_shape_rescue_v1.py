#!/usr/bin/env python3
"""Use admitted auxiliary HWR evidence only for context-free singleton glyphs."""

from __future__ import annotations

from collections import Counter
import math

import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-singleton-shape-rescue/v1"
DEFAULT_CONFIGURATION = {
    "auxiliary_policy": "current_expanded_writer_loo_probability_fusion",
    "auxiliary_weight": 0.6,
    "token_confidence_thresholds": {"/": 0.50, r"\times": 0.45},
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("singleton shape rescue configuration fields mismatch")
    policy = str(configuration["auxiliary_policy"])
    if not policy:
        raise ValueError("singleton shape rescue auxiliary policy must be non-empty")
    weight = float(configuration["auxiliary_weight"])
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("singleton shape rescue auxiliary weight must be in [0,1]")
    thresholds = configuration["token_confidence_thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"/", r"\times"}:
        raise ValueError("singleton shape rescue token thresholds mismatch")
    clean_thresholds = {}
    for token, raw in thresholds.items():
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid singleton shape threshold: {token}")
        clean_thresholds[str(token)] = value
    return {
        "auxiliary_policy": policy,
        "auxiliary_weight": weight,
        "token_confidence_thresholds": clean_thresholds,
    }


def _evidence_by_record(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if not record_id or record_id in output:
            raise ValueError(f"singleton shape evidence record_id is invalid: {record_id!r}")
        if str(row.get("hwr_policy", "")) != configuration["auxiliary_policy"]:
            raise ValueError(f"singleton shape evidence policy mismatch: {record_id}")
        weight = float(row.get("hwr_fusion_weight", -1.0))
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"singleton shape evidence weight mismatch: {record_id}")
        candidates = [str(value) for value in row.get("final_topk", [])]
        probabilities = [float(value) for value in row.get("final_topk_probabilities", [])]
        if (
            not candidates or len(candidates) != len(probabilities)
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"singleton shape evidence candidates are invalid: {record_id}")
        output[record_id] = {
            "formula_id": str(row.get("formula_id", "")),
            "token": candidates[0],
            "confidence": probabilities[0],
        }
    return output


def apply_singleton_shape_rescue(
    rows: list[dict], predictions: dict[str, str], auxiliary_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[dict[str, str], dict]:
    """Select one original candidate when auxiliary shape evidence is strong."""
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("singleton shape rescue prediction coverage mismatch")
    evidence = _evidence_by_record(auxiliary_rows, configuration)
    if set(evidence) != record_ids:
        raise ValueError("singleton shape rescue evidence coverage mismatch")
    output = {record_id: str(token) for record_id, token in predictions.items()}
    changes = []
    skipped = Counter()
    thresholds = configuration["token_confidence_thresholds"]
    for formula_id, sequence in masked._formulae(rows).items():
        if len(sequence) != 1:
            skipped["context_available"] += 1
            continue
        row = sequence[0]
        record_id = str(row["record_id"])
        item = evidence[record_id]
        if item["formula_id"] != str(formula_id):
            raise ValueError(f"singleton shape evidence formula mismatch: {record_id}")
        token = str(item["token"])
        threshold = thresholds.get(token)
        if threshold is None:
            skipped["token_not_admitted"] += 1
            continue
        if float(item["confidence"]) < threshold:
            skipped["below_confidence_threshold"] += 1
            continue
        if token not in row["final_topk"]:
            skipped["outside_original_top5"] += 1
            continue
        before = output[record_id]
        if before == token:
            skipped["already_selected"] += 1
            continue
        output[record_id] = token
        changes.append({
            "formula_id": str(formula_id),
            "record_id": record_id,
            "before": before,
            "after": token,
            "auxiliary_confidence": float(item["confidence"]),
            "threshold": float(threshold),
        })
    if any(output[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("singleton shape rescue invented a candidate")
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "formulas": len(masked._formulae(rows)),
        "changed": len(changes),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "new_tokens": 0,
        "deleted_glyphs": 0,
        "grouping_mutations": 0,
    }


def _self_test() -> None:
    rows = [{
        "record_id": "r", "formula_id": "f",
        "final_topk": ["1", "/"], "final_topk_probabilities": [0.8, 0.2],
        "context": {"index": 0, "length": 1}, "geometry": {},
    }]
    evidence = [{
        "record_id": "r", "formula_id": "f",
        "final_topk": ["/", "1"], "final_topk_probabilities": [0.51, 0.49],
        "hwr_policy": DEFAULT_CONFIGURATION["auxiliary_policy"],
        "hwr_fusion_weight": DEFAULT_CONFIGURATION["auxiliary_weight"],
    }]
    rescued, audit = apply_singleton_shape_rescue(rows, {"r": "1"}, evidence)
    assert rescued == {"r": "/"} and audit["changed"] == 1
    evidence[0]["final_topk_probabilities"] = [0.49, 0.48]
    preserved, _ = apply_singleton_shape_rescue(rows, {"r": "1"}, evidence)
    assert preserved == {"r": "1"}


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
