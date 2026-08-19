#!/usr/bin/env python3
"""Restore a straight two-stroke equality sign in a numeric relation slot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import _sha256
from wide_candidate_syntax_rescue_v1 import DIGITS


SCHEMA = "aiflow-straight-equality-slot-rescue/v1"
OUTPUT_SCHEMA = "aiflow-straight-equality-slot-finalized/v1"
CONFIG_SCHEMA = "aiflow-straight-equality-slot-rescue-runtime-config/v1"
DEFAULT_CONFIGURATION = {
    "candidate_width": 20,
    "auxiliary_weight": 0.6,
    "allowed_auxiliary_policies": [
        "current_expanded_writer_loo_probability_fusion",
        "old_new_product_probability_fusion",
    ],
    "replacement": "=",
    "maximum_candidate_rank": 5,
    "minimum_candidate_probability": 0.04,
    "required_stroke_count": 2,
    "minimum_aspect_log": 0.8,
    "maximum_path_over_diagonal": 1.55,
    "maximum_absolute_direction_y": 0.2,
    "minimum_formula_length": 3,
    "maximum_formula_length": 64,
    "maximum_changes_per_formula": 1,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("straight-equality configuration fields mismatch")
    output = {
        **configuration,
        "candidate_width": int(configuration["candidate_width"]),
        "auxiliary_weight": float(configuration["auxiliary_weight"]),
        "allowed_auxiliary_policies": [
            str(value) for value in configuration["allowed_auxiliary_policies"]
        ],
        "replacement": str(configuration["replacement"]),
        "maximum_candidate_rank": int(configuration["maximum_candidate_rank"]),
        "minimum_candidate_probability": float(
            configuration["minimum_candidate_probability"]
        ),
        "required_stroke_count": int(configuration["required_stroke_count"]),
        "minimum_aspect_log": float(configuration["minimum_aspect_log"]),
        "maximum_path_over_diagonal": float(
            configuration["maximum_path_over_diagonal"]
        ),
        "maximum_absolute_direction_y": float(
            configuration["maximum_absolute_direction_y"]
        ),
        "minimum_formula_length": int(configuration["minimum_formula_length"]),
        "maximum_formula_length": int(configuration["maximum_formula_length"]),
        "maximum_changes_per_formula": int(
            configuration["maximum_changes_per_formula"]
        ),
    }
    numeric = (
        output["auxiliary_weight"], output["minimum_candidate_probability"],
        output["minimum_aspect_log"], output["maximum_path_over_diagonal"],
        output["maximum_absolute_direction_y"],
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("straight-equality numeric configuration is not finite")
    if output["candidate_width"] != 20 or output["replacement"] != "=":
        raise ValueError("straight-equality requires Top-20 evidence and '='")
    if not 1 <= output["maximum_candidate_rank"] <= output["candidate_width"]:
        raise ValueError("straight-equality candidate rank is invalid")
    if not 0.0 <= output["minimum_candidate_probability"] <= 1.0:
        raise ValueError("straight-equality probability is invalid")
    if output["required_stroke_count"] != 2:
        raise ValueError("straight-equality requires two strokes")
    if not 0.0 <= output["auxiliary_weight"] <= 1.0:
        raise ValueError("straight-equality auxiliary weight is invalid")
    if output["minimum_aspect_log"] < 0.0:
        raise ValueError("straight-equality aspect threshold is invalid")
    if not 1.0 <= output["maximum_path_over_diagonal"]:
        raise ValueError("straight-equality path threshold is invalid")
    if not 0.0 <= output["maximum_absolute_direction_y"] <= 1.0:
        raise ValueError("straight-equality direction threshold is invalid")
    if not (
        3 <= output["minimum_formula_length"]
        <= output["maximum_formula_length"] <= 64
        and output["maximum_changes_per_formula"] == 1
    ):
        raise ValueError("straight-equality formula contract is invalid")
    policies = output["allowed_auxiliary_policies"]
    if not policies or len(policies) != len(set(policies)) or any(not value for value in policies):
        raise ValueError("straight-equality candidate policies are invalid")
    return output


def _candidate_map(rows: list[dict], configuration: dict) -> dict[str, dict]:
    output = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        tokens = [str(value) for value in row.get("final_topk", [])]
        probabilities = [float(value) for value in row.get(
            "final_topk_probabilities", []
        )]
        if not record_id or record_id in output:
            raise ValueError(f"straight-equality candidate identity is invalid: {record_id}")
        if (
            len(tokens) != configuration["candidate_width"]
            or len(tokens) != len(set(tokens))
            or len(probabilities) != len(tokens)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(left < right for left, right in zip(probabilities, probabilities[1:]))
        ):
            raise ValueError(f"straight-equality candidate array is invalid: {record_id}")
        policy = str(row.get("hwr_policy", ""))
        weight = float(row.get("hwr_fusion_weight", -1.0))
        if policy not in configuration["allowed_auxiliary_policies"]:
            raise ValueError(f"straight-equality candidate policy is invalid: {record_id}")
        if not math.isclose(weight, configuration["auxiliary_weight"], abs_tol=1e-12):
            raise ValueError(f"straight-equality candidate weight is invalid: {record_id}")
        output[record_id] = {**row, "tokens": tokens, "probabilities": probabilities}
    return output


def _formulae(rows: list[dict]) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row["formula_id"])].append(row)
    for formula_id, sequence in output.items():
        sequence.sort(key=lambda row: int(row["context_index"]))
        if [int(row["context_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"straight-equality formula order is invalid: {formula_id}")
    return dict(output)


def _eligible(candidate: dict, configuration: dict) -> tuple[bool, dict]:
    replacement = configuration["replacement"]
    tokens = candidate["tokens"]
    if replacement not in tokens:
        return False, {}
    index = tokens.index(replacement)
    probability = candidate["probabilities"][index]
    geometry = candidate.get("geometry", {})
    try:
        evidence = {
            "candidate_rank": index + 1,
            "candidate_probability": probability,
            "aspect_log": float(geometry["aspect_log"]),
            "path_over_diag": float(geometry["path_over_diag"]),
            "direction_y": float(geometry["direction_y"]),
            "stroke_count": int(round(float(geometry["stroke_count"]))),
        }
    except (KeyError, TypeError, ValueError):
        return False, {}
    passed = bool(
        evidence["candidate_rank"] <= configuration["maximum_candidate_rank"]
        and evidence["candidate_probability"]
        >= configuration["minimum_candidate_probability"]
        and evidence["stroke_count"] == configuration["required_stroke_count"]
        and evidence["aspect_log"] >= configuration["minimum_aspect_log"]
        and evidence["path_over_diag"]
        <= configuration["maximum_path_over_diagonal"]
        and abs(evidence["direction_y"])
        <= configuration["maximum_absolute_direction_y"]
    )
    return passed, evidence


def apply_straight_equality_slot_rescue(
    baseline_rows: list[dict], candidate_rows: list[dict],
    configuration: dict | None = None,
) -> tuple[list[dict], dict]:
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    candidates = _candidate_map(candidate_rows, configuration)
    baseline_ids = [str(row["record_id"]) for row in baseline_rows]
    if not baseline_ids or len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("straight-equality baseline IDs are invalid")
    if set(baseline_ids) != set(candidates):
        raise ValueError("straight-equality coverage mismatch")
    selected = {
        str(row["record_id"]): str(row["finalized_top1"])
        for row in baseline_rows
    }
    changes = []
    skipped = Counter()
    replacement = configuration["replacement"]
    formulae = _formulae(baseline_rows)
    for formula_id, sequence in formulae.items():
        before = [str(row["finalized_top1"]) for row in sequence]
        if not configuration["minimum_formula_length"] <= len(before) <= configuration[
            "maximum_formula_length"
        ]:
            skipped["formula_length"] += 1
            continue
        if replacement in before:
            skipped["existing_equality"] += 1
            continue
        eligible = []
        for index in range(1, len(sequence) - 1):
            if before[index - 1] not in DIGITS or before[index + 1] not in DIGITS:
                continue
            row = sequence[index]
            record_id = str(row["record_id"])
            candidate = candidates[record_id]
            if str(candidate.get("formula_id", "")) != formula_id:
                raise ValueError(f"straight-equality formula identity mismatch: {record_id}")
            passed, evidence = _eligible(candidate, configuration)
            if passed:
                eligible.append((index, row, evidence))
        if not eligible:
            skipped["no_unique_straight_equality_slot"] += 1
            continue
        if len(eligible) != configuration["maximum_changes_per_formula"]:
            skipped["ambiguous_straight_equality_slots"] += 1
            continue
        index, row, evidence = eligible[0]
        record_id = str(row["record_id"])
        selected[record_id] = replacement
        changes.append({
            "formula_id": formula_id,
            "record_id": record_id,
            "context_index": index,
            "before": before[index],
            "after": replacement,
            **evidence,
            "rule": "digit_straight_two_stroke_equality_digit",
        })
    output = []
    for row in baseline_rows:
        record_id = str(row["record_id"])
        token = selected[record_id]
        changed = token != str(row["finalized_top1"])
        output.append({
            **row,
            "schema": OUTPUT_SCHEMA,
            "baseline_top1_before_straight_equality_rescue": str(
                row["finalized_top1"]
            ),
            "finalized_top1": token,
            "changed_by_straight_equality_rescue": changed,
            "decision_source": (
                "straight_equality_slot_rescue_v1"
                if changed else str(row.get("decision_source", "baseline"))
            ),
        })
    return output, {
        "schema": SCHEMA,
        "status": "shadow_runtime_only",
        "configuration": configuration,
        "records": len(output),
        "formulas": len(formulae),
        "changed_glyphs": len(changes),
        "changed_formulas": len({change["formula_id"] for change in changes}),
        "changes": changes,
        "skipped": dict(sorted(skipped.items())),
        "candidate_preservation_rate": 1.0,
        "inserted_or_deleted_glyphs": 0,
        "glyph_order_mutations": 0,
        "grouping_mutations": 0,
        "layout_mutations": 0,
        "arithmetic_evaluation": False,
    }


def _write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    baseline = [
        {"record_id": f"r{i}", "formula_id": "f", "context_index": i,
         "finalized_top1": token}
        for i, token in enumerate(["7", r"\approx", "1"])
    ]
    candidates = []
    for index, token in enumerate(["7", r"\approx", "1"]):
        tokens = ([r"\div", r"\curvearrowright", "="] if index == 1 else [token])
        tokens += [f"z{index}_{offset}" for offset in range(20 - len(tokens))]
        candidates.append({
            "record_id": f"r{index}", "formula_id": "f",
            "final_topk": tokens,
            "final_topk_probabilities": (
                [0.5, 0.3, 0.15] + [0.0] * 17 if index == 1
                else [1.0] + [0.0] * 19
            ),
            "hwr_policy": "old_new_product_probability_fusion",
            "hwr_fusion_weight": 0.6,
            "geometry": {
                "stroke_count": 2.0, "aspect_log": 1.0,
                "path_over_diag": 1.1, "direction_y": 0.0,
            },
        })
    output, audit = apply_straight_equality_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["7", "=", "1"]
    assert audit["changed_glyphs"] == 1 and audit["arithmetic_evaluation"] is False
    candidates[1]["geometry"]["path_over_diag"] = 1.6
    output, audit = apply_straight_equality_slot_rescue(baseline, candidates)
    assert [row["finalized_top1"] for row in output] == ["7", r"\approx", "1"]
    assert audit["changed_glyphs"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-finalized", type=Path)
    parser.add_argument("--top20-candidates", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print('{"self_test":"pass"}')
        return 0
    if any(value is None for value in (
        args.baseline_finalized, args.top20_candidates, args.config, args.output,
    )):
        parser.error("baseline, Top-20 candidates, config, and output are required")
    baseline_path = args.baseline_finalized.expanduser().resolve()
    candidate_path = args.top20_candidates.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not all(path.is_file() for path in (baseline_path, candidate_path, config_path)):
        parser.error("straight-equality runtime input is missing")
    if output_path.exists():
        parser.error(f"refusing to overwrite straight-equality output: {output_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != CONFIG_SCHEMA
        or payload.get("rescue_schema") != SCHEMA
        or gate.get("shadow_runtime_admitted") is not True
        or gate.get("product_default_admitted") is not False
    ):
        parser.error("straight-equality shadow runtime config is not admitted")
    rows, audit = apply_straight_equality_slot_rescue(
        list(_json_lines(baseline_path)), list(_json_lines(candidate_path)),
        payload["configuration"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, rows)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "status": "shadow_runtime_only",
        "changed_glyphs": audit["changed_glyphs"],
        "changed_formulas": audit["changed_formulas"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
