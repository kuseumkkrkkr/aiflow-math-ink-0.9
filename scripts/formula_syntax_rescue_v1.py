#!/usr/bin/env python3
"""Restore one strongly implied formula token without evaluating arithmetic."""

from __future__ import annotations

from collections import Counter
import math

import train_masked_context_reranker_v1 as masked
from formula_sequence_guard_v1 import (
    DIGITS, _sequence_score, valid_numeric_sequence,
)


SCHEMA = "aiflow-formula-syntax-rescue/v1"
EQUALITY_EVIDENCE = frozenset({
    "=", r"\approx", r"\asymp", r"\cong", r"\doteq", r"\equiv",
    r"\neq", r"\sim", r"\simeq",
})
DEFAULT_CONFIGURATION = {
    "shape_weight": 0.0,
    "context_weight": 0.25,
    "grammar_weight": 0.5,
    "retention_bonus": 0.25,
    "margin": 0.5,
    "minimum_candidate_probability_ratio": 0.0005,
}


def validate_configuration(configuration: dict) -> dict:
    if set(configuration) != set(DEFAULT_CONFIGURATION):
        raise ValueError("formula syntax rescue configuration fields mismatch")
    output = {}
    for key in DEFAULT_CONFIGURATION:
        value = float(configuration[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid formula syntax rescue configuration: {key}")
        output[key] = value
    if output["minimum_candidate_probability_ratio"] > 1.0:
        raise ValueError("formula syntax rescue probability ratio must not exceed one")
    return output


def _allowed_change(
    sequence: list[dict], baseline: list[str], index: int, token: str,
) -> str | None:
    if (
        token == "+" and len(sequence) == 3 and index == 1
        and all(value in DIGITS for value in baseline)
    ):
        return "three_digit_center_plus"
    if (
        token in EQUALITY_EVIDENCE
        and str(sequence[index]["final_topk"][0]) in EQUALITY_EVIDENCE
    ):
        return "hwr_equality_family_relation"
    return None


def apply_formula_syntax_rescue(
    rows: list[dict], predictions: dict[str, str], components: dict,
    grammar: dict, configuration: dict | None = None,
) -> tuple[dict[str, str], dict]:
    """Repair one missing structural token in an otherwise valid numeric row."""
    configuration = validate_configuration(configuration or DEFAULT_CONFIGURATION)
    record_ids = {str(row["record_id"]) for row in rows}
    if set(predictions) != record_ids:
        raise ValueError("formula syntax rescue prediction coverage mismatch")
    component_rows = components.get("rows") or []
    if {str(row["record_id"]) for row in component_rows} != record_ids:
        raise ValueError("formula syntax rescue component coverage mismatch")
    candidate_context = {
        str(row["record_id"]): {
            str(token): float(
                components["candidate_context"][row_index, candidate_index]
            )
            for candidate_index, token in enumerate(row["final_topk"])
        }
        for row_index, row in enumerate(component_rows)
    }

    output = {record_id: str(token) for record_id, token in predictions.items()}
    changes = []
    skipped = Counter()
    for formula_id, sequence in masked._formulae(rows).items():
        baseline = [output[str(row["record_id"])] for row in sequence]
        if not valid_numeric_sequence(baseline):
            skipped["not_valid_numeric_baseline"] += 1
            continue
        scored = [(
            _sequence_score(
                sequence, baseline, baseline, candidate_context, grammar,
                configuration,
            ),
            baseline,
            None,
        )]
        for index, row in enumerate(sequence):
            before_probability = float(
                row["final_topk_probabilities"][
                    [str(value) for value in row["final_topk"]].index(baseline[index])
                ]
            )
            for candidate_index, candidate in enumerate(row["final_topk"]):
                token = str(candidate)
                rule = _allowed_change(sequence, baseline, index, token)
                if rule is None or token == baseline[index]:
                    continue
                probability = float(row["final_topk_probabilities"][candidate_index])
                if (
                    probability / max(before_probability, 1e-12)
                    < configuration["minimum_candidate_probability_ratio"]
                ):
                    continue
                proposed = list(baseline)
                proposed[index] = token
                if not valid_numeric_sequence(proposed):
                    continue
                scored.append((
                    _sequence_score(
                        sequence, proposed, baseline, candidate_context, grammar,
                        configuration,
                    ),
                    proposed,
                    rule,
                ))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best, rule = scored[0]
        if best == baseline:
            skipped["retained_by_score"] += 1
            continue
        runner_up = scored[1][0]
        if best_score - runner_up < configuration["margin"]:
            skipped["insufficient_margin"] += 1
            continue
        changed = [
            index for index, (before, after) in enumerate(zip(baseline, best, strict=True))
            if before != after
        ]
        if len(changed) != 1 or rule is None:
            raise AssertionError("formula syntax rescue exceeded one change")
        index = changed[0]
        record_id = str(sequence[index]["record_id"])
        output[record_id] = best[index]
        changes.append({
            "formula_id": str(formula_id), "record_id": record_id,
            "before": baseline[index], "after": best[index], "rule": rule,
            "score_margin": best_score - runner_up,
        })

    if any(output[str(row["record_id"])] not in row["final_topk"] for row in rows):
        raise AssertionError("formula syntax rescue invented a candidate")
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
        "arithmetic_evaluation": False,
    }


def _self_test() -> None:
    import numpy as np
    import train_context_decision_layer_v1 as context

    rows = []
    for index, candidates in enumerate((["5"], ["4", "+"], ["4"])):
        rows.append({
            "record_id": f"r{index}", "formula_id": "plus",
            "label": ("5", "+", "4")[index],
            "final_topk": list(candidates),
            "final_topk_probabilities": (
                [1.0] if len(candidates) == 1 else [0.999, 0.001]
            ),
            "context": {"index": index, "length": 3},
            "geometry": {},
        })
    scores = np.full((3, 2), -1e9, dtype=np.float32)
    scores[0, 0], scores[1, :2], scores[2, 0] = 0.0, (-2.0, 2.0), 0.0
    components = {"rows": rows, "candidate_context": scores}
    grammar = context._fit_role_grammar(rows)
    baseline = {row["record_id"]: row["final_topk"][0] for row in rows}
    rescued, audit = apply_formula_syntax_rescue(rows, baseline, components, grammar)
    assert [rescued[f"r{index}"] for index in range(3)] == ["5", "+", "4"]
    assert audit["changed"] == 1

    rows[1]["final_topk"] = ["4", "="]
    rows[1]["final_topk_probabilities"] = [0.8, 0.2]
    preserved, _ = apply_formula_syntax_rescue(rows, baseline, components, grammar)
    assert preserved == baseline

    equality_rows = []
    equality_truth = ["1", "2", "-", "0", "=", "1", "2"]
    equality_baseline = ["1", "2", "-", "0", "1", "1", "2"]
    for index, token in enumerate(equality_truth):
        candidates = ["=", "1"] if index == 4 else [token]
        equality_rows.append({
            "record_id": f"e{index}", "formula_id": "equality",
            "label": token, "final_topk": candidates,
            "final_topk_probabilities": [0.6, 0.4] if index == 4 else [1.0],
            "context": {"index": index, "length": len(equality_truth)},
            "geometry": {},
        })
    equality_scores = np.full((len(equality_rows), 2), -1e9, dtype=np.float32)
    equality_scores[:, 0] = 0.0
    equality_scores[4, :2] = (3.0, 0.0)
    equality_components = {
        "rows": equality_rows, "candidate_context": equality_scores,
    }
    equality_predictions = {
        row["record_id"]: equality_baseline[index]
        for index, row in enumerate(equality_rows)
    }
    equality_grammar = context._fit_role_grammar(equality_rows)
    rescued, audit = apply_formula_syntax_rescue(
        equality_rows, equality_predictions, equality_components,
        equality_grammar,
    )
    assert [rescued[f"e{index}"] for index in range(7)] == equality_truth
    assert audit["changes"][0]["rule"] == "hwr_equality_family_relation"


if __name__ == "__main__":
    _self_test()
    print('{"self_test":"pass"}')
