#!/usr/bin/env python3
"""Evaluate the formula review boundary without treating abstention as Top-1."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evaluate_48hz_prefix_v1 import _sha256
from formula_acceptance_guard_v1 import (
    AUTO_ACCEPTED, REVIEW_REQUIRED, load_configuration,
)


SCHEMA = "aiflow-formula-acceptance-guard-evaluation/v1"


def _d_file(path: Path, kind: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:" or not resolved.is_file():
        raise ValueError(f"{kind} must be an existing file on D:: {resolved}")
    return resolved


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _truth(path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    truth = {
        str(row["sample_id"]): [str(cell["token"]) for cell in row["target_cells"]]
        for row in rows
    }
    if len(truth) != len(rows):
        raise ValueError("duplicate formula ids in project truth")
    return truth, {str(row["sample_id"]): row for row in rows}


def _runtime(payload: dict[str, Any], expected: set[str], guarded: bool) -> dict[str, dict]:
    if (
        payload.get("schema") != "aiflow-raw-formula-context-runtime/v1"
        or payload.get("status") != "development_only_posthoc_shadow"
        or payload.get("audit", {}).get("product_default_enabled") is not False
    ):
        raise ValueError("formula runtime contract mismatch")
    rows = {str(row["formula_id"]): row for row in payload.get("formulas") or []}
    if set(rows) != expected or len(rows) != len(payload.get("formulas") or []):
        raise ValueError("formula runtime coverage mismatch")
    if guarded and any(
        row.get("decision_status") not in {AUTO_ACCEPTED, REVIEW_REQUIRED}
        for row in rows.values()
    ):
        raise ValueError("guarded runtime decision coverage mismatch")
    return rows


def _score(
    rows: dict[str, dict], truth: dict[str, list[str]], formula_ids: list[str],
) -> dict[str, Any]:
    exact_ids = [
        formula_id for formula_id in formula_ids
        if rows[formula_id]["finalized_tokens"] == truth[formula_id]
    ]
    error_ids = [formula_id for formula_id in formula_ids if formula_id not in exact_ids]
    accepted = [
        formula_id for formula_id in formula_ids
        if rows[formula_id]["decision_status"] == AUTO_ACCEPTED
    ]
    reviewed = [formula_id for formula_id in formula_ids if formula_id not in accepted]
    accepted_exact = [formula_id for formula_id in accepted if formula_id in exact_ids]
    accepted_errors = [formula_id for formula_id in accepted if formula_id in error_ids]
    reviewed_errors = [formula_id for formula_id in reviewed if formula_id in error_ids]
    return {
        "formula_exact": len(exact_ids),
        "total": len(formula_ids),
        "formula_exact_rate": len(exact_ids) / max(len(formula_ids), 1),
        "auto_accepted": len(accepted),
        "auto_accept_coverage": len(accepted) / max(len(formula_ids), 1),
        "auto_accepted_exact": len(accepted_exact),
        "auto_accepted_errors": len(accepted_errors),
        "auto_accepted_precision": len(accepted_exact) / max(len(accepted), 1),
        "review_required": len(reviewed),
        "reviewed_errors": len(reviewed_errors),
        "reviewed_correct": len(reviewed) - len(reviewed_errors),
        "error_capture_rate": len(reviewed_errors) / max(len(error_ids), 1),
        "reviewed_error_ids": reviewed_errors,
        "remaining_auto_accept_error_ids": accepted_errors,
    }


def evaluate(
    truth_path: Path, baseline_path: Path, guarded_path: Path,
    configuration_path: Path, crohme_path: Path,
) -> dict[str, Any]:
    paths = {
        "truth": _d_file(truth_path, "project truth"),
        "baseline": _d_file(baseline_path, "baseline runtime"),
        "guarded": _d_file(guarded_path, "guarded runtime"),
        "configuration": _d_file(configuration_path, "acceptance configuration"),
        "crohme": _d_file(crohme_path, "CROHME diagnostic"),
    }
    _configuration, configuration_sha256 = load_configuration(paths["configuration"])
    truth, metadata = _truth(paths["truth"])
    baseline_payload = _json(paths["baseline"])
    guarded_payload = _json(paths["guarded"])
    baseline = _runtime(baseline_payload, set(truth), False)
    guarded = _runtime(guarded_payload, set(truth), True)
    guard_audit = dict(guarded_payload.get("audit", {}).get("formula_acceptance_guard") or {})
    if (
        guard_audit.get("schema") != "aiflow-formula-acceptance-guard/v1"
        or guard_audit.get("configuration_sha256") != configuration_sha256
        or guard_audit.get("product_default_enabled") is not False
        or guard_audit.get("token_mutations") != 0
        or guard_audit.get("grouping_mutations") != 0
        or guard_audit.get("arithmetic_evaluation") is not False
    ):
        raise ValueError("formula acceptance guard audit mismatch")
    for formula_id in truth:
        before = baseline[formula_id]
        after = guarded[formula_id]
        if (
            before["finalized_tokens"] != after["finalized_tokens"]
            or before["groups"] != after["groups"]
            or before["symbols"] != after["symbols"]
        ):
            raise ValueError(f"formula acceptance guard mutated output: {formula_id}")

    formula_ids = sorted(truth)
    subsets = {
        "all_159": formula_ids,
        "public_web_110": [
            formula_id for formula_id in formula_ids
            if metadata[formula_id].get("source_partition") == "public-web-collection"
        ],
        "owned_phone_replay_49": [
            formula_id for formula_id in formula_ids
            if metadata[formula_id].get("source_partition") == "owned-phone-replay"
        ],
    }
    if {name: len(ids) for name, ids in subsets.items()} != {
        "all_159": 159, "public_web_110": 110, "owned_phone_replay_49": 49,
    }:
        raise ValueError("project evaluation subset coverage mismatch")
    project_scores = {
        name: _score(guarded, truth, ids) for name, ids in subsets.items()
    }
    reason_counts = Counter(
        reason for row in guarded.values()
        for reason in row["acceptance"]["review_reasons"]
    )

    crohme = _json(paths["crohme"])
    crohme_acceptance = dict(crohme.get("formula_acceptance_guard") or {})
    if (
        crohme.get("schema") != "aiflow-formula-context-crohme-diagnostic/v1"
        or crohme.get("status") != "repeated_noncommercial_diagnostic_only"
        or crohme.get("product_validation") is not False
        or crohme_acceptance.get("enabled") is not True
        or crohme_acceptance.get("configuration_sha256") != configuration_sha256
        or crohme_acceptance.get("candidate_preserving") is not True
        or crohme_acceptance.get("token_mutations") != 0
        or crohme_acceptance.get("grouping_mutations") != 0
        or crohme_acceptance.get("arithmetic_evaluation") is not False
        or crohme_acceptance.get("product_validation") is not False
    ):
        raise ValueError("CROHME formula acceptance diagnostic mismatch")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_posthoc_shadow",
        "decision": "retain_as_shadow_only_not_product",
        "top1_formula_exact_unchanged": True,
        "project_selective_scores": project_scores,
        "project_review_reasons": dict(sorted(reason_counts.items())),
        "crohme_repeated_noncommercial_diagnostic": crohme_acceptance,
        "contracts": {
            "candidate_preserving": True,
            "token_mutations": 0,
            "grouping_mutations": 0,
            "glyph_order_mutations": 0,
            "target_label_or_glyph_count_input": False,
            "writer_identity_input": False,
            "arithmetic_evaluation": False,
            "raw_fallback_on_review_required": True,
            "product_default_enabled": False,
        },
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "limits": [
            "project thresholds were inspected against the same 159 formulas",
            "selective precision is not a replacement for all-formula Top-1 exact",
            "CROHME is repeated noncommercial diagnostic evidence only",
            "fresh commercial writer/formula-disjoint acceptance is required before promotion",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--guarded", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--crohme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.drive.upper() != "D:" or output.exists():
        parser.error("--output must be a new file on D:")
    report = evaluate(
        args.truth, args.baseline, args.guarded, args.configuration, args.crohme,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output), "sha256": _sha256(output),
        "decision": report["decision"],
        "project_selective_scores": report["project_selective_scores"],
        "crohme": report["crohme_repeated_noncommercial_diagnostic"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
