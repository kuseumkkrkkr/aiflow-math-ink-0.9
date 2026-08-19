#!/usr/bin/env python3
"""Finalize formula characters inside immutable HWR Top-k candidates.

The runtime never receives truth labels and never creates a token, deletes a
glyph, or changes stroke ownership. Load ``OwnedFormulaContextFinalizer`` once
per process, then call ``finalize`` for each candidate batch. The selected r6
context model is followed by a support-aware exact-context gate, then equation,
locked-fence, and horizontal-infix guards.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from semantic_equation_guard_v2 import apply_semantic_equation_guard_v2
from semantic_fence_guard_v1 import apply_semantic_fence_guard
from semantic_infix_guard_v1 import apply_semantic_infix_guard
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import (
    decide_owned_formula_rows_supported_exact,
    load_owned_formula_context,
)


DEFAULT_CONTEXT = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "owned_formula_context_20260822_r6"
    / "owned_formula_context_product.pt"
)
REQUIRED = {
    "record_id", "formula_id", "final_topk", "final_topk_probabilities",
    "context", "geometry",
}


def _runtime_rows(rows: list[dict], labels: set[str]) -> list[dict]:
    if not rows:
        raise ValueError("context finalizer requires at least one candidate row")
    clean = []
    record_ids = set()
    for source in rows:
        missing = REQUIRED - set(source)
        if missing:
            raise ValueError(f"candidate row missing fields: {sorted(missing)}")
        row = {key: value for key, value in source.items() if key != "label"}
        record_id = str(row["record_id"])
        if not record_id or record_id in record_ids:
            raise ValueError(f"candidate record_id must be unique: {record_id!r}")
        record_ids.add(record_id)
        candidates = [str(value) for value in row["final_topk"]]
        probabilities = [float(value) for value in row["final_topk_probabilities"]]
        if not 1 <= len(candidates) <= 5 or len(candidates) != len(set(candidates)):
            raise ValueError(f"candidate Top-k is invalid: {record_id}")
        if any(value not in labels for value in candidates):
            raise ValueError(f"candidate token outside frozen HWR vocabulary: {record_id}")
        if len(probabilities) != len(candidates) or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities
        ):
            raise ValueError(f"candidate probabilities are invalid: {record_id}")
        if any(left < right for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError(f"candidate probabilities must be descending: {record_id}")
        row["record_id"] = record_id
        row["formula_id"] = str(row["formula_id"])
        if not row["formula_id"]:
            raise ValueError(f"candidate formula_id must be non-empty: {record_id}")
        row["final_topk"] = candidates
        row["final_topk_probabilities"] = probabilities
        clean.append(row)
    masked._formulae(clean)
    return clean


def _decision_metadata(
    row: dict, hwr: str, context_token: str, equation_token: str,
    fence_token: str, final_token: str, supported_exact: bool = False,
) -> tuple[bool, str, str]:
    context_available = int(row["context"]["length"]) > 1
    if not context_available:
        source = "context_prior_only" if final_token != hwr else "hwr_top1_no_context"
    elif final_token != fence_token:
        source = "semantic_infix_guard"
    elif fence_token != equation_token:
        source = "semantic_fence_guard"
    elif equation_token != context_token:
        source = "semantic_equation_guard_v2"
    elif supported_exact:
        source = "owned_supported_exact_context_guard"
    elif context_token != hwr:
        source = "owned_formula_context"
    else:
        source = "owned_context_retained_hwr"
    status = (
        "ambiguous_no_formula_context"
        if not context_available and len(row["final_topk"]) > 1
        else "finalized"
    )
    return context_available, status, source


class OwnedFormulaContextFinalizer:
    def __init__(
        self, checkpoint: Path, hwr_checkpoint: Path = DEFAULT_PRODUCT,
        *, device: str = "auto", batch_size: int = 128,
        semantic_guards: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        if resolved_device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device: {resolved_device}")
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.device = torch.device(resolved_device)
        self.batch_size = batch_size
        self.semantic_guards = semantic_guards
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.hwr_checkpoint_sha256 = _sha256(hwr_checkpoint)
        self.model, self.contract, self.payload = load_owned_formula_context(
            checkpoint, hwr_checkpoint, self.device
        )
        self.labels = set(self.contract["label_to_index"])

    def finalize(self, rows: list[dict]) -> tuple[list[dict], dict]:
        runtime_rows = _runtime_rows(rows, self.labels)
        context_predictions, model_audit = decide_owned_formula_rows_supported_exact(
            self.model, self.contract, self.payload, runtime_rows,
            self.device, self.batch_size,
        )
        supported_exact_records = {
            str(change["record_id"])
            for change in model_audit["supported_exact_context"]["changes"]
        }
        if self.semantic_guards:
            probability_ratio_floor = float(
                self.payload["configuration"]["candidate_probability_ratio_floor"]
            )
            equation_predictions, equation_audit = apply_semantic_equation_guard_v2(
                runtime_rows, context_predictions, probability_ratio_floor
            )
            fence_predictions, fence_audit = apply_semantic_fence_guard(
                runtime_rows, equation_predictions
            )
            predictions, infix_audit = apply_semantic_infix_guard(
                runtime_rows, fence_predictions,
                probability_ratio_floor,
            )
            semantic_audit = {
                "enabled": True,
                "equation": equation_audit,
                "fence": fence_audit,
                "infix": infix_audit,
            }
        else:
            equation_predictions = context_predictions
            fence_predictions = context_predictions
            predictions = context_predictions
            semantic_audit = {"enabled": False}
        output = []
        for row in runtime_rows:
            record_id = str(row["record_id"])
            prediction = str(predictions[record_id])
            if prediction not in row["final_topk"]:
                raise AssertionError("context finalizer invented a candidate")
            context_available, decision_status, decision_source = _decision_metadata(
                row,
                str(row["final_topk"][0]),
                str(context_predictions[record_id]),
                str(equation_predictions[record_id]),
                str(fence_predictions[record_id]),
                prediction,
                record_id in supported_exact_records,
            )
            output.append({
                "schema": "aiflow-formula-context-finalized/v4",
                "record_id": record_id,
                "formula_id": str(row["formula_id"]),
                "context_index": int(row["context"]["index"]),
                "context_available": context_available,
                "decision_status": decision_status,
                "decision_source": decision_source,
                "hwr_top1": str(row["final_topk"][0]),
                "finalized_top1": prediction,
                "changed": prediction != str(row["final_topk"][0]),
                "final_topk": list(row["final_topk"]),
                "final_topk_probabilities": list(row["final_topk_probabilities"]),
            })
        return output, {
            "context_checkpoint_sha256": self.checkpoint_sha256,
            "hwr_checkpoint_sha256": self.hwr_checkpoint_sha256,
            "pipeline": [
                "owned_formula_context_r6",
                "owned_supported_exact_context_guard_v1",
            ] + (
                [
                    "semantic_equation_guard_v2",
                    "semantic_fence_guard_v1",
                    "semantic_infix_guard_v1",
                ]
                if self.semantic_guards else []
            ),
            "records": len(output),
            "formulas": len({row["formula_id"] for row in output}),
            "changed": sum(bool(row["changed"]) for row in output),
            "decision_status": dict(Counter(row["decision_status"] for row in output)),
            "decision_source": dict(Counter(row["decision_source"] for row in output)),
            "candidate_preservation_rate": 1.0,
            "new_tokens": 0,
            "deleted_glyphs": 0,
            "grouping_mutations": 0,
            "model": model_audit,
            "semantic_guards": semantic_audit,
        }


def _write(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    labels = {"1", "+"}
    rows = [
        {
            "record_id": "a", "formula_id": "f", "label": "not-runtime-input",
            "final_topk": ["1", "+"], "final_topk_probabilities": [0.8, 0.2],
            "context": {"index": 0, "length": 1},
            "geometry": {"center_x": 0.0, "center_y": 0.0, "width_rel": 1.0, "height_rel": 1.0},
        }
    ]
    clean = _runtime_rows(rows, labels)
    assert "label" not in clean[0]
    assert clean[0]["final_topk"] == ["1", "+"]
    assert _decision_metadata(clean[0], "1", "+", "+", "+", "+") == (
        False, "ambiguous_no_formula_context", "context_prior_only",
    )
    contextual = {**clean[0], "context": {"index": 1, "length": 3}}
    assert _decision_metadata(contextual, "1", "1", "1", "1", "+") == (
        True, "finalized", "semantic_infix_guard",
    )
    assert _decision_metadata(
        contextual, "h", "b", "b", "b", "b", True
    ) == (True, "finalized", "owned_supported_exact_context_guard")
    try:
        _runtime_rows([{**rows[0], "final_topk": ["1", "x"]}], labels)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-vocabulary candidate was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--disable-semantic-guards", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    input_path = masked._d_path(args.input, "candidate input")
    output_path = masked._d_path(args.output, "finalized output")
    if output_path.exists():
        parser.error(f"refusing to overwrite finalized output: {output_path}")
    finalizer = OwnedFormulaContextFinalizer(
        args.checkpoint, args.hwr_checkpoint,
        device=args.device, batch_size=args.batch_size,
        semantic_guards=not args.disable_semantic_guards,
    )
    finalized, audit = finalizer.finalize(list(_json_lines(input_path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, finalized)
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path), "audit": audit,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
