#!/usr/bin/env python3
"""Evaluate a frozen owned formula-context checkpoint on candidate caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from character_tensor_v1 import _json_lines
from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _sha256
from evaluate_homograph_context_reranker_v1 import _metrics
import train_masked_context_reranker_v1 as masked
from train_owned_formula_context_v1 import (
    _assert_candidate_contract,
    decide_owned_formula_rows,
    load_owned_formula_context,
)


def _scope(rows: list[dict], predictions: dict[str, str]) -> dict:
    baseline = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
    context_metrics = _metrics(rows, predictions)
    baseline_metrics = _metrics(rows, baseline)
    audit = masked._candidate_audit(rows, predictions)
    _assert_candidate_contract("frozen checkpoint evaluation", audit)
    return {
        "baseline": baseline_metrics,
        "owned_context": context_metrics,
        "delta": masked._metric_delta(context_metrics, baseline_metrics),
        "candidate_audit": audit,
    }


def _subsets(rows: list[dict], predictions: dict[str, str]) -> dict:
    output = {"all": _scope(rows, predictions)}
    by_partition: dict[str, list[dict]] = defaultdict(list)
    by_writer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_partition[str(row.get("evaluation_partition", "unspecified"))].append(row)
        by_writer[str(row["writer_group"])].append(row)
    for name, subset in sorted(by_partition.items()):
        output[name] = _scope(subset, predictions)
    arrivals = [
        row for row in rows if str(row.get("evaluation_partition")) != "legacy47"
    ]
    if arrivals:
        output["new_arrivals"] = _scope(arrivals, predictions)
    output["by_writer"] = {
        writer: _scope(subset, predictions) for writer, subset in sorted(by_writer.items())
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    checkpoint = masked._d_path(args.checkpoint, "owned context checkpoint")
    hwr_checkpoint = masked._d_path(args.hwr_checkpoint, "HWR checkpoint")
    input_root = masked._d_path(args.input, "candidate input")
    output = masked._d_path(args.output, "evaluation report")
    if output.exists():
        parser.error(f"refusing to overwrite evaluation report: {output}")
    predictions_output = (
        masked._d_path(args.predictions_output, "prediction output")
        if args.predictions_output else None
    )
    if predictions_output is not None and predictions_output.exists():
        parser.error(f"refusing to overwrite predictions: {predictions_output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model, contract, payload = load_owned_formula_context(
        checkpoint, hwr_checkpoint, device
    )
    direct_path = input_root / "direct_candidates.jsonl.gz"
    direct_rows = list(_json_lines(direct_path))
    direct_predictions, direct_runtime = decide_owned_formula_rows(
        model, contract, payload, direct_rows, device, args.batch_size
    )
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        masked._write_prediction_rows(predictions_output, direct_rows, direct_predictions)
    report = {
        "schema": "aiflow-owned-formula-context-checkpoint-evaluation/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "device": str(device),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "hwr_checkpoint_sha256": _sha256(hwr_checkpoint),
        "direct_candidate_sha256": _sha256(direct_path),
        "direct_predictions": (
            {"path": str(predictions_output), "sha256": _sha256(predictions_output)}
            if predictions_output is not None else None
        ),
        "direct": {
            "runtime_audit": direct_runtime,
            "evaluation": _subsets(direct_rows, direct_predictions),
        },
    }
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    if crohme_path.is_file():
        crohme_rows = list(_json_lines(crohme_path))
        crohme_predictions, crohme_runtime = decide_owned_formula_rows(
            model, contract, payload, crohme_rows, device, args.batch_size
        )
        report["crohme"] = {
            "candidate_sha256": _sha256(crohme_path),
            "runtime_audit": crohme_runtime,
            "evaluation": _scope(crohme_rows, crohme_predictions),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    direct = report["direct"]["evaluation"]
    print(json.dumps({
        "report": str(output),
        "all": direct["all"]["owned_context"],
        "new_arrivals": direct.get("new_arrivals", {}).get("owned_context"),
        "candidate_preservation": direct["all"]["candidate_audit"]["candidate_preservation_rate"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
