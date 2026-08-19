#!/usr/bin/env python3
"""Research-only raw online-ink to candidate-preserving formula runtime.

The runtime selects one exact-cover stroke partition from a frozen Top-N
lattice, classifies each selected group with the frozen HWR head, and lets the
frozen masked-context finalizer choose only among supplied HWR candidates.  It
never inserts/deletes a glyph, evaluates arithmetic, or uses a target label or
target glyph count.  The current partition gate was tuned after inspecting its
chronological evaluation set, so callers must explicitly opt into shadow use.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from evaluate_48hz_prefix_v1 import DEFAULT_PRODUCT, _load_model, _sha256
from evaluate_joint_hwr_grouping_v1 import _candidate_embeddings, _probabilities
from evaluate_partition_context_ranker_v1 import (
    FEATURE_NAMES, POSTHOC_DESIGN_GUARD, SCHEMA as RANKER_SCHEMA,
    _attach_features, _gate_accept, _geometry_selection, _partition_rows,
    _select,
)
from finalize_formula_context_v1 import DEFAULT_CONTEXT, OwnedFormulaContextFinalizer
from stroke_grouping_v1 import build_lattice, candidate_features, enumerate_partitions
from train_project_owned_grouping_v1 import Sample


SCHEMA = "aiflow-raw-formula-context-runtime/v1"
OUTPUT_SCHEMA = "aiflow-raw-formula-context-result/v1"


def _device(name: str) -> torch.device:
    resolved = "cuda" if name == "auto" and torch.cuda.is_available() else (
        "cpu" if name == "auto" else name
    )
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(resolved)


def _validate_strokes(source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    formula_id = str(source.get("formula_id") or source.get("sample_id") or "").strip()
    if not formula_id:
        raise ValueError("raw formula requires formula_id or sample_id")
    strokes = list(source.get("strokes") or [])
    if not strokes:
        raise ValueError(f"raw formula has no strokes: {formula_id}")
    ordered = sorted(strokes, key=lambda row: int(row.get("order", -1)))
    orders = [int(row.get("order", -1)) for row in ordered]
    if orders != list(range(len(ordered))):
        raise ValueError(
            f"stroke orders must be unique contiguous zero-based indices: {formula_id}"
        )
    for stroke in ordered:
        points = list(stroke.get("points") or [])
        if not points:
            raise ValueError(f"empty stroke in formula: {formula_id}")
        previous_time = -math.inf
        for point in points:
            x = float(point["x"]); y = float(point["y"])
            time = float(point.get("t_ms", 0.0))
            if not all(math.isfinite(value) for value in (x, y, time)):
                raise ValueError(f"non-finite stroke point in formula: {formula_id}")
            if time < previous_time:
                raise ValueError(f"stroke timestamps must be monotonic: {formula_id}")
            previous_time = time
    return formula_id, ordered


def _load_inputs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("formulas", [payload])
    if not isinstance(payload, list) or not payload or any(not isinstance(row, dict) for row in payload):
        raise ValueError("raw input must be one formula, a formula list, or JSONL")
    return payload


@dataclass(frozen=True)
class RawFormulaContextRuntimeV1:
    ranker_payload: dict[str, Any]
    hwr: Any
    labels: list[str]
    finalizer: OwnedFormulaContextFinalizer
    device: torch.device
    partition_ranker_sha256: str

    @classmethod
    def from_artifacts(
        cls, partition_ranker: Path, hwr_checkpoint: Path, context_checkpoint: Path,
        *, device: str = "auto", batch_size: int = 128,
        formula_sequence_config: Path | None = None,
        formula_syntax_rescue_config: Path | None = None,
        allow_posthoc_shadow: bool = False,
    ) -> "RawFormulaContextRuntimeV1":
        ranker_path = Path(partition_ranker).expanduser().resolve()
        hwr_path = Path(hwr_checkpoint).expanduser().resolve()
        context_path = Path(context_checkpoint).expanduser().resolve()
        for path in (ranker_path, hwr_path, context_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = joblib.load(ranker_path)
        if (
            payload.get("schema") != RANKER_SCHEMA
            or tuple(payload.get("feature_names") or ()) != FEATURE_NAMES
            or "model" not in payload
            or "grouping_model" not in payload
        ):
            raise ValueError("partition-context ranker artifact contract mismatch")
        if payload.get("requires_explicit_shadow_opt_in") is not True:
            raise ValueError("partition-context artifact lacks explicit shadow contract")
        if not allow_posthoc_shadow:
            raise ValueError(
                "posthoc partition gate is development-only; explicit shadow opt-in is required"
            )
        if payload.get("product_default_enabled") is not False:
            raise ValueError("partition-context product gate must remain disabled")
        if payload.get("posthoc_test_tuning") is not True:
            raise ValueError("partition-context tuning provenance is missing")
        if payload.get("evaluation_only_writer_loo") is not False:
            raise ValueError("writer-LOO evaluation artifact is not a deployable runtime")
        if dict(payload.get("selection_guard") or {}) != POSTHOC_DESIGN_GUARD:
            raise ValueError("partition-context selection guard mismatch")
        if _sha256(hwr_path) != str(payload.get("hwr_checkpoint_sha256")):
            raise ValueError("partition-context HWR checkpoint hash mismatch")
        if _sha256(context_path) != str(payload.get("context_checkpoint_sha256")):
            raise ValueError("partition-context context checkpoint hash mismatch")
        finalizer_contract = dict(payload.get("finalizer_contract") or {})
        expected_finalizer = {
            "formula_layout": True,
            "semantic_guards": True,
            "equation_correction": False,
        }
        if any(finalizer_contract.get(key) != value for key, value in expected_finalizer.items()):
            raise ValueError("partition-context finalizer contract mismatch")
        if dict(finalizer_contract.get("auxiliary_candidates") or {}).get("enabled") is not False:
            raise ValueError("unimplemented auxiliary candidate contract is not runtime admissible")
        sequence_path = (
            Path(formula_sequence_config).expanduser().resolve()
            if formula_sequence_config is not None else None
        )
        syntax_path = (
            Path(formula_syntax_rescue_config).expanduser().resolve()
            if formula_syntax_rescue_config is not None else None
        )
        for name, path, hash_field in (
            ("formula sequence", sequence_path, "formula_sequence_config_sha256"),
            ("formula syntax rescue", syntax_path, "formula_syntax_rescue_config_sha256"),
        ):
            expected_hash = finalizer_contract.get(hash_field)
            if bool(path) != bool(expected_hash):
                raise ValueError(f"{name} config presence does not match ranker artifact")
            if path is not None and (not path.is_file() or _sha256(path) != expected_hash):
                raise ValueError(f"{name} config hash mismatch")
        resolved_device = _device(device)
        hwr, labels, _ = _load_model(hwr_path, resolved_device)
        finalizer = OwnedFormulaContextFinalizer(
            context_path, hwr_path, device=str(resolved_device), batch_size=batch_size,
            semantic_guards=True, equation_correction=False, formula_layout=True,
            formula_sequence_config=sequence_path,
            formula_syntax_rescue_config=syntax_path,
        )
        return cls(payload, hwr, labels, finalizer, resolved_device, _sha256(ranker_path))

    def _sample(self, source: dict[str, Any]) -> Sample:
        formula_id, strokes = _validate_strokes(source)
        lattice_config = dict(self.ranker_payload["lattice_config"])
        candidates = build_lattice(strokes, **lattice_config)
        return Sample(
            formula_id, "runtime", strokes, (), candidates,
            candidate_features(candidates, strokes),
        )

    def infer(self, source: dict[str, Any]) -> dict[str, Any]:
        sample = self._sample(source)
        embeddings, slices = _candidate_embeddings([sample], self.hwr, self.device)
        probabilities = _probabilities(self.hwr.math_head, embeddings, self.device)
        local_probability = probabilities[slices[sample.sample_id]]
        grouping_probability = self.ranker_payload["grouping_model"].predict_proba(
            sample.features
        )[:, 1]
        logits = np.log(
            np.clip(grouping_probability, 1e-6, 1 - 1e-6)
            / np.clip(1 - grouping_probability, 1e-6, 1)
        )
        top_n = int(self.ranker_payload["top_n"])
        ranked = enumerate_partitions(
            sample.candidates, logits, len(sample.strokes), top_n=top_n,
        )
        if not ranked:
            raise ValueError(f"no complete stroke partition: {sample.sample_id}")
        candidate_index = {
            frozenset(row["source_indices"]): index
            for index, row in enumerate(sample.candidates)
        }
        best_geometry = float(ranked[0][0])
        partitions = []
        for rank, (score, groups) in enumerate(ranked, 1):
            partitions.append({
                "sample": sample,
                "rank": rank,
                "geometry_score": float(score),
                "geometry_delta": float(score) - best_geometry,
                "groups": groups,
                "rows": _partition_rows(
                    sample, rank, groups, local_probability, self.labels,
                    candidate_index,
                ),
                "truth": False,
            })
        _attach_features(
            partitions, self.finalizer.model, self.finalizer.contract,
            self.finalizer.payload, self.device,
        )
        baseline = _geometry_selection(partitions)[sample.sample_id]
        proposal = _select(
            partitions, self.ranker_payload["model"], len(FEATURE_NAMES),
        )[sample.sample_id]
        accepted = _gate_accept(
            baseline, proposal, self.ranker_payload["selection_guard"],
        )
        selected = proposal if accepted else baseline
        runtime_rows = [
            {
                key: value for key, value in row.items()
                if key not in {"group", "full_probability", "grouping_features"}
            }
            for row in selected["rows"]
        ]
        finalized, finalizer_audit = self.finalizer.finalize(runtime_rows)
        finalized.sort(key=lambda row: int(row["context_index"]))
        runtime_by_record = {str(row["record_id"]): row for row in runtime_rows}
        source_by_group = {
            frozenset(int(value) for value in row["group"]): row
            for row in selected["rows"]
        }
        groups = []
        candidate_map = {
            frozenset(row["source_indices"]): row for row in sample.candidates
        }
        ordered_groups = sorted(
            selected["groups"],
            key=lambda group: (
                candidate_map[frozenset(group)]["box"]["left"],
                candidate_map[frozenset(group)]["box"]["top"], min(group),
            ),
        )
        for index, group in enumerate(ordered_groups):
            box = candidate_map[frozenset(group)]["box"]
            groups.append({
                "partition_input_index": index,
                "record_id": str(source_by_group[frozenset(group)]["record_id"]),
                "stroke_indices": sorted(int(value) for value in group),
                "box": {key: float(value) for key, value in box.items()},
            })
        assigned = sorted(
            index for group in groups for index in group["stroke_indices"]
        )
        if assigned != list(range(len(sample.strokes))):
            raise AssertionError("runtime grouping is not an exact stroke cover")
        hwr_tokens = [
            str(runtime_by_record[str(row["record_id"])]["final_topk"][0])
            for row in finalized
        ]
        finalized_tokens = [str(row["finalized_top1"]) for row in finalized]
        group_by_record = {row["record_id"]: row for row in groups}
        symbols = [
            {
                "record_id": str(row["record_id"]),
                "context_index": int(row["context_index"]),
                "relation_from_previous": row.get("layout_relation_from_previous"),
                "stroke_indices": list(group_by_record[str(row["record_id"])]["stroke_indices"]),
                "hwr_top1": str(runtime_by_record[str(row["record_id"])]["final_topk"][0]),
                "finalized_top1": str(row["finalized_top1"]),
                "changed": bool(row["changed"]),
            }
            for row in finalized
        ]
        return {
            "schema": OUTPUT_SCHEMA,
            "status": "development_only_posthoc_shadow",
            "formula_id": sample.sample_id,
            "groups": groups,
            "symbols": symbols,
            "hwr_top1_tokens": hwr_tokens,
            "partition_context_tokens": list(selected["tokens"]),
            "finalized_tokens": finalized_tokens,
            "formula_text": " ".join(finalized_tokens),
            "audit": {
                "candidate_groups": len(sample.candidates),
                "candidate_partitions": len(partitions),
                "geometry_baseline_rank": int(baseline["rank"]),
                "ranker_proposal_rank": int(proposal["rank"]),
                "selected_partition_rank": int(selected["rank"]),
                "partition_change_accepted": bool(accepted),
                "ranker_probability_gain": float(
                    proposal["ranker_probability"] - baseline["ranker_probability"]
                ),
                "all_strokes_exactly_once": True,
                "target_label_or_glyph_count_input": False,
                "inserted_or_deleted_glyphs": 0,
                "arithmetic_evaluation": False,
                "partition_ranker_sha256": self.partition_ranker_sha256,
                "hwr_checkpoint_sha256": self.finalizer.hwr_checkpoint_sha256,
                "context_checkpoint_sha256": self.finalizer.checkpoint_sha256,
                "product_default_enabled": False,
                "posthoc_test_tuning": True,
                "candidate_partitions_detail": [
                    {
                        "rank": int(partition["rank"]),
                        "geometry_delta": float(partition["geometry_delta"]),
                        "ranker_probability": float(partition["ranker_probability"]),
                        "groups": [sorted(int(value) for value in group) for group in partition["groups"]],
                        "hwr_top1_tokens": [str(row["final_topk"][0]) for row in partition["rows"]],
                        "context_tokens": list(partition["tokens"]),
                    }
                    for partition in partitions
                ],
                "finalizer": finalizer_audit,
            },
        }


def _self_test() -> None:
    source = {
        "formula_id": "self-test",
        "strokes": [{
            "order": 0,
            "points": [
                {"x": 0.0, "y": 0.0, "t_ms": 0.0},
                {"x": 1.0, "y": 1.0, "t_ms": 20.0},
            ],
        }],
    }
    formula_id, strokes = _validate_strokes(source)
    assert formula_id == "self-test" and len(strokes) == 1
    try:
        _validate_strokes({**source, "strokes": [{**source["strokes"][0], "order": 1}]})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid stroke order was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-ranker", type=Path)
    parser.add_argument("--hwr-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--context-checkpoint", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--formula-sequence-config", type=Path)
    parser.add_argument("--formula-syntax-rescue-config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-posthoc-shadow", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.partition_ranker is None or args.input is None or args.output is None:
        parser.error("--partition-ranker, --input, and --output are required")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"raw formula input is missing: {input_path}")
    if output_path.exists():
        parser.error(f"refusing to overwrite raw formula output: {output_path}")
    if not args.allow_posthoc_shadow:
        parser.error("--allow-posthoc-shadow is required for this development-only gate")
    runtime = RawFormulaContextRuntimeV1.from_artifacts(
        args.partition_ranker, args.hwr_checkpoint, args.context_checkpoint,
        device=args.device, batch_size=args.batch_size,
        formula_sequence_config=args.formula_sequence_config,
        formula_syntax_rescue_config=args.formula_syntax_rescue_config,
        allow_posthoc_shadow=args.allow_posthoc_shadow,
    )
    results = [runtime.infer(row) for row in _load_inputs(input_path)]
    output = {
        "schema": SCHEMA,
        "status": "development_only_posthoc_shadow",
        "formulas": results,
        "audit": {
            "formulas": len(results),
            "product_default_enabled": False,
            "posthoc_test_tuning": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "output": str(output_path), "sha256": _sha256(output_path),
        "formulas": len(results), "status": output["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
