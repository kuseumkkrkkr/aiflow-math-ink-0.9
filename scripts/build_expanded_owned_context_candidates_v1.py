#!/usr/bin/env python3
"""Build direct formula-context candidates after append-only ownership expansion.

Legacy writers use their frozen held-writer HWR heads. Writers absent from the
legacy ownership set use the frozen product HWR checkpoint, which has never
seen those writers. The output remains a Top-5 candidate cache; no context
training is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from character_tensor_v1 import _digest, _json_lines, iter_direct_ownership_examples
from evaluate_48hz_prefix_v1 import (
    DEFAULT_BASE,
    DEFAULT_LOO,
    DEFAULT_PRODUCT,
    _load_loo_heads,
    _load_model,
    _sha256,
    resample_direct_48hz,
)
from evaluate_homograph_context_reranker_v1 import (
    _decorate,
    _d_path,
    _geometry,
    _score_items,
    _write_rows,
)


FORMULA_FINGERPRINT_FIELDS = (
    "target_display", "target_cells", "target_relations", "formula_cells",
    "canvas", "strokes", "stroke_count", "point_count",
)


def _formula_rows(root: Path) -> tuple[Path, Path, dict[str, dict], list[dict]]:
    formulas = root / "data" / "formulas_valid.jsonl"
    ownership = root / "data" / "ownership_train.jsonl"
    if not formulas.is_file() or not ownership.is_file():
        raise FileNotFoundError(f"dataset formulas/ownership missing: {root}")
    formula_rows = {str(row["sample_id"]): row for row in _json_lines(formulas)}
    annotations = [row for row in _json_lines(ownership) if row.get("accepted")]
    return formulas, ownership, formula_rows, annotations


def _fingerprint(row: dict) -> str:
    payload = {key: row.get(key) for key in FORMULA_FINGERPRINT_FIELDS}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_writer_map(candidate_root: Path, legacy_root: Path) -> tuple[dict[str, str], dict]:
    _, _, candidate_formulas, candidate = _formula_rows(candidate_root)
    _, _, legacy_formulas, legacy = _formula_rows(legacy_root)
    if len(candidate) < len(legacy) or not legacy:
        raise ValueError("expanded ownership must contain the complete non-empty legacy prefix")
    writer_map: dict[str, str] = {}
    for index, (candidate_row, legacy_row) in enumerate(
        zip(candidate[:len(legacy)], legacy, strict=True)
    ):
        if (
            candidate_row["labels"] != legacy_row["labels"]
            or candidate_row["groups"] != legacy_row["groups"]
            or _fingerprint(candidate_formulas[candidate_row["sample_id"]])
            != _fingerprint(legacy_formulas[legacy_row["sample_id"]])
        ):
            raise ValueError(f"legacy ownership prefix mismatch at row {index}")
        candidate_writer = str(candidate_row["writer_id"])
        legacy_writer = str(legacy_row["writer_id"])
        previous = writer_map.setdefault(candidate_writer, legacy_writer)
        if previous != legacy_writer:
            raise ValueError(f"candidate writer maps to multiple legacy writers: {candidate_writer}")
    if len(set(writer_map.values())) != len(writer_map):
        raise ValueError("legacy writer mapping is not one-to-one")
    return writer_map, {
        "legacy_formulas": len(legacy),
        "legacy_glyphs": sum(len(row["labels"]) for row in legacy),
        "legacy_writers": len(set(writer_map.values())),
    }


def _items(
    candidate_root: Path, legacy_writer_map: dict[str, str], legacy_formula_count: int,
) -> tuple[list[dict], dict[str, list[np.ndarray]], dict]:
    formulas_path, ownership_path, formulas, annotations = _formula_rows(candidate_root)
    glyphs = list(iter_direct_ownership_examples(formulas_path, ownership_path))
    items: list[dict] = []
    raw_strokes: dict[str, list[np.ndarray]] = {}
    policy = Counter()
    policy_formulas: dict[str, set[str]] = defaultdict(set)
    offset = 0
    partitions = Counter()
    for annotation_index, annotation in enumerate(annotations):
        sample_id = str(annotation["sample_id"])
        formula = formulas[sample_id]
        source_strokes = sorted(formula["strokes"], key=lambda row: int(row["order"]))
        formula_glyphs = glyphs[offset:offset + len(annotation["labels"])]
        if len(formula_glyphs) != len(annotation["groups"]):
            raise ValueError(f"ownership glyph count mismatch: {sample_id}")
        candidate_writer = str(annotation["writer_id"])
        legacy_writer = legacy_writer_map.get(candidate_writer)
        mode = "legacy_writer_loo" if legacy_writer is not None else "unseen_writer_product"
        partition = (
            "legacy47" if annotation_index < legacy_formula_count
            else ("new_known_writer" if legacy_writer is not None else "new_writer")
        )
        head_group = (
            _digest(["project_owned_writer", legacy_writer])[:24]
            if legacy_writer is not None else None
        )
        for glyph, group, label in zip(
            formula_glyphs, annotation["groups"], annotation["labels"], strict=True
        ):
            if str(glyph["label"]) != str(label):
                raise ValueError(f"ownership label order mismatch: {sample_id}")
            record_id = str(glyph["record_id"])
            items.append({
                "record_id": record_id,
                "label": str(label),
                "source": "project_owned",
                "formula_id": sample_id,
                "writer_group": str(glyph["writer_group"]),
                "strokes": resample_direct_48hz(glyph),
                "hwr_policy": mode,
                "hwr_head_group": head_group,
                "evaluation_partition": partition,
            })
            raw_strokes[record_id] = [
                np.asarray([
                    [float(point["x"]), float(point["y"]), float(point["t_ms"])]
                    for point in source_strokes[int(stroke_index)]["points"]
                ], dtype=np.float64)
                for stroke_index in sorted({int(value) for value in group})
            ]
            policy[mode] += 1
            policy_formulas[mode].add(sample_id)
            partitions[partition] += 1
        offset += len(annotation["labels"])
    if offset != len(glyphs) or set(raw_strokes) != {row["record_id"] for row in items}:
        raise AssertionError("expanded ownership item coverage mismatch")
    return items, raw_strokes, {
        "hwr_policy": {
            name: {"formulas": len(policy_formulas[name]), "glyphs": count}
            for name, count in sorted(policy.items())
        },
        "evaluation_partition_glyphs": dict(sorted(partitions.items())),
    }


def _score(
    items: list[dict], base_path: Path, loo_path: Path, product_path: Path,
    device: torch.device,
) -> tuple[dict[str, dict], dict, list[str]]:
    base, base_labels, _ = _load_model(base_path, device)
    heads = _load_loo_heads(loo_path, base_path, base_labels, device)
    product, product_labels, _ = _load_model(product_path, device)
    if product_labels != base_labels:
        raise ValueError("base/product HWR vocabularies differ")
    legacy = [row for row in items if row["hwr_policy"] == "legacy_writer_loo"]
    unseen = [row for row in items if row["hwr_policy"] == "unseen_writer_product"]
    legacy_for_scoring = [
        {**row, "writer_group": str(row["hwr_head_group"])} for row in legacy
    ]
    missing_heads = {
        str(row["hwr_head_group"]) for row in legacy
        if str(row["hwr_head_group"]) not in heads
    }
    if missing_heads:
        raise ValueError(f"legacy writer-LOO heads missing: {sorted(missing_heads)}")
    scores = {}
    if legacy_for_scoring:
        scores.update(_score_items(legacy_for_scoring, base, base_labels, device, heads))
    if unseen:
        scores.update(_score_items(unseen, product, product_labels, device))
    if set(scores) != {str(row["record_id"]) for row in items}:
        raise AssertionError("HWR score coverage mismatch")
    return scores, {
        "base_checkpoint_sha256": _sha256(base_path),
        "writer_loo_heads_sha256": _sha256(loo_path),
        "product_checkpoint_sha256": _sha256(product_path),
        "labels": len(base_labels),
    }, base_labels


def _admit_supported_formulas(
    items: list[dict], labels: list[str], raw_strokes: dict[str, list[np.ndarray]],
    scores: dict[str, dict],
) -> tuple[list[dict], dict[str, list[np.ndarray]], dict[str, dict], dict]:
    supported = set(labels)
    by_formula: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        by_formula[str(row["formula_id"])].append(row)
    excluded = {
        formula_id: sorted({str(row["label"]) for row in rows if str(row["label"]) not in supported})
        for formula_id, rows in by_formula.items()
        if any(str(row["label"]) not in supported for row in rows)
    }
    admitted = [row for row in items if str(row["formula_id"]) not in excluded]
    admitted_ids = {str(row["record_id"]) for row in admitted}
    admitted_policy = Counter(str(row["hwr_policy"]) for row in admitted)
    admitted_policy_formulas: dict[str, set[str]] = defaultdict(set)
    for row in admitted:
        admitted_policy_formulas[str(row["hwr_policy"])].add(str(row["formula_id"]))
    return (
        admitted,
        {key: value for key, value in raw_strokes.items() if key in admitted_ids},
        {key: value for key, value in scores.items() if key in admitted_ids},
        {
            "input_formulas": len(by_formula),
            "input_glyphs": len(items),
            "admitted_formulas": len({str(row["formula_id"]) for row in admitted}),
            "admitted_glyphs": len(admitted),
            "admitted_hwr_policy": {
                name: {
                    "formulas": len(admitted_policy_formulas[name]),
                    "glyphs": count,
                }
                for name, count in sorted(admitted_policy.items())
            },
            "admitted_evaluation_partition_glyphs": dict(sorted(Counter(
                str(row["evaluation_partition"]) for row in admitted
            ).items())),
            "excluded_formulas": excluded,
            "policy": "exclude the complete formula when any truth token is outside the frozen HWR vocabulary",
        },
    )


def _self_test() -> None:
    row = {key: key for key in FORMULA_FINGERPRINT_FIELDS}
    assert _fingerprint(row) == _fingerprint(dict(reversed(list(row.items()))))
    items = [
        {"record_id": "a", "formula_id": "kept", "label": "1", "hwr_policy": "legacy_writer_loo", "evaluation_partition": "legacy47"},
        {"record_id": "b", "formula_id": "dropped", "label": "1", "hwr_policy": "unseen_writer_product", "evaluation_partition": "new_writer"},
        {"record_id": "c", "formula_id": "dropped", "label": "t", "hwr_policy": "unseen_writer_product", "evaluation_partition": "new_writer"},
    ]
    admitted, _, _, audit = _admit_supported_formulas(
        items, ["1"], {key: [] for key in "abc"}, {key: {} for key in "abc"},
    )
    assert [item["record_id"] for item in admitted] == ["a"]
    assert audit["excluded_formulas"] == {"dropped": ["t"]}
    assert audit["admitted_hwr_policy"]["legacy_writer_loo"]["glyphs"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--legacy-dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "hf-dataset")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--loo-heads", type=Path, default=DEFAULT_LOO)
    parser.add_argument("--product-checkpoint", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print(json.dumps({"self_test": "pass"}))
        return 0
    if args.dataset_root is None or args.output is None:
        parser.error("--dataset-root and --output are required unless --self-test is used")
    candidate_root = _d_path(args.dataset_root, "expanded dataset")
    legacy_root = _d_path(args.legacy_dataset_root, "legacy dataset")
    output = _d_path(args.output, "candidate cache")
    report_path = output.with_suffix(output.suffix + ".report.json")
    if output.exists() or report_path.exists():
        parser.error(f"refusing to overwrite expanded candidate cache: {output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    writer_map, legacy_summary = _legacy_writer_map(candidate_root, legacy_root)
    items, raw_strokes, policy = _items(
        candidate_root, writer_map, int(legacy_summary["legacy_formulas"])
    )
    scores, checkpoints, hwr_labels = _score(
        items,
        _d_path(args.base_checkpoint, "base HWR checkpoint"),
        _d_path(args.loo_heads, "writer-LOO heads"),
        _d_path(args.product_checkpoint, "product HWR checkpoint"),
        device,
    )
    items, raw_strokes, scores, admission = _admit_supported_formulas(
        items, hwr_labels, raw_strokes, scores
    )
    geometry = _geometry(items, raw_strokes)
    seed_rows = [
        {
            key: row[key]
            for key in ("record_id", "label", "source", "formula_id", "writer_group")
        } | {"final_topk": scores[row["record_id"]]["final_topk"]}
        for row in items
    ]
    rows = _decorate(seed_rows, items, scores, geometry)
    item_by_id = {str(row["record_id"]): row for row in items}
    for row in rows:
        item = item_by_id[str(row["record_id"])]
        row["hwr_policy"] = item["hwr_policy"]
        row["evaluation_partition"] = item["evaluation_partition"]
    _write_rows(output, rows)
    report = {
        "schema": "aiflow-expanded-owned-context-candidates/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "dataset_root": str(candidate_root),
        "legacy_dataset_root": str(legacy_root),
        "formulas": len({row["formula_id"] for row in rows}),
        "glyphs": len(rows),
        "writers": len({row["writer_group"] for row in rows}),
        "admission": admission,
        "legacy_prefix": legacy_summary,
        **policy,
        "hwr_policy_contract": {
            "legacy_writer": "frozen head trained with that writer held out",
            "unseen_writer": "frozen product head trained before that writer arrived",
        },
        "checkpoints": checkpoints,
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": _sha256(output)},
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
