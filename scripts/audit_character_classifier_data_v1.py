#!/usr/bin/env python3
"""Audit UCI duplicates, HWRT support, and BDSHWA character segmentation.

This script never trains a model and never mutates source archives. It writes
auditable reports and a before/after UCI plot for the character-classifier
admission decision.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from build_normalized_ink_v1 import DEFAULT_BDSHWA_ARCHIVE, _bdshwa_csv_archive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_ROOT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = ROOT / "research" / "character_classifier_data_audit_20260812"


def _json_lines(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def deduplicate_exact_consecutive_xy(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], int]:
    """Remove only adjacent exact XY duplicates; return-to-point loops remain."""
    output: list[tuple[float, float]] = []
    removed = 0
    for point in points:
        if output and point == output[-1]:
            removed += 1
        else:
            output.append(point)
    return output, removed


def _uci_raw_samples() -> list[tuple[int, str, list[tuple[float, float]], int]]:
    path = ROOT / "datasets" / "10_approved_external" / "uci_character_trajectories" / "raw" / "character+trajectories.official.zip"
    with zipfile.ZipFile(path) as archive:
        data = loadmat(io.BytesIO(archive.read("mixoutALL_shifted.mat")), squeeze_me=True, struct_as_record=False)
    labels = data["consts"].key
    class_ids = data["consts"].charlabels
    output = []
    for index, matrix in enumerate(data["mixout"]):
        points = [(float(matrix[0, point]), float(matrix[1, point])) for point in range(matrix.shape[1])]
        output.append((index, str(labels[int(class_ids[index]) - 1]), points, deduplicate_exact_consecutive_xy(points)[1]))
    return output


def _plot_uci(samples: list[tuple[int, str, list[tuple[float, float]], int]], output: Path) -> dict:
    selected = sorted((row for row in samples if row[3]), key=lambda row: (-row[3], row[0]))[:6]
    if not selected:
        raise AssertionError("UCI duplicate audit found no duplicate samples")
    figure, axes = plt.subplots(2, len(selected), figsize=(3.2 * len(selected), 6), constrained_layout=True)
    for column, (index, label, points, removed) in enumerate(selected):
        before = np.asarray(points, dtype=float)
        after = np.asarray(deduplicate_exact_consecutive_xy(points)[0], dtype=float)
        for axis, values, title in ((axes[0, column], before, "before"), (axes[1, column], after, "after")):
            axis.plot(values[:, 0], values[:, 1], color="#1f2937", linewidth=1.1)
            axis.scatter(values[:, 0], values[:, 1], s=6, color="#2563eb")
            axis.set_aspect("equal", adjustable="datalim")
            axis.invert_yaxis()
            axis.set_xticks([]); axis.set_yticks([])
            axis.set_title(f"{title}: #{index} '{label}'\nremoved={removed}", fontsize=9)
    figure.suptitle("UCI exact consecutive XY duplicates only", fontsize=13)
    path = output / "uci_exact_duplicate_before_after.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {"file": path.name, "selected_samples": [row[0] for row in selected]}


def _canonical_uci_dedup(canonical_root: Path) -> dict:
    removed = remaining = records = 0
    for row in _json_lines(canonical_root / "uci.jsonl.gz"):
        records += 1
        removed += int(row["normalization"]["removed_consecutive_exact_xy_points"])
        for stroke in row["strokes"]:
            points = stroke["points"]
            remaining += sum(left[:2] == right[:2] for left, right in zip(points, points[1:]))
    return {"records": records, "canonical_removed": removed, "remaining_consecutive_exact_xy": remaining}


def _quantile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1)]


def _support_band(count: int, mean: float) -> str:
    ratio = count / mean
    if ratio <= 0.25:
        return "at_or_below_0.25x_mean"
    if ratio <= 0.5:
        return "0.25x_to_0.5x_mean"
    if ratio <= 1.0:
        return "0.5x_to_1x_mean"
    if ratio <= 2.0:
        return "1x_to_2x_mean"
    if ratio <= 4.0:
        return "2x_to_4x_mean"
    return "above_4x_mean"


def _hwrt_support(canonical_root: Path, output: Path) -> dict:
    counts: Counter[str] = Counter()
    for row in _json_lines(canonical_root / "hwrt.jsonl.gz"):
        if row["point_count"] < 2:
            raise AssertionError("one-point HWRT row survived canonical filtering")
        counts[str(row["label"])] += 1
    values = list(counts.values())
    mean = statistics.fmean(values)
    bands = Counter(_support_band(value, mean) for value in values)
    with (output / "hwrt_class_support.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "count", "mean_ratio", "delta_from_mean", "band"))
        writer.writeheader()
        for label, count in sorted(counts.items(), key=lambda pair: (pair[1], pair[0])):
            writer.writerow({"label": label, "count": count, "mean_ratio": f"{count / mean:.6f}", "delta_from_mean": f"{count - mean:.6f}", "band": _support_band(count, mean)})
    return {
        "classes": len(counts),
        "samples": sum(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": max(values),
        "bands": dict(sorted(bands.items())),
        "under_0_25x_mean": [label for label, count in sorted(counts.items()) if count <= mean * 0.25],
        "over_4x_mean": [label for label, count in sorted(counts.items()) if count > mean * 4.0],
        "decision": {
            "rebalance": "class_balanced_sampler_plus_capped_loss_weight",
            "do_not_use": "predictive_rebalancer",
            "atomic_label_policy": "do_not_split_HWRT_labels_without_box_level_annotation",
            "reason": "frequency correction cannot create missing handwriting variation; compositional LaTex labels are source annotation units and automatic splitting would corrupt stroke ownership",
        },
    }


def _bdshwa_segmentability(archive_path: Path) -> dict:
    category_counts: Counter[str] = Counter()
    with _bdshwa_csv_archive(archive_path) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith("Raw_Data/") and name.endswith(".csv"))
        raw_header: list[str] = []
        for name in names:
            with archive.open(name) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
                raw_header = raw_header or list(reader.fieldnames or [])
                first_row = next(reader, None)
                if first_row is not None:
                    category_counts[str(first_row.get("category", ""))] += 1
    task_keys = {"category", "mode", "script", "task_index", "task_label", "task_prompt", "writing_condition"} & set(raw_header)
    boundary_fields = {"character", "char", "char_id", "character_index", "character_label", "segment", "segment_id", "char_start", "char_end"}
    return {
        "status": "rejected_for_character_classifier",
        "character_segmentable": False,
        "records": len(names),
        "category_counts": dict(sorted(category_counts.items())),
        "task_keys": sorted(task_keys),
        "raw_header": raw_header,
        "character_boundary_fields_present": sorted(boundary_fields & set(raw_header)),
        "reason": "Each file supplies only a whole-task prompt plus pen-down stroke IDs. It has no character boundary, character label, or character-to-stroke ownership field; freehand, shape, and wave tasks also have no text transcript suitable for character supervision.",
    }


def _write_markdown(output: Path, result: dict) -> None:
    support = result["hwrt_support"]
    lines = [
        "# Character classifier data audit",
        "",
        "## UCI duplicate decision",
        "",
        f"- Raw UCI samples: {result['uci_duplicates']['records']}",
        f"- Raw exact consecutive duplicate XY points: {result['uci_duplicates']['raw_points_removed']}",
        f"- Canonical duplicate removals: {result['uci_duplicates']['canonical_removed']} (including {result['uci_duplicates']['rounding_collisions_removed']} rounding collisions).",
        f"- Remaining canonical exact consecutive duplicates: {result['uci_duplicates']['remaining_consecutive_exact_xy']}",
        f"- Samples affected: {result['uci_duplicates']['affected_records']}",
        "- Decision: remove only adjacent exact XY duplicates; preserve return-to-point loops.",
        "",
        "## HWRT support distribution",
        "",
        f"- Usable samples: {support['samples']} across {support['classes']} classes.",
        f"- Mean {support['mean']:.2f}; median {support['median']:.1f}; p05 {support['p05']}; p95 {support['p95']}; range {support['min']}..{support['max']}.",
        f"- Distribution bands: `{json.dumps(support['bands'], sort_keys=True)}`.",
        "- Decision: use class-balanced sampling plus a capped loss weight when training begins. Do not use a predictive rebalancer and do not split source labels automatically.",
        "",
        "## BDSHWA decision",
        "",
        "- Rejected from the character-classifier corpus: whole-task prompts and stroke IDs do not supply character boundaries or ownership.",
        "- Raw archive remains untouched as audit evidence; it is not a tensor, training, or evaluation input.",
        "",
    ]
    (output / "CHARACTER_CLASSIFIER_DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def audit(canonical_root: Path, output: Path, bdshwa_archive: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    uci = _uci_raw_samples()
    duplicates = [row for row in uci if row[3]]
    canonical_uci = _canonical_uci_dedup(canonical_root)
    raw_removed = sum(row[3] for row in uci)
    if canonical_uci["canonical_removed"] < raw_removed or canonical_uci["remaining_consecutive_exact_xy"]:
        raise AssertionError("UCI canonical duplicate removal is incomplete")
    result = {
        "schema": "aiflow-character-classifier-data-audit/v1",
        "uci_duplicates": {
            "records": len(uci),
            "affected_records": len(duplicates),
            "raw_points_removed": raw_removed,
            "canonical_removed": canonical_uci["canonical_removed"],
            "rounding_collisions_removed": canonical_uci["canonical_removed"] - raw_removed,
            "remaining_consecutive_exact_xy": canonical_uci["remaining_consecutive_exact_xy"],
            "plot": _plot_uci(uci, output),
        },
        "hwrt_support": _hwrt_support(canonical_root, output),
        "bdshwa": _bdshwa_segmentability(bdshwa_archive),
    }
    (output / "character_classifier_data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _write_markdown(output, result)
    return result


def _self_test() -> None:
    points = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    result, removed = deduplicate_exact_consecutive_xy(points)
    assert result == [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)] and removed == 1
    assert _support_band(1, 10) == "at_or_below_0.25x_mean"
    assert _support_band(50, 10) == "above_4x_mean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bdshwa-archive", type=Path, default=DEFAULT_BDSHWA_ARCHIVE)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    print(json.dumps(audit(args.canonical_root, args.output, args.bdshwa_archive), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
