"""Fail-closed checks for the published AIFlow Math Ink dataset."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


FORBIDDEN = {"contributor_id", "session_id", "prompt_id", "recorded_at", "ink_png_data_url", "time_origin_ms", "raw_pressure", "source_path"}


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(root: Path) -> dict:
    manifest = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    formulas = []
    for status in ("valid", "pending", "reject"):
        path = root / "data" / f"formulas_{status}.jsonl"
        rows = _rows(path)
        assert len(rows) == manifest["quality_status"][status]
        assert all(row["quality_status"] == status for row in rows)
        formulas.extend(rows)
    assert len(formulas) == manifest["records"] == 119
    assert len({row["sample_id"] for row in formulas}) == len(formulas)
    assert not (FORBIDDEN & set(_walk(formulas)))
    assert all(row["stroke_count"] == len(row["strokes"]) for row in formulas)
    assert all(row["point_count"] == sum(len(stroke["points"]) for stroke in row["strokes"]) for row in formulas)
    assert all(min(float(point["t_ms"]) for stroke in row["strokes"] for point in stroke["points"]) == 0.0 for row in formulas if row["point_count"])
    ownership = _rows(root / "data" / "ownership_train.jsonl")
    by_id = {row["sample_id"]: row for row in formulas}
    assert len(ownership) == manifest["ownership_train_formulas"] == 47
    assert all(row["sample_id"] in by_id and len(row["groups"]) == len(row["labels"]) for row in ownership)
    for name, expected in manifest["files"].items():
        path = root / "data" / name
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    return {"records": len(formulas), "ownership": len(ownership), "forbidden_keys": 0, "status": "pass"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
