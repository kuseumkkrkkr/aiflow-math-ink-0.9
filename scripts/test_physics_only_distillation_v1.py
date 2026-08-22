#!/usr/bin/env python3
"""물리 단독 생성·동결·가짜 InkML 평가 경계를 실제 CROHME 없이 검사한다."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

import evaluate_physics_only_hwr_crohme_v1 as evaluation
import physics_only_glyph_teacher_v1 as teacher
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = ROOT / "research" / "physics_distillation" / "math_labels_372_v1.json"


def _fake_inkml(path: Path) -> None:
    """두 truth group을 가진 최소 InkML 수식을 UTF-8로 만든다."""

    value = """<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <annotation type="truth">1+</annotation>
  <annotation type="writer">synthetic-test-writer</annotation>
  <trace id="0">0 0, 0 10</trace>
  <trace id="1">10 5, 20 5</trace>
  <trace id="2">15 0, 15 10</trace>
  <traceGroup>
    <traceGroup><annotation type="truth">1</annotation><traceView traceDataRef="0"/></traceGroup>
    <traceGroup><annotation type="truth">+</annotation><traceView traceDataRef="1"/><traceView traceDataRef="2"/></traceGroup>
  </traceGroup>
</ink>
"""
    path.write_text(value, encoding="utf-8", newline="\n")


def _fake_artifact(path: Path, labels: list[str]) -> None:
    """무작위 HWR와 0행 계보를 가진 동결 모의 artifact를 만든다."""

    path.mkdir(parents=True)
    checkpoint_path = path / "physics_only_hwr_checkpoint.pt"
    model = InkClassifierV1(len(labels))
    torch.save({
        "schema": evaluation.EXPECTED_CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "math_labels": labels,
        "auxiliary_labels": [],
        "report": {
            "input_contract": {"observed_channel_mode": "uniform-time"},
            "data_policy": {
                "training_source": "procedural_vector_topology_plus_48hz_pen_physics_only",
                "pretrained_weights_loaded": False,
                "vercel_rows": 0,
                "private_data_rows": 0,
                "project_owned_rows": 0,
                "crohme_rows": 0,
                "mathwriting_rows": 0,
                "existing_trajectory_rows": 0,
            },
        },
    }, checkpoint_path)
    freeze = {
        "schema": "aiflow-physics-only-hwr-freeze/v1",
        "passed": True,
        "checkpoint": {"sha256": evaluation._sha256(checkpoint_path)},
        "crohme_exposures_before_freeze": 0,
        "crohme_gradient_updates": 0,
        "post_freeze_gradient_updates": 0,
    }
    (path / "freeze_manifest.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def main() -> int:
    """교사 결정성, 372 계약, 동결 로더, truth-group 추론을 연속 검사한다."""

    labels = json.loads(VOCABULARY.read_text(encoding="utf-8"))["labels"]
    if len(labels) != 372 or len(set(labels)) != 372:
        raise AssertionError("invalid 372-class contract")
    template = teacher.build_template(r"\cdot", "stix")
    profile = teacher.writer_physics(0, "development", 20260822)
    first, _ = teacher.synthesize(template, profile, 7)
    second, _ = teacher.synthesize(template, profile, 7)
    if not np.array_equal(first, second) or first.shape != (128, 5):
        raise AssertionError("procedural teacher replay contract failed")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="physics-distillation-test-", dir=ROOT / "artifacts") as temporary:
        temporary_root = Path(temporary)
        artifact = temporary_root / "artifact"
        protocol = temporary_root / "protocol"
        protocol.mkdir()
        _fake_artifact(artifact, labels)
        _fake_inkml(protocol / "fake.inkml")
        model, loaded_labels, _freeze, _checkpoint, _payload = evaluation._load_frozen(
            artifact, torch.device("cpu")
        )
        features, glyphs, formulas, coverage = evaluation._load_protocol(protocol, loaded_labels)
        predictions, timing = evaluation._infer(
            model, features, loaded_labels, glyphs, torch.device("cpu"), 2
        )
        metrics, details = evaluation._metrics(predictions, formulas)
        if features.shape != (2, 128, 5) or coverage["fully_supported_formulas"] != 1:
            raise AssertionError("fake truth-group protocol failed")
        if metrics["character"]["records"] != 2 or len(details) != 1:
            raise AssertionError("fake evaluation metrics failed")
        if timing["milliseconds_per_supported_glyph"] < 0.0:
            raise AssertionError("invalid inference timing")
    print(json.dumps({"self_test": "pass", "classes": len(labels), "fake_glyphs": 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
