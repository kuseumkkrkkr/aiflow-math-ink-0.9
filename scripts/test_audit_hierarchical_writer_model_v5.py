#!/usr/bin/env python3
"""계층형 작가모델 독립 감사기의 핵심 provenance 검증 테스트."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_hierarchical_writer_model_v5 import (
    _assignment_audit,
    _evaluation_boundary,
    _formula_audit,
    _source_audit,
)


class WriterModelAuditTest(unittest.TestCase):
    """합성 원천과 수식 partition을 작은 fixture로 검증한다."""

    def test_source_marks_synthetic_rows_without_raw_human_evidence(self) -> None:
        """synthetic_id가 있는 행을 인간 관측으로 세지 않는다."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            features = np.zeros((1, 128, 5), dtype=np.float32)
            features[:, :, 4] = 1.0
            np.savez_compressed(path, features=features, labels=np.asarray([7], dtype=np.int64))
            with gzip.open(path.with_suffix(".metadata.jsonl.gz"), "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"synthetic_id": "s1", "cross_writer": True, "parent_writer_fingerprints": ["w1", "w2"]}) + "\n")
            report, rows = _source_audit(path, None)
            self.assertEqual(report["synthetic_rows"], 1)
            self.assertEqual(report["raw_human_observed_rows"], 0)
            self.assertEqual(report["cross_writer_synthetic_rows"], 1)
            self.assertEqual(len(rows), 1)

    def test_formula_requires_exact_glyph_partition_and_monotonic_time(self) -> None:
        """정상 수식 행은 glyph/source/stroke/time 검증을 모두 통과한다."""

        features = np.zeros((1, 128, 5), dtype=np.float32)
        labels = np.asarray([3], dtype=np.int64)
        metadata = [{"synthetic_id": "s1", "cross_writer": False}]
        rows = [{
            "formula_id": "f1",
            "writer_id": "w1",
            "glyphs": [{"glyph_id": "g1", "token": "0"}],
            "production_order": ["g1"],
            "allographs": [{"glyph_id": "g1", "plan": {
                "source_domain": "approved_external_cleanroom", "source_row": 0,
                "label_index": 3, "stroke_count": 1,
            }}],
            "events": [{"glyph_id": "g1", "points": [
                {"x": 0.0, "y": 0.0, "t_ms": 1.0},
                {"x": 1.0, "y": 1.0, "t_ms": 2.0},
            ]}],
        }]
        report = _formula_audit(rows, {"approved_external_cleanroom": (features, labels, metadata)})
        self.assertEqual(report["provenance_failure_count"], 0)
        self.assertEqual(report["selected_synthetic_glyphs"], 1)
        self.assertEqual(report["selected_raw_human_observed_glyphs"], 0)

    def test_assignment_counts_real_source_variants(self) -> None:
        """작가가 다른 source row를 고른 토큰만 다양성으로 센다."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.json"
            path.write_text(json.dumps({
                "w1": {"A": {"source_domain": "d", "source_row": 1}},
                "w2": {"A": {"source_domain": "d", "source_row": 2}},
            }), encoding="utf-8")
            report = _assignment_audit(path)
            self.assertEqual(report["tokens_with_multiple_source_rows"], 1)

    def test_evaluation_only_parent_is_blocked(self) -> None:
        """평가 전용 원시 부모의 파생은행을 학습 증거로 승인하지 않는다."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "writer_group": "w1",
                    "training_role": "box_local_project_owned_evaluation_only",
                    "split": "project_owned_evaluation",
                }) + "\n")
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = _evaluation_boundary({"inputs": {
                "project_owned_rows": str(path),
                "project_owned_rows_sha256": digest,
            }}, project_bank_used=True)
            self.assertTrue(report["evaluation_only_parent_lineage"])
            self.assertTrue(report["evaluation_only_parent_lineage_used"])
            self.assertTrue(report["declared_sha256_matches"])
            isolated = _evaluation_boundary({"inputs": {
                "project_owned_rows": str(path),
                "project_owned_rows_sha256": digest,
            }}, project_bank_used=False)
            self.assertTrue(isolated["evaluation_only_parent_lineage"])
            self.assertFalse(isolated["evaluation_only_parent_lineage_used"])


if __name__ == "__main__":
    unittest.main()
