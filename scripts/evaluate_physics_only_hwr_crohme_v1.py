#!/usr/bin/env python3
"""동결된 물리 단독 HWR를 CROHME test에 한 번만 노출해 truth-group 평가한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from character_tensor_v1 import CHANNELS, POINTS, tensorize
from train_character_classifier_v1 import InkClassifierV1


SCHEMA = "aiflow-physics-only-hwr-crohme-evaluation/v1"
EXPECTED_CHECKPOINT_SCHEMA = "aiflow-physics-only-hwr-distillation/v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "physics_only_hwr_distillation_20260822_r1"
DEFAULT_CROHME = (
    Path(r"D:\AIFlow-Workspace\Projects\Aiflow\aiflow-math-ink-1.0")
    / "datasets" / "30_noncommercial_evaluation" / "crohme2019"
    / "crohme2019" / "crohme2019" / "test"
)
ALIASES = {r"\sqrt": r"\sqrt{}", r"\ldots": r"\dots", r"\lt": "<", r"\gt": ">"}
HOMOGRAPH_FAMILIES = {
    "vertical_slash": ("1", "|", "/"),
    "circle": ("0", "O", "o"),
    "cross": ("x", r"\times"),
}
CORE_OPERATORS = frozenset({
    "+", "-", "=", "/", "<", ">", "|", r"\times", r"\div",
    r"\pm", r"\neq", r"\leq", r"\geq",
})


def _event(name: str, **values: object) -> None:
    """한 번뿐인 평가 진행률을 UTF-8 JSON 한 줄로 출력한다."""

    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _sha256(path: Path) -> str:
    """파일의 SHA-256 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, label: str, *, must_exist: bool) -> Path:
    """평가 입력·출력이 D:에서만 움직이는지 확인한다."""

    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _local_name(node: ET.Element) -> str:
    """InkML XML namespace를 제거한 태그 이름을 반환한다."""

    return node.tag.rsplit("}", 1)[-1]


def _trace_points(text: str | None) -> list[tuple[float, float]]:
    """InkML trace 문자열에서 앞의 x·y 좌표 두 개만 안전하게 읽는다."""

    points: list[tuple[float, float]] = []
    for piece in (text or "").split(","):
        values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", piece)
        if len(values) >= 2:
            points.append((float(values[0]), float(values[1])))
    return points


def _parse_root(path: Path) -> tuple[ET.Element, str | None]:
    """UTF-8 InkML을 읽고 실패할 때만 기존 Latin-1 파일을 감사 표시와 함께 복구한다."""

    try:
        return ET.parse(path).getroot(), None
    except ET.ParseError as error:
        raw = path.read_bytes()
        try:
            return ET.fromstring(raw.decode("latin-1")), str(error)
        except (UnicodeDecodeError, ET.ParseError) as fallback:
            raise ET.ParseError(f"unrecoverable InkML {path}: {error}; {fallback}") from fallback


def _formula_fingerprint(truth: str, trace_ids: list[str], traces: dict[str, list[tuple[float, float]]]) -> str:
    """완전히 같은 수식 InkML 중복을 truth와 순서형 좌표로 식별한다."""

    payload = [truth, [[round(x, 6), round(y, 6)] for key in trace_ids for x, y in traces[key]]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _writer(root: ET.Element, path: Path) -> str:
    """writer annotation을 우선하고 없으면 CROHME 파일 세션 접두어를 사용한다."""

    values = [
        str(node.text).strip()
        for node in root.iter()
        if _local_name(node) == "annotation"
        and node.attrib.get("type") == "writer"
        and node.text and str(node.text).strip()
    ]
    if values:
        return values[0]
    match = re.match(r"^(\d+-\d+)-\d+$", path.stem)
    return f"inferred-session:{match.group(1)}" if match else f"missing-writer:{path.stem}"


def _sample(path: Path) -> tuple[dict | None, dict | None]:
    """한 CROHME 수식에서 원시 획과 truth stroke-group을 순서대로 복원한다."""

    root, fallback = _parse_root(path)
    trace_ids: list[str] = []
    traces: dict[str, list[tuple[float, float]]] = {}
    for node in root.iter():
        if _local_name(node) != "trace":
            continue
        points = _trace_points(node.text)
        if points:
            trace_id = node.attrib.get("id", str(len(trace_ids))).strip()
            trace_ids.append(trace_id)
            traces[trace_id] = points
    truths = [
        str(node.text or "").strip()
        for node in root.iter()
        if _local_name(node) == "annotation" and node.attrib.get("type") == "truth"
    ]
    if not trace_ids or not truths:
        return None, None
    index_by_id = {trace_id: index for index, trace_id in enumerate(trace_ids)}
    groups: list[list[int]] = []
    labels: list[str] = []
    for group in root.iter():
        if _local_name(group) != "traceGroup":
            continue
        annotation = next((
            child for child in group
            if _local_name(child) == "annotation"
            and child.attrib.get("type") == "truth" and child.text
        ), None)
        refs = [
            child.attrib.get("traceDataRef", "").strip()
            for child in group if _local_name(child) == "traceView"
        ]
        if annotation is None or not refs or any(value not in index_by_id for value in refs):
            continue
        raw = annotation.text.strip().replace("$", "")
        labels.append(ALIASES.get(raw, raw))
        groups.append([index_by_id[value] for value in refs])
    if not groups:
        return None, None
    sample = {
        "strokes": [traces[value] for value in trace_ids],
        "groups": groups,
        "labels": labels,
        "truth": truths[0],
        "writer": _writer(root, path),
        "partition_complete": set().union(*(set(value) for value in groups)) == set(range(len(trace_ids))),
        "fingerprint": _formula_fingerprint(truths[0], trace_ids, traces),
    }
    fallback_row = None if fallback is None else {
        "record_id": path.name,
        "original_parse_error": fallback,
        "fallback_encoding": "latin-1",
        "file_sha256": _sha256(path),
    }
    return sample, fallback_row


def _tensor(strokes: list[list[tuple[float, float]]]) -> np.ndarray:
    """한 truth-group의 원시 획을 letterbox·ordinal time 뒤 128x5로 만든다."""

    flat = np.asarray([point for stroke in strokes for point in stroke], dtype=np.float64)
    if len(flat) < 2:
        raise ValueError("single-point truth group is missing trajectory data")
    low, high = flat.min(axis=0), flat.max(axis=0)
    center = (low + high) * 0.5
    extent = max(float((high - low).max()), 1.0e-8)
    denominator = max(len(flat) - 1, 1)
    record_strokes = []
    ordinal = 0
    for source_order, stroke in enumerate(strokes):
        points = []
        for x, y in stroke:
            normalized = (np.asarray((x, y), dtype=np.float64) - center) / extent + 0.5
            points.append([
                float(np.clip(normalized[0], 0.0, 1.0)),
                float(np.clip(normalized[1], 0.0, 1.0)),
                ordinal / denominator,
            ])
            ordinal += 1
        record_strokes.append({"source_order": source_order, "points": points})
    features = tensorize({
        "source": "crohme2019_test_evaluation_only",
        "normalization": {"source_time_available": True},
        "strokes": record_strokes,
    })
    features[:, 2] = 1.0 / (POINTS - 1)
    features[0, 2] = 0.0
    features[:, 4] = 1.0
    return features


def _load_protocol(root: Path, labels: list[str]) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    """중복 제거 CROHME test에서 지원 가능한 truth-group과 전체 수식 계약을 만든다."""

    vocabulary = set(labels)
    seen: set[str] = set()
    features: list[np.ndarray] = []
    glyph_rows: list[dict] = []
    formula_rows: list[dict] = []
    duplicates: list[str] = []
    fallbacks: list[dict] = []
    unsupported = Counter()
    invalid = Counter()
    incomplete = 0
    for path in sorted(root.rglob("*.inkml")):
        sample, fallback = _sample(path)
        record_id = path.relative_to(root).as_posix()
        if sample is None:
            continue
        if sample["fingerprint"] in seen:
            duplicates.append(record_id)
            continue
        seen.add(sample["fingerprint"])
        if fallback is not None:
            fallbacks.append(dict(fallback, record_id=record_id))
        formula = {
            "formula_id": record_id,
            "writer_group": sample["writer"],
            "truth_tokens": list(sample["labels"]),
            "glyph_record_ids": [],
            "unsupported_labels": [],
            "invalid_labels": [],
            "partition_complete": bool(sample["partition_complete"]),
        }
        incomplete += int(not sample["partition_complete"])
        for index, (group, label) in enumerate(zip(sample["groups"], sample["labels"], strict=True)):
            if label not in vocabulary:
                unsupported[label] += 1
                formula["unsupported_labels"].append(label)
                continue
            strokes = [sample["strokes"][stroke_index] for stroke_index in group]
            if sum(len(stroke) for stroke in strokes) < 2:
                invalid[label] += 1
                formula["invalid_labels"].append(label)
                continue
            glyph_id = f"{record_id}:{index}"
            features.append(_tensor(strokes))
            glyph_rows.append({
                "record_id": glyph_id,
                "formula_id": record_id,
                "writer_group": sample["writer"],
                "group_index": index,
                "label": label,
            })
            formula["glyph_record_ids"].append(glyph_id)
        formula_rows.append(formula)
    if not features:
        raise ValueError("CROHME protocol produced no supported glyphs")
    coverage = {
        "inkml_files": len(list(root.rglob("*.inkml"))),
        "protocol_formulas": len(formula_rows),
        "exact_formula_duplicates_removed": len(duplicates),
        "duplicate_record_ids": duplicates,
        "supported_glyphs": len(glyph_rows),
        "supported_formulas": len({row["formula_id"] for row in glyph_rows}),
        "fully_supported_formulas": sum(
            row["partition_complete"]
            and not row["unsupported_labels"] and not row["invalid_labels"]
            and len(row["glyph_record_ids"]) == len(row["truth_tokens"])
            for row in formula_rows
        ),
        "unsupported_truth_groups": sum(unsupported.values()),
        "unsupported_labels": dict(unsupported.most_common()),
        "invalid_single_point_truth_groups": sum(invalid.values()),
        "invalid_labels": dict(invalid.most_common()),
        "incomplete_truth_partition_formulas": incomplete,
        "parse_fallback_files": len(fallbacks),
        "parse_fallbacks": fallbacks,
    }
    return np.stack(features).astype(np.float32), glyph_rows, formula_rows, coverage


def _load_frozen(artifact: Path, device: torch.device) -> tuple[InkClassifierV1, list[str], dict, Path, dict]:
    """freeze manifest의 해시·0행 계보를 확인한 뒤 HWR 추론 가중치만 연다."""

    freeze_path = artifact / "freeze_manifest.json"
    checkpoint_path = artifact / "physics_only_hwr_checkpoint.pt"
    receipt_path = artifact / "crohme_evaluation_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"one-time CROHME receipt already exists: {receipt_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "aiflow-physics-only-hwr-freeze/v1" or freeze.get("passed") is not True:
        raise ValueError("invalid physics-only freeze manifest")
    if freeze.get("checkpoint", {}).get("sha256") != _sha256(checkpoint_path):
        raise ValueError("frozen checkpoint hash mismatch")
    if any(int(freeze.get(key, -1)) != 0 for key in (
        "crohme_exposures_before_freeze", "crohme_gradient_updates", "post_freeze_gradient_updates",
    )):
        raise ValueError("freeze manifest reports pre-evaluation contamination")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = [str(value) for value in payload.get("math_labels", [])]
    policy = payload.get("report", {}).get("data_policy", {})
    if payload.get("schema") != EXPECTED_CHECKPOINT_SCHEMA or len(labels) != 372 or payload.get("auxiliary_labels"):
        raise ValueError("unexpected physics-only checkpoint contract")
    if policy.get("training_source") != "procedural_vector_topology_plus_48hz_pen_physics_only":
        raise ValueError("checkpoint is not physics-only")
    if policy.get("pretrained_weights_loaded") is not False:
        raise ValueError("checkpoint loaded pretrained weights")
    for key in (
        "vercel_rows", "private_data_rows", "project_owned_rows", "crohme_rows",
        "mathwriting_rows", "existing_trajectory_rows",
    ):
        if int(policy.get(key, -1)) != 0:
            raise ValueError(f"nonzero forbidden training provenance: {key}")
    contract = payload.get("report", {}).get("input_contract", {})
    if contract.get("observed_channel_mode") != "uniform-time":
        raise ValueError("checkpoint input contract is not uniform-time")
    model = InkClassifierV1(len(labels))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), labels, freeze, checkpoint_path, payload


@torch.inference_mode()
def _infer(
    model: InkClassifierV1, features: np.ndarray, labels: list[str],
    truths: list[dict], device: torch.device, batch_size: int,
) -> tuple[list[dict], dict]:
    """전체 372개 로짓으로 Top-5와 정답의 완전 순위를 한 번에 계산한다."""

    label_indices = {label: index for index, label in enumerate(labels)}
    top_rows: list[np.ndarray] = []
    rank_rows: list[np.ndarray] = []
    durations: list[float] = []
    if len(features):
        warmup = torch.from_numpy(features[: min(batch_size, len(features))]).to(device)
        model(warmup, "math")
        if device.type == "cuda":
            torch.cuda.synchronize()
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        logits = model(batch, "math")
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - began)
        top_rows.append(logits.topk(5, dim=1).indices.cpu().numpy())
        truth_indices = torch.tensor(
            [label_indices[row["label"]] for row in truths[start:start + len(batch)]],
            device=device,
        )
        truth_scores = logits.gather(1, truth_indices[:, None])
        ranks = 1 + torch.sum(logits > truth_scores, dim=1)
        rank_rows.append(ranks.cpu().numpy())
        _event("crohme_inference_progress", completed=min(start + batch_size, len(features)), total=len(features))
    top = np.concatenate(top_rows, axis=0)
    ranks = np.concatenate(rank_rows, axis=0)
    predictions = []
    for row, indices, rank in zip(truths, top, ranks, strict=True):
        predictions.append({
            **row,
            "top5": [labels[int(index)] for index in indices],
            "truth_rank": int(rank),
        })
    timing = {
        "batches": len(durations),
        "total_inference_seconds": sum(durations),
        "milliseconds_per_supported_glyph": 1000.0 * sum(durations) / len(features),
        "glyphs_per_second": len(features) / max(sum(durations), 1.0e-9),
    }
    return predictions, timing


def _category(label: str) -> str:
    """CROHME 라벨을 숫자·영문·핵심 연산자·기타 수학으로 나눈다."""

    if len(label) == 1 and label.isascii() and label.isdigit():
        return "digit"
    if len(label) == 1 and label.isascii() and label.isalpha():
        return "latin"
    return "core_operator" if label in CORE_OPERATORS else "other_math"


def _subset(rows: list[dict]) -> dict:
    """비어 있지 않은 문자 부분집합의 Top-1·Top-5·truth rank를 계산한다."""

    if not rows:
        return {"records": 0, "top1": None, "top5": None, "outside_top5": 0}
    top1 = [row["top5"][0] == row["label"] for row in rows]
    top5 = [row["label"] in row["top5"] for row in rows]
    ranks = np.asarray([row["truth_rank"] for row in rows], dtype=np.float64)
    outside = ranks[ranks > 5]
    return {
        "records": len(rows),
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "outside_top5": int(np.sum(~np.asarray(top5))),
        "mean_truth_rank": float(np.mean(ranks)),
        "median_truth_rank": float(np.median(ranks)),
        "outside_top5_mean_rank": float(np.mean(outside)) if len(outside) else None,
    }


def _formula_metrics(formulas: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    """truth grouping이 완전한 수식의 Top-1 exact와 Top-5 oracle을 계산한다."""

    by_id = {row["record_id"]: row for row in predictions}
    details = []
    for formula in formulas:
        complete = bool(
            formula["partition_complete"]
            and not formula["unsupported_labels"] and not formula["invalid_labels"]
            and len(formula["glyph_record_ids"]) == len(formula["truth_tokens"])
        )
        rows = [by_id[value] for value in formula["glyph_record_ids"] if value in by_id]
        top1_tokens = [row["top5"][0] for row in rows]
        exact = complete and top1_tokens == formula["truth_tokens"]
        oracle = complete and all(row["label"] in row["top5"] for row in rows)
        details.append({
            **formula,
            "fully_supported": complete,
            "top1_tokens": top1_tokens,
            "truth_group_top1_exact": exact,
            "truth_group_top5_oracle": oracle,
        })
    supported = [row for row in details if row["fully_supported"]]
    return {
        "protocol_formulas": len(details),
        "fully_supported_formulas": len(supported),
        "protocol_truth_group_top1_exact": sum(row["truth_group_top1_exact"] for row in details) / len(details),
        "protocol_truth_group_top5_oracle": sum(row["truth_group_top5_oracle"] for row in details) / len(details),
        "fully_supported_truth_group_top1_exact": sum(row["truth_group_top1_exact"] for row in supported) / len(supported),
        "fully_supported_truth_group_top5_oracle": sum(row["truth_group_top5_oracle"] for row in supported) / len(supported),
        "interpretation_limit": "truth stroke groups supplied; not raw grouping/layout/LgEval expression rate",
    }, details


def _metrics(predictions: list[dict], formulas: list[dict]) -> tuple[dict, list[dict]]:
    """문자·작가·클래스·동형군·수식 지표와 주요 혼동을 한 보고서로 모은다."""

    by_label: dict[str, list[dict]] = defaultdict(list)
    by_writer: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)
    confusion = Counter()
    outside = Counter()
    ranks = Counter()
    for row in predictions:
        by_label[row["label"]].append(row)
        by_writer[row["writer_group"]].append(row)
        by_category[_category(row["label"])].append(row)
        ranks[str(row["truth_rank"] if row["truth_rank"] <= 5 else "outside")] += 1
        if row["top5"][0] != row["label"]:
            confusion[(row["label"], row["top5"][0])] += 1
        if row["label"] not in row["top5"]:
            outside[row["label"]] += 1
    label_metrics = {label: _subset(rows) for label, rows in sorted(by_label.items())}
    writer_metrics = {writer: _subset(rows) for writer, rows in sorted(by_writer.items())}
    formula_metrics, formula_details = _formula_metrics(formulas, predictions)
    homographs = {}
    for name, family in HOMOGRAPH_FAMILIES.items():
        rows = [row for row in predictions if row["label"] in family]
        exact = _subset(rows)
        exact["labels"] = list(family)
        exact["family_top1"] = float(np.mean([row["top5"][0] in family for row in rows])) if rows else None
        exact["family_top5"] = float(np.mean([any(value in family for value in row["top5"]) for row in rows])) if rows else None
        homographs[name] = exact
    overall = _subset(predictions)
    overall.update({
        "strict_macro_top1": float(np.mean([row["top1"] for row in label_metrics.values()])),
        "strict_macro_top5": float(np.mean([row["top5"] for row in label_metrics.values()])),
        "writer_macro_top1": float(np.mean([row["top1"] for row in writer_metrics.values()])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in writer_metrics.values()])),
        "truth_rank_counts": dict(sorted(ranks.items())),
    })
    return {
        "character": overall,
        "formula": formula_metrics,
        "by_category": {name: _subset(rows) for name, rows in sorted(by_category.items())},
        "by_label": label_metrics,
        "by_writer": writer_metrics,
        "homograph_families": homographs,
        "top_confusions": [
            {"truth": pair[0], "prediction": pair[1], "records": count}
            for pair, count in confusion.most_common(40)
        ],
        "top_outside_labels": [
            {"label": label, "records": count} for label, count in outside.most_common(40)
        ],
    }, formula_details


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """평가 행을 결정적 mtime의 UTF-8 gzip JSONL로 저장한다."""

    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_markdown(path: Path, report: dict) -> None:
    """CROHME 단회 평가 수치와 상용화 경계를 간결한 한국어 표로 작성한다."""

    metrics = report["metrics"]
    character = metrics["character"]
    formula = metrics["formula"]
    lines = [
        "# 물리 지식 단독 HWR — CROHME test 단회 평가",
        "",
        "## 결론",
        "",
        f"- 문자 Top-1/Top-5: **{character['top1']:.2%} / {character['top5']:.2%}**",
        f"- Top-5 밖: **{character['outside_top5']} / {character['records']}건**; 밖 표본 평균 truth rank {character['outside_top5_mean_rank']:.2f}",
        f"- truth-group 식 exact/Top-5 oracle(완전 지원식): **{formula['fully_supported_truth_group_top1_exact']:.2%} / {formula['fully_supported_truth_group_top5_oracle']:.2%}**",
        "- 기존 HWR·Vercel·PrivateData 궤적과 사전학습 가중치는 0개였다. CROHME는 체크포인트 동결 뒤 이 평가 한 번에만 열었다.",
        "- 이 수치는 truth stroke-group이 주어진 **형태 분류 전이 평가**다. raw grouping·layout·공식 LgEval expression rate가 아니다.",
        "",
        "## 문자 지표",
        "",
        "| 지표 | 결과 |",
        "|---|---:|",
        f"| 지원 문자 | {character['records']} |",
        f"| Top-1 | {character['top1']:.2%} |",
        f"| Top-5 | {character['top5']:.2%} |",
        f"| strict macro Top-1 | {character['strict_macro_top1']:.2%} |",
        f"| strict macro Top-5 | {character['strict_macro_top5']:.2%} |",
        f"| 평균 truth rank | {character['mean_truth_rank']:.2f} |",
        f"| glyph당 추론 | {report['runtime']['milliseconds_per_supported_glyph']:.3f} ms |",
        "",
        "## 동형문자군",
        "",
        "| 군 | 표본 | exact Top-1 | exact Top-5 | family Top-1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in metrics["homograph_families"].items():
        lines.append(
            f"| `{name}` | {row['records']} | {row['top1']:.2%} | {row['top5']:.2%} | {row['family_top1']:.2%} |"
        )
    lines.extend([
        "",
        "## 범위와 채택 판단",
        "",
        f"- 체크포인트 SHA 평가 전·후: `{report['checkpoint']['sha256_before']}` / `{report['checkpoint']['sha256_after']}` (동일)",
        "- CROHME gradient update: **0회**; 평가 후 튜닝: **0회**.",
        "- 물리 엔진은 필기 동역학 불변성을 전달하지만 372개 문자 의미·사람의 획 토폴로지를 스스로 알지는 못한다. 낮은 전이 정확도라면 이 경로 단독 상용 채택은 불가하다.",
        "- 결과와 무관하게 동일 artifact의 CROHME 재평가·재학습은 evaluation receipt로 차단했다.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    """동결 감사, 단회 CROHME 추론, 영수증 기록을 순서대로 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--crohme-root", type=Path, default=DEFAULT_CROHME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    artifact = _d_path(args.artifact, "frozen artifact", must_exist=True)
    crohme_root = _d_path(args.crohme_root, "CROHME test root", must_exist=True)
    output = _d_path(args.output or artifact / "crohme_test_evaluation", "evaluation output", must_exist=False)
    if output.exists():
        parser.error("refusing to overwrite CROHME evaluation evidence")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise AssertionError("CROHME evaluation must disable autograd")
    model, labels, freeze, checkpoint_path, payload = _load_frozen(artifact, device)
    checkpoint_sha_before = _sha256(checkpoint_path)
    checkpoint_mtime_before = checkpoint_path.stat().st_mtime_ns
    output.mkdir(parents=True)
    started = time.perf_counter()
    features, glyph_rows, formula_rows, coverage = _load_protocol(crohme_root, labels)
    predictions, runtime = _infer(model, features, labels, glyph_rows, device, args.batch_size)
    metrics, formula_details = _metrics(predictions, formula_rows)
    checkpoint_sha_after = _sha256(checkpoint_path)
    checkpoint_mtime_after = checkpoint_path.stat().st_mtime_ns
    if checkpoint_sha_before != checkpoint_sha_after or checkpoint_mtime_before != checkpoint_mtime_after:
        raise AssertionError("frozen checkpoint changed during CROHME evaluation")
    prediction_path = output / "glyph_predictions.jsonl.gz"
    formula_path = output / "formula_truth_group_predictions.jsonl.gz"
    _write_gzip_jsonl(prediction_path, predictions)
    _write_gzip_jsonl(formula_path, formula_details)
    report = {
        "schema": SCHEMA,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "one_time_evaluation": True,
        "rights": {
            "dataset": "CROHME2019",
            "workspace_license": "CC BY-NC 4.0",
            "research_evaluation_only": True,
            "training_or_commercial_adoption_input": False,
        },
        "protocol": {
            "split": "test",
            "root": str(crohme_root),
            "truth_grouping_supplied": True,
            "streaming_prefix_scored": False,
            "official_lgeval_expression_rate": False,
        },
        "coverage": coverage,
        "metrics": metrics,
        "runtime": {
            **runtime,
            "device": str(device),
            "wall_seconds": time.perf_counter() - started,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "hwr_parameters": sum(value.numel() for value in model.parameters()),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": payload["schema"],
            "sha256_before": checkpoint_sha_before,
            "sha256_after": checkpoint_sha_after,
            "mtime_ns_before": checkpoint_mtime_before,
            "mtime_ns_after": checkpoint_mtime_after,
        },
        "freeze_manifest": freeze,
        "training_boundary": {
            "crohme_rows": 0,
            "crohme_gradient_updates": 0,
            "post_evaluation_gradient_updates": 0,
            "threshold_or_variant_selection_from_crohme": False,
            "vercel_calls": 0,
        },
        "outputs": {
            "glyph_predictions": {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
            "formula_predictions": {"path": str(formula_path), "sha256": _sha256(formula_path)},
        },
    }
    report_path = output / "crohme_evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown_path = output / "CROHME_EVALUATION.md"
    _write_markdown(markdown_path, report)
    receipt = {
        "schema": "aiflow-physics-only-hwr-crohme-receipt/v1",
        "evaluated_at": report["evaluated_at"],
        "checkpoint_sha256": checkpoint_sha_after,
        "crohme_split": "test",
        "crohme_supported_glyphs": len(predictions),
        "evaluation_report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "post_evaluation_gradient_updates": 0,
        "same_artifact_re_evaluation_forbidden": True,
        "passed": True,
    }
    receipt_path = artifact / "crohme_evaluation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _event(
        "physics_only_crohme_evaluation_complete",
        records=metrics["character"]["records"],
        top1=metrics["character"]["top1"],
        top5=metrics["character"]["top5"],
        outside_top5=metrics["character"]["outside_top5"],
        formula_exact=metrics["formula"]["fully_supported_truth_group_top1_exact"],
        report=str(report_path), receipt=str(receipt_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
