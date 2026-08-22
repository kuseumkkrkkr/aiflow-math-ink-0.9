#!/usr/bin/env python3
"""절차형 펜 물리 교사만으로 무작위 초기화 372-class HWR를 증류한다."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from physics_only_glyph_teacher_v1 import (
    CHANNELS,
    FONTSETS,
    POINTS,
    SCHEMA as TEACHER_SCHEMA,
    build_templates,
    stable_seed,
    synthesize,
    template_manifest,
    trajectory_descriptor,
    writer_physics,
)
from train_character_classifier_v1 import InkClassifierV1


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "aiflow-physics-only-hwr-distillation/v1"
DEFAULT_VOCABULARY = ROOT / "research" / "physics_distillation" / "math_labels_372_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "physics_only_hwr_distillation_20260822_r1"
FORBIDDEN_TRAINING_MARKERS = (
    "vercel", "privatedata", "collector", "crohme", "mathwriting",
    "project_owned", "datasets/normalized", "30_noncommercial_evaluation",
)
LABEL_SMOOTHING = 0.02
CONFUSION_MASS = 0.24
SIMILARITY_TEMPERATURE = 0.16
DISTILLATION_WEIGHT = 0.18
PHYSICS_REGRESSION_WEIGHT = 0.05


class PhysicsCorpus(Dataset):
    """메모리에 고정된 절차형 128x5 표본과 물리 기술자를 제공한다."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, descriptors: np.ndarray) -> None:
        """입력·정답·교사 기술자를 dtype 고정 CPU 텐서로 보관한다."""

        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.descriptors = torch.from_numpy(np.asarray(descriptors, dtype=np.float32))

    def __len__(self) -> int:
        """절차형 표본 수를 반환한다."""

        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 표본의 입력, 정답 클래스, 표준화 물리 기술자를 반환한다."""

        return self.features[index], self.labels[index], self.descriptors[index]


class PhysicsDistilledStudent(nn.Module):
    """기존 128x5 HWR와 학습 때만 쓰는 물리 기술자 회귀 헤드를 묶는다."""

    def __init__(self, classes: int, descriptor_dimensions: int) -> None:
        """372-class HWR와 지정 차원의 학습 전용 물리 회귀 헤드를 초기화한다."""

        super().__init__()
        self.hwr = InkClassifierV1(classes)
        self.physics_head = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 192), nn.GELU(),
            nn.Linear(192, descriptor_dimensions),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """공유 HWR 임베딩에서 클래스 로짓과 물리 기술자를 함께 예측한다."""

        embedding = self.hwr.encode(features)
        return self.hwr.math_head(embedding), self.physics_head(embedding)


def _event(name: str, **values: object) -> None:
    """장시간 생성·학습 진행률을 UTF-8 JSON 한 줄로 출력한다."""

    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _sha256(path: Path) -> str:
    """파일의 SHA-256 지문을 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(paths: list[Path]) -> str:
    """정렬된 파일명·크기·내용으로 번들 폰트 집합 지문을 만든다."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _d_path(path: Path, label: str, *, must_exist: bool) -> Path:
    """모든 연구 입력·출력이 D: 안에 머무르는지 확인한다."""

    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{label} must remain on D: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _assert_no_forbidden_training_input(values: list[object]) -> None:
    """Vercel·실데이터·평가 데이터 표식이 학습 입력 경로에 있으면 닫힌 채 실패한다."""

    for value in values:
        normalized = str(value).replace("\\", "/").casefold()
        marker = next((item for item in FORBIDDEN_TRAINING_MARKERS if item in normalized), None)
        if marker is not None:
            raise ValueError(f"forbidden training marker {marker!r}: {value}")


def _load_vocabulary(path: Path) -> tuple[list[str], dict]:
    """궤적 없이 고정된 372개 출력 라벨 계약만 읽는다."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = [str(value) for value in payload.get("labels", [])]
    if payload.get("schema") != "aiflow-math-label-contract/v1":
        raise ValueError("unexpected vocabulary schema")
    if payload.get("classes") != 372 or len(labels) != 372 or len(set(labels)) != 372:
        raise ValueError("vocabulary must contain exactly 372 unique labels")
    forbidden_keys = {"features", "strokes", "probabilities", "state_dict", "writers"}
    if forbidden_keys.intersection(payload):
        raise ValueError("vocabulary contract contains non-label model/data content")
    return labels, payload


def _font_lineage() -> dict:
    """실제 사용한 Matplotlib 번들 폰트와 라이선스 파일의 지문을 기록한다."""

    data_root = Path(matplotlib.get_data_path()).resolve()
    font_root = data_root / "fonts" / "ttf"
    fonts = sorted(list(font_root.glob("STIX*.ttf")) + list(font_root.glob("DejaVu*.ttf")))
    licenses = [font_root / "LICENSE_STIX", font_root / "LICENSE_DEJAVU"]
    if not fonts or any(not path.is_file() for path in licenses):
        raise FileNotFoundError("bundled STIX/DejaVu font lineage is incomplete")
    return {
        "matplotlib_version": matplotlib.__version__,
        "fontsets": list(FONTSETS),
        "font_directory": str(font_root),
        "font_files": len(fonts),
        "font_bundle_sha256": _directory_sha256(fonts),
        "licenses": [
            {"path": str(path), "sha256": _sha256(path)} for path in licenses
        ],
        "commercial_rights_basis": [
            "STIX Font License / SIL Open Font License 1.1",
            "DejaVu bundled font license",
        ],
    }


def _split_arrays(rows: dict[str, list[np.ndarray]], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """누적 리스트를 한 분할의 연속 배열 네 개로 고정한다."""

    features = np.stack(rows[f"{split}_features"]).astype(np.float32, copy=False)
    labels = np.asarray(rows[f"{split}_labels"], dtype=np.int64)
    descriptors = np.stack(rows[f"{split}_descriptors"]).astype(np.float32, copy=False)
    writers = np.asarray(rows[f"{split}_writers"], dtype=np.int64)
    return features, labels, descriptors, writers


def _build_corpus(
    labels: list[str], train_writers: int, development_writers: int,
    stress_writers: int, seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict, list[dict]]:
    """벡터 토폴로지와 분할 독립 작가 물리만으로 전체 균형 코퍼스를 만든다."""

    templates = build_templates(labels)
    rows: dict[str, list] = defaultdict(list)
    writer_rows: list[dict] = []
    split_counts = {
        "train": train_writers,
        "development": development_writers,
        "stress": stress_writers,
    }
    for split, writer_count in split_counts.items():
        for writer_index in range(writer_count):
            profile = writer_physics(writer_index, split, seed)
            writer_rows.append(profile.__dict__)
            for label_index, label in enumerate(labels):
                font_index = stable_seed(SCHEMA, split, writer_index, label) % len(FONTSETS)
                template = templates[label][font_index]
                sample_seed = stable_seed(seed, split, writer_index, label_index)
                features, _metadata = synthesize(template, profile, sample_seed)
                rows[f"{split}_features"].append(features)
                rows[f"{split}_labels"].append(label_index)
                rows[f"{split}_descriptors"].append(trajectory_descriptor(features))
                rows[f"{split}_writers"].append(writer_index)
            _event("physics_writer_generated", split=split, completed=writer_index + 1, total=writer_count)
    corpus = {split: _split_arrays(rows, split) for split in split_counts}
    return corpus, template_manifest(templates), writer_rows


def _standardize_descriptors(corpus: dict[str, tuple[np.ndarray, ...]]) -> tuple[dict[str, tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
    """훈련 분할 통계만으로 물리 기술자를 표준화하고 극단값을 제한한다."""

    train = corpus["train"][2]
    mean = train.mean(axis=0)
    standard = train.std(axis=0)
    standard[standard < 1.0e-5] = 1.0
    adjusted: dict[str, tuple[np.ndarray, ...]] = {}
    for split, (features, labels, descriptors, writers) in corpus.items():
        normalized = np.clip((descriptors - mean) / standard, -6.0, 6.0).astype(np.float32)
        adjusted[split] = (features, labels, normalized, writers)
    return adjusted, mean.astype(np.float32), standard.astype(np.float32)


def _class_centers(descriptors: np.ndarray, labels: np.ndarray, classes: int) -> np.ndarray:
    """각 클래스의 절차형 물리 기술자 중심을 훈련 표본만으로 계산한다."""

    centers = np.stack([descriptors[labels == index].mean(axis=0) for index in range(classes)])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1.0e-8)
    return centers.astype(np.float32)


def _teacher_probabilities(
    descriptors: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor,
    similarity_temperature: float, confusion_mass: float,
) -> torch.Tensor:
    """생성 클래스와 물리 중심 유사도를 합쳐 후보 보존형 연성 교사 분포를 만든다."""

    normalized = F.normalize(descriptors, dim=1)
    physical = F.softmax(normalized @ centers.T / similarity_temperature, dim=1)
    hard = F.one_hot(labels, num_classes=centers.shape[0]).to(physical.dtype)
    return (1.0 - confusion_mass) * hard + confusion_mass * physical


@torch.inference_mode()
def _evaluate(
    model: PhysicsDistilledStudent, arrays: tuple[np.ndarray, ...],
    device: torch.device, batch_size: int,
) -> dict:
    """한 절차형 분할의 micro·macro·writer Top-1/Top-5와 물리 회귀 오차를 계산한다."""

    features, labels, descriptors, writers = arrays
    dataset = PhysicsCorpus(features, labels, descriptors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    regression_squared = 0.0
    regression_values = 0
    model.eval()
    for batch_features, _batch_labels, batch_descriptors in loader:
        logits, predicted_descriptors = model(batch_features.to(device, non_blocking=True))
        predictions.append(logits.topk(5, dim=1).indices.cpu().numpy())
        delta = predicted_descriptors.cpu() - batch_descriptors
        regression_squared += float(torch.sum(delta * delta))
        regression_values += delta.numel()
    top = np.concatenate(predictions, axis=0)
    top1 = top[:, 0] == labels
    top5 = np.any(top == labels[:, None], axis=1)
    per_class_top1 = [float(np.mean(top1[labels == index])) for index in range(int(labels.max()) + 1)]
    per_class_top5 = [float(np.mean(top5[labels == index])) for index in range(int(labels.max()) + 1)]
    per_writer = []
    for writer in sorted(set(int(value) for value in writers)):
        selected = writers == writer
        per_writer.append({
            "writer": writer,
            "top1": float(np.mean(top1[selected])),
            "top5": float(np.mean(top5[selected])),
        })
    return {
        "records": len(labels),
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "outside_top5": int(np.sum(~top5)),
        "macro_top1": float(np.mean(per_class_top1)),
        "macro_top5": float(np.mean(per_class_top5)),
        "writer_macro_top1": float(np.mean([row["top1"] for row in per_writer])),
        "writer_macro_top5": float(np.mean([row["top5"] for row in per_writer])),
        "worst_writer_top1": min(row["top1"] for row in per_writer),
        "worst_writer_top5": min(row["top5"] for row in per_writer),
        "physics_descriptor_mse": regression_squared / max(regression_values, 1),
    }


def _retrieval_metrics(descriptors: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> dict:
    """신경망과 무관한 물리 기술자 최근접 중심의 분류 가능성을 측정한다."""

    normalized = descriptors / np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1.0e-8)
    similarities = normalized @ centers.T
    top = np.argpartition(-similarities, kth=4, axis=1)[:, :5]
    top1 = np.argmax(similarities, axis=1) == labels
    top5 = np.any(top == labels[:, None], axis=1)
    return {"top1": float(np.mean(top1)), "top5": float(np.mean(top5))}


def _selection_score(development: dict, stress: dict) -> float:
    """CROHME 없이 unseen-writer와 강한 물리 분할을 함께 보는 선택 점수를 계산한다."""

    return (
        development["macro_top1"]
        + 0.25 * development["macro_top5"]
        + 0.50 * stress["macro_top1"]
        + 0.10 * stress["macro_top5"]
    )


def _train(
    corpus: dict[str, tuple[np.ndarray, ...]], centers_np: np.ndarray,
    classes: int, epochs: int, patience: int, batch_size: int,
    learning_rate: float, device: torch.device, seed: int,
) -> tuple[PhysicsDistilledStudent, list[dict], dict]:
    """물리 연성분포·기술자 회귀를 결합해 학생 HWR를 학습하고 합성 dev로만 선택한다."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = PhysicsDistilledStudent(classes, corpus["train"][2].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-2)
    training = PhysicsCorpus(corpus["train"][0], corpus["train"][1], corpus["train"][2])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    centers = torch.from_numpy(centers_np).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict] = []
    best_state: dict | None = None
    best_row: dict | None = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = defaultdict(float)
        records = 0
        started = time.perf_counter()
        for features, labels, descriptors in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            descriptors = descriptors.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, predicted_descriptors = model(features)
                hard_loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
                teacher = _teacher_probabilities(
                    descriptors, labels, centers,
                    SIMILARITY_TEMPERATURE, CONFUSION_MASS,
                )
                distillation_loss = F.kl_div(
                    F.log_softmax(logits, dim=1), teacher,
                    reduction="batchmean",
                )
                physics_loss = F.smooth_l1_loss(predicted_descriptors, descriptors)
                loss = (
                    hard_loss
                    + DISTILLATION_WEIGHT * distillation_loss
                    + PHYSICS_REGRESSION_WEIGHT * physics_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(labels)
            records += count
            totals["loss"] += float(loss.detach()) * count
            totals["hard_loss"] += float(hard_loss.detach()) * count
            totals["distillation_loss"] += float(distillation_loss.detach()) * count
            totals["physics_loss"] += float(physics_loss.detach()) * count
        development = _evaluate(model, corpus["development"], device, batch_size * 2)
        stress = _evaluate(model, corpus["stress"], device, batch_size * 2)
        score = _selection_score(development, stress)
        row = {
            "epoch": epoch,
            "train": {key: value / records for key, value in totals.items()},
            "development": development,
            "stress": stress,
            "selection_score": score,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        improved = best_row is None or score > float(best_row["selection_score"]) + 1.0e-6
        if improved:
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_row = deepcopy(row)
            stale = 0
        else:
            stale += 1
        _event(
            "physics_distillation_epoch",
            epoch=epoch,
            loss=row["train"]["loss"],
            development_top1=development["top1"],
            development_top5=development["top5"],
            stress_top1=stress["top1"],
            selection_score=score,
            best_epoch=int(best_row["epoch"]),
        )
        if epoch >= 6 and stale >= patience:
            break
    if best_state is None or best_row is None:
        raise AssertionError("student selection produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, history, best_row


def _collision_audit(centers: np.ndarray, labels: list[str], limit: int = 40) -> dict:
    """절차형 클래스 중심이 같은 형상으로 겹치는 상위 쌍을 찾는다."""

    similarities = centers @ centers.T
    pairs = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            pairs.append((float(similarities[left, right]), labels[left], labels[right]))
    pairs.sort(reverse=True)
    return {
        "near_identical_at_0_995": sum(value >= 0.995 for value, _left, _right in pairs),
        "top_pairs": [
            {"similarity": value, "left": left, "right": right}
            for value, left, right in pairs[:limit]
        ],
    }


def _save_preview(path: Path, features: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    """합성 development 표본 24개의 획 경계와 방향을 시각 검수용 PNG로 저장한다."""

    indices = np.linspace(0, len(features) - 1, 24, dtype=np.int64)
    figure, axes = plt.subplots(4, 6, figsize=(12, 8), dpi=150)
    for axis, index in zip(axes.flat, indices, strict=True):
        values = features[int(index)]
        starts = np.flatnonzero(values[:, 3] > 0.5)
        ends = np.r_[starts[1:], POINTS]
        for stroke_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            stroke = values[int(start):int(end), :2]
            axis.plot(stroke[:, 0], stroke[:, 1], linewidth=1.0)
            axis.scatter(stroke[0, 0], stroke[0, 1], s=7)
            if stroke_index == 0:
                axis.set_title(names[int(labels[int(index)])], fontsize=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(1.0, 0.0)
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle("Physics-only 48 Hz online ink: unseen procedural writers", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _write_markdown(path: Path, report: dict) -> None:
    """CROHME 노출 전 동결 상태와 합성 게이트를 짧은 한국어 보고서로 기록한다."""

    best = report["selection"]["best"]
    development = best["development"]
    stress = best["stress"]
    lines = [
        "# AIFlow Math Ink 1.0 물리 지식 단독 HWR 증류",
        "",
        "## 동결 결론",
        "",
        "- Vercel·PrivateData·직접수집·기존 궤적·기존 모델 가중치는 학습 입력에서 0행이다.",
        "- 372개 출력 이름만 계약으로 고정하고, STIX/DejaVu 벡터 기호를 스켈레톤화한 뒤 48 Hz 펜 운동계로 온라인 궤적을 절차 생성했다.",
        "- 학생 HWR는 무작위 초기화했다. hard label, 물리 중심의 연성 혼동분포, 76차원 운동 기술자 회귀를 함께 학습했다.",
        "- 이 체크포인트는 CROHME를 아직 보지 않은 **동결 연구 후보**다. CROHME 결과로 재학습하거나 임계값을 바꾸지 않는다.",
        "",
        "## 합성 선택 게이트",
        "",
        "| 분할 | 표본 | Top-1 | Top-5 | 최악 작가 Top-1 |",
        "|---|---:|---:|---:|---:|",
        f"| unseen development | {development['records']} | {development['top1']:.2%} | {development['top5']:.2%} | {development['worst_writer_top1']:.2%} |",
        f"| stronger physics stress | {stress['records']} | {stress['top1']:.2%} | {stress['top5']:.2%} | {stress['worst_writer_top1']:.2%} |",
        "",
        f"- 선택 epoch: **{best['epoch']}** / 실행 epoch {len(report['selection']['history'])}",
        f"- 체크포인트 SHA-256: `{report['checkpoint']['sha256']}`",
        f"- 템플릿 중심 0.995 이상 근접쌍: {report['class_collision_audit']['near_identical_at_0_995']}쌍",
        "",
        "## 데이터 계보",
        "",
        f"- 절차형 train/development/stress: {report['corpus']['counts']['train']}/{report['corpus']['counts']['development']}/{report['corpus']['counts']['stress']}행",
        f"- CROHME·MathWriting·Vercel·PrivateData·직접수집 학습 행: **0/0/0/0/0**",
        "- 폰트 라이선스 원문과 실제 번들 지문을 freeze manifest에 기록했다.",
        "- 래스터는 벡터 글리프를 중심선으로 변환하기 위한 일시적 내부 단계일 뿐 저장·입력되지 않는다. 학생 입력은 끝까지 128x5 온라인 잉크다.",
        "",
        "## 다음 한 번의 평가",
        "",
        "- `evaluate_physics_only_hwr_crohme_v1.py`가 freeze SHA를 확인한 뒤 CROHME test를 한 번만 연다.",
        "- 평가는 truth-group 문자 Top-1/Top-5, Top-5 밖, 식 exact/oracle, 동형문자군, 전체 truth rank, 지연시간을 기록한다.",
        "- 평가 영수증이 생긴 뒤에는 같은 artifact에서 재평가·재학습을 거부한다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    """절차형 코퍼스 생성, 학생 증류, 동결 해시 기록을 순서대로 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-writers", type=int, default=16)
    parser.add_argument("--development-writers", type=int, default=4)
    parser.add_argument("--stress-writers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(
        args.train_writers, args.development_writers, args.stress_writers,
        args.epochs, args.patience, args.batch_size,
    ) < 1 or args.learning_rate <= 0.0:
        parser.error("writer, epoch, patience, batch, and learning-rate values must be positive")
    vocabulary_path = _d_path(args.vocabulary, "vocabulary contract", must_exist=True)
    output = _d_path(args.output, "distillation output", must_exist=False)
    _assert_no_forbidden_training_input([vocabulary_path, output])
    if output.exists():
        parser.error("refusing to overwrite an existing distillation artifact")
    labels, vocabulary = _load_vocabulary(vocabulary_path)
    font_lineage = _font_lineage()
    _assert_no_forbidden_training_input([
        font_lineage["font_directory"],
        *[row["path"] for row in font_lineage["licenses"]],
    ])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output.mkdir(parents=True)
    started = time.perf_counter()
    corpus, templates, writers = _build_corpus(
        labels, args.train_writers, args.development_writers,
        args.stress_writers, args.seed,
    )
    corpus, descriptor_mean, descriptor_standard = _standardize_descriptors(corpus)
    centers = _class_centers(corpus["train"][2], corpus["train"][1], len(labels))
    corpus_path = output / "physics_teacher_corpus.npz"
    np.savez_compressed(
        corpus_path,
        train_features=corpus["train"][0], train_labels=corpus["train"][1],
        train_descriptors=corpus["train"][2], train_writers=corpus["train"][3],
        development_features=corpus["development"][0], development_labels=corpus["development"][1],
        development_descriptors=corpus["development"][2], development_writers=corpus["development"][3],
        stress_features=corpus["stress"][0], stress_labels=corpus["stress"][1],
        stress_descriptors=corpus["stress"][2], stress_writers=corpus["stress"][3],
        descriptor_mean=descriptor_mean, descriptor_standard=descriptor_standard,
        class_centers=centers,
    )
    preview_path = output / "physics_teacher_preview.png"
    _save_preview(preview_path, corpus["development"][0], corpus["development"][1], labels)
    retrieval = {
        split: _retrieval_metrics(values[2], values[1], centers)
        for split, values in corpus.items()
    }
    model, history, best = _train(
        corpus, centers, len(labels), args.epochs, args.patience,
        args.batch_size, args.learning_rate, device, args.seed,
    )
    final_development = _evaluate(model, corpus["development"], device, args.batch_size * 2)
    final_stress = _evaluate(model, corpus["stress"], device, args.batch_size * 2)
    if not math.isclose(final_development["top1"], best["development"]["top1"], abs_tol=1.0e-8):
        raise AssertionError("reloaded best checkpoint does not reproduce development Top-1")
    checkpoint_path = output / "physics_only_hwr_checkpoint.pt"
    checkpoint = {
        "schema": SCHEMA,
        "state_dict": model.hwr.state_dict(),
        "math_labels": labels,
        "auxiliary_labels": [],
        "report": {
            "input_contract": {
                "channels": list(CHANNELS),
                "observed_channel_mode": "uniform-time",
                "math_observed_transform": "uniform-time",
                "observed_channel_policy": "constant_one_for_source_invariance",
                "delta_t_policy": "normalized_sequence_progress:first_zero_then_1/(points-1)",
            },
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
            "architecture": {
                "input": [POINTS, len(CHANNELS)],
                "projection": [len(CHANNELS), 128],
                "transformer_blocks": 4,
                "hidden": 128,
                "attention_heads": 4,
                "math_head": len(labels),
                "training_only_physics_descriptor_head": int(centers.shape[1]),
            },
            "selection": {
                "source": "procedural_unseen_writer_and_stress_only",
                "best_epoch": int(best["epoch"]),
                "crohme_used": False,
            },
        },
        "distillation": {
            "teacher_schema": TEACHER_SCHEMA,
            "physics_head_state_dict": model.physics_head.state_dict(),
            "descriptor_mean": torch.from_numpy(descriptor_mean),
            "descriptor_standard": torch.from_numpy(descriptor_standard),
            "class_centers": torch.from_numpy(centers),
            "confusion_mass": CONFUSION_MASS,
            "similarity_temperature": SIMILARITY_TEMPERATURE,
            "hard_label_smoothing": LABEL_SMOOTHING,
            "distillation_weight": DISTILLATION_WEIGHT,
            "physics_regression_weight": PHYSICS_REGRESSION_WEIGHT,
            "random_initialization_seed": args.seed,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = _sha256(checkpoint_path)
    counts = {split: len(values[1]) for split, values in corpus.items()}
    collision = _collision_audit(centers, labels)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_crohme",
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": _sha256(vocabulary_path),
            "source_checkpoint_sha256_metadata_only": vocabulary["source_checkpoint_sha256"],
            "classes": len(labels),
        },
        "font_lineage": font_lineage,
        "templates": templates,
        "writers": writers,
        "corpus": {
            "path": str(corpus_path),
            "sha256": _sha256(corpus_path),
            "counts": counts,
            "balanced_rows_per_class": {
                "train": args.train_writers,
                "development": args.development_writers,
                "stress": args.stress_writers,
            },
            "input_shape": [POINTS, len(CHANNELS)],
            "admitted_sources": {
                "procedural_vector_topology": sum(counts.values()),
                "48hz_pen_motor_physics": sum(counts.values()),
            },
            "forbidden_rows": {
                "vercel": 0, "private_data": 0, "project_owned": 0,
                "crohme": 0, "mathwriting": 0, "existing_trajectory": 0,
            },
        },
        "teacher_retrieval": retrieval,
        "selection": {"best": best, "history": history},
        "final_replay": {"development": final_development, "stress": final_stress},
        "class_collision_audit": collision,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "bytes": checkpoint_path.stat().st_size,
            "pretrained_weights_loaded": False,
        },
        "preview": {"path": str(preview_path), "sha256": _sha256(preview_path)},
        "training_boundary": {
            "forbidden_markers": list(FORBIDDEN_TRAINING_MARKERS),
            "crohme_gradient_updates": 0,
            "mathwriting_gradient_updates": 0,
            "post_freeze_gradient_updates": 0,
            "vercel_calls": 0,
        },
    }
    report_path = output / "distillation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    freeze = {
        "schema": "aiflow-physics-only-hwr-freeze/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "distillation_report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
        "crohme_exposures_before_freeze": 0,
        "crohme_gradient_updates": 0,
        "post_freeze_gradient_updates": 0,
        "evaluation_receipt_must_not_exist": str(output / "crohme_evaluation_receipt.json"),
        "passed": True,
    }
    freeze_path = output / "freeze_manifest.json"
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown_path = output / "DISTILLATION_BEFORE_CROHME.md"
    report["checkpoint"]["freeze_manifest"] = str(freeze_path)
    _write_markdown(markdown_path, report)
    _event(
        "physics_only_hwr_frozen",
        checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
        best_epoch=int(best["epoch"]), development_top1=final_development["top1"],
        stress_top1=final_stress["top1"], output=str(output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
