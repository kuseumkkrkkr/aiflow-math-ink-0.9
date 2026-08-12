#!/usr/bin/env python3
"""Run a controlled external-only Math Ink 1.0 classifier training trial.

This is intentionally not a production adoption command.  It trains only on
approved external character records outside the fixed 10% holdout, retains the
project-owned ownership set for evaluation, and does not train or score formula
decoding (CROHME/direct canonical formula replay).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from character_tensor_v1 import CHANNELS, POINTS, ROOT, _json_lines, tensorize
from replay_evaluate_hwr_v1 import score_glyphs_from_rows


DEFAULT_CANONICAL_ROOT = ROOT / "datasets" / "normalized" / "v1"
DEFAULT_OUTPUT = ROOT / "artifacts" / "character_classifier_v1_pilot_2ep"
CACHE_SCHEMA = "aiflow-character-classifier-cache/v2"
PILOT_SCHEMA = "aiflow-character-classifier-two-epoch-pilot/v2"
EXTERNAL_TRAINING_SCHEMA = "aiflow-character-classifier-external-training/v1"
SELECTION_TRAINING_SCHEMA = "aiflow-character-classifier-selection/v1"
SOURCE_HEAD = {"hwrt": "math", "uji": "auxiliary", "isgl": "auxiliary", "uci": "auxiliary"}
EXTRA_MATH_LABELS = ("(", ")", "=")
# UJI provides commercial isolated examples for these exact math tokens.  They
# train the deployed math head, not the auxiliary pretraining head, so each
# source record has one target head and one evaluation prediction.
MATH_TRANSFERRED_AUXILIARY_LABELS = frozenset({"(", ")"})
DEFAULT_EPOCHS = 2
WEIGHT_DECAY = 1e-2
BALANCE_MODES = ("sampler-and-loss", "sampler", "loss", "none")


def _checkpoint_contract(epochs: int, selection_enabled: bool = False) -> tuple[str, str]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if selection_enabled:
        return SELECTION_TRAINING_SCHEMA, "selection_checkpoint.pt"
    # Keep the committed two-epoch feasibility artifact reproducible while
    # giving longer controlled trials an explicit, non-misleading schema.
    return (PILOT_SCHEMA, "two_epoch_checkpoint.pt") if epochs == 2 else (EXTERNAL_TRAINING_SCHEMA, "classifier_checkpoint.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(path: Path) -> list[str]:
    return sorted({str(row["label"]) for row in _json_lines(path)})


def load_vocabs(canonical_root: Path) -> tuple[list[str], list[str]]:
    math_labels = _labels(canonical_root / "hwrt.jsonl.gz")
    all_uji_labels = _labels(canonical_root / "uji.jsonl.gz")
    aux_labels = sorted(set(all_uji_labels) - MATH_TRANSFERRED_AUXILIARY_LABELS)
    if len(math_labels) != 369 or any(label in math_labels for label in EXTRA_MATH_LABELS):
        raise ValueError("unexpected HWRT math vocabulary")
    if len(all_uji_labels) != 97 or len(aux_labels) != 95:
        raise ValueError("unexpected UJI auxiliary vocabulary")
    if not MATH_TRANSFERRED_AUXILIARY_LABELS <= set(all_uji_labels) or not MATH_TRANSFERRED_AUXILIARY_LABELS <= set(EXTRA_MATH_LABELS):
        raise ValueError("math transfer labels are not available in both vocabularies")
    for corpus in ("isgl", "uci"):
        missing = set(_labels(canonical_root / f"{corpus}.jsonl.gz")) - set(aux_labels)
        if missing:
            raise ValueError(f"{corpus} labels are outside UJI auxiliary vocabulary: {sorted(missing)}")
    return math_labels + list(EXTRA_MATH_LABELS), aux_labels


def _head_for_row(corpus: str, label: str) -> str:
    if corpus == "uji" and label in MATH_TRANSFERRED_AUXILIARY_LABELS:
        return "math"
    return SOURCE_HEAD[corpus]


class InkClassifierV1(nn.Module):
    """The fixed 128-point, 5-channel, two-head architecture decision."""

    def __init__(self, math_classes: int, auxiliary_classes: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(len(CHANNELS), 128), nn.LayerNorm(128), nn.GELU())
        self.position = nn.Parameter(torch.empty(1, POINTS, 128))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=4)
        self.pool_score = nn.Linear(128, 1)
        self.math_head = nn.Linear(128, math_classes)
        self.latin_aux_head = nn.Linear(128, auxiliary_classes)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[1:] != (POINTS, len(CHANNELS)):
            raise ValueError(f"expected batch x {POINTS} x {len(CHANNELS)} tensor, got {tuple(points.shape)}")
        hidden = self.encoder(self.input_projection(points) + self.position)
        weights = F.softmax(self.pool_score(hidden).squeeze(-1), dim=1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1)

    def forward(self, points: torch.Tensor, head: str) -> torch.Tensor:
        embedding = self.encode(points)
        if head == "math":
            return self.math_head(embedding)
        if head == "auxiliary":
            return self.latin_aux_head(embedding)
        raise ValueError(f"unknown head: {head}")


class NpyDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, feature_path: Path, label_path: Path) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if len(self.features) != len(self.labels) or self.features.shape[1:] != (POINTS, len(CHANNELS)):
            raise ValueError(f"invalid cache arrays: {feature_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True)), int(self.labels[index])


class ArrayDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features, self.labels = features, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[index].copy()), int(self.labels[index])


class IndexedDataset(Dataset[tuple[torch.Tensor, int]]):
    """A stable subset of a memory-mapped cache without copying its features."""

    def __init__(self, source: NpyDataset, indices: np.ndarray) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(source.labels[self.indices], dtype=np.int64)
        if not len(self.indices) or len(self.indices) != len(self.labels):
            raise ValueError("invalid indexed dataset")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.source[int(self.indices[index])]


def _holdout_ids(canonical_root: Path) -> set[str]:
    path = canonical_root / "character_classifier_v1" / "current_external_holdout_ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed external holdout IDs: {path}; run character_tensor_v1.py --build-manifests")
    return set(json.loads(path.read_text(encoding="utf-8")))


def _iter_external_rows(canonical_root: Path, holdout_ids: set[str], label_indices: dict[str, dict[str, int]]) -> Iterator[tuple[str, dict, dict]]:
    for corpus in SOURCE_HEAD:
        for row in _json_lines(canonical_root / f"{corpus}.jsonl.gz"):
            record_id = f"{corpus}:{row['record_id']}"
            label = str(row["label"])
            head = _head_for_row(corpus, label)
            if label not in label_indices[head]:
                raise ValueError(f"unmapped label: {corpus}:{label}")
            split = "eval" if record_id in holdout_ids else "train"
            yield f"{head}_{split}", row, {
                "record_id": record_id,
                "source": corpus,
                "head": head,
                "label": label,
                "label_index": label_indices[head][label],
            }


def _cache_config(canonical_root: Path, holdout_path: Path, math_labels: list[str], aux_labels: list[str]) -> dict:
    return {
        "schema": CACHE_SCHEMA,
        "canonical_manifest_sha256": _sha256(canonical_root / "manifest.json"),
        "holdout_ids_sha256": _sha256(holdout_path),
        "points": POINTS,
        "channels": list(CHANNELS),
        "math_labels": math_labels,
        "auxiliary_labels": aux_labels,
        "math_transferred_auxiliary_labels": sorted(MATH_TRANSFERRED_AUXILIARY_LABELS),
    }


def prepare_cache(canonical_root: Path, cache_dir: Path, math_labels: list[str], aux_labels: list[str]) -> dict:
    holdout_path = canonical_root / "character_classifier_v1" / "current_external_holdout_ids.json"
    holdout_ids = _holdout_ids(canonical_root)
    config = _cache_config(canonical_root, holdout_path, math_labels, aux_labels)
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = [cache_dir / item["features"] for item in manifest.get("sets", {}).values()]
        expected += [cache_dir / item["labels"] for item in manifest.get("sets", {}).values()]
        if manifest.get("config") == config and all(path.is_file() for path in expected):
            return manifest
        raise RuntimeError(f"stale or partial cache at {cache_dir}; choose a fresh --output path")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.iterdir()):
        raise RuntimeError(f"cache directory is not empty: {cache_dir}")
    label_indices = {"math": {label: index for index, label in enumerate(math_labels)}, "auxiliary": {label: index for index, label in enumerate(aux_labels)}}
    names = ("math_train", "math_eval", "auxiliary_train", "auxiliary_eval")
    counts = {name: 0 for name in names}
    seen_eval: set[str] = set()
    for name, _, metadata in _iter_external_rows(canonical_root, holdout_ids, label_indices):
        counts[name] += 1
        if name.endswith("_eval"):
            seen_eval.add(metadata["record_id"])
    if seen_eval != holdout_ids:
        raise AssertionError("external holdout does not match cache scan")
    features = {
        name: np.lib.format.open_memmap(cache_dir / f"{name}_features.npy", mode="w+", dtype=np.float32, shape=(count, POINTS, len(CHANNELS)))
        for name, count in counts.items()
    }
    labels = {
        name: np.lib.format.open_memmap(cache_dir / f"{name}_labels.npy", mode="w+", dtype=np.int64, shape=(count,))
        for name, count in counts.items()
    }
    truth_streams = {
        name: (cache_dir / f"{name}_truth.jsonl").open("w", encoding="utf-8", newline="\n")
        for name in names if name.endswith("_eval")
    }
    offsets = {name: 0 for name in names}
    support = {name: Counter() for name in names}
    sources = {name: Counter() for name in names}
    try:
        for name, row, metadata in _iter_external_rows(canonical_root, holdout_ids, label_indices):
            offset = offsets[name]
            features[name][offset] = tensorize(row)
            labels[name][offset] = metadata["label_index"]
            support[name][metadata["label"]] += 1
            sources[name][metadata["source"]] += 1
            if name in truth_streams:
                truth_streams[name].write(json.dumps({key: value for key, value in metadata.items() if key != "label_index"}, ensure_ascii=False, separators=(",", ":")) + "\n")
            offsets[name] += 1
    finally:
        for stream in truth_streams.values():
            stream.close()
        for array in [*features.values(), *labels.values()]:
            array.flush()
    if offsets != counts:
        raise AssertionError("cache write count mismatch")
    manifest = {
        "config": config,
        "sets": {
            name: {
                "records": counts[name],
                "features": f"{name}_features.npy",
                "labels": f"{name}_labels.npy",
                "truth": f"{name}_truth.jsonl" if name.endswith("_eval") else None,
                "sources": dict(sorted(sources[name].items())),
                "labels_present": len(support[name]),
            }
            for name in names
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def _dataset(cache_dir: Path, item: dict) -> NpyDataset:
    return NpyDataset(cache_dir / item["features"], cache_dir / item["labels"])


def _selection_indices_from_groups(groups: dict[str, list[tuple[int, str]]], expected_records: int, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Select deterministic indices while retaining at least one row in each group."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("selection validation ratio must be in (0, 1)")
    selected: list[int] = []
    for key, rows in sorted(groups.items()):
        count = min(len(rows) - 1, max(1, math.floor(len(rows) * ratio)))
        if count:
            ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{key}:{row[1]}".encode("utf-8")).digest())
            selected.extend(index for index, _ in ranked[:count])
    selection_indices = np.asarray(sorted(selected), dtype=np.int64)
    selection_mask = np.zeros(expected_records, dtype=bool)
    selection_mask[selection_indices] = True
    training_indices = np.flatnonzero(~selection_mask).astype(np.int64)
    if not len(training_indices) or not len(selection_indices):
        raise AssertionError("invalid selection split")
    return training_indices, selection_indices


def _selection_split_indices(
    canonical_root: Path,
    holdout_ids: set[str],
    label_indices: dict[str, dict[str, int]],
    split_name: str,
    expected_records: int,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a deterministic source-label slice while retaining every group in training."""
    groups: dict[str, list[tuple[int, str]]] = {}
    offsets: Counter[str] = Counter()
    for name, _, metadata in _iter_external_rows(canonical_root, holdout_ids, label_indices):
        index = offsets[name]
        offsets[name] += 1
        if name == split_name:
            key = f"{metadata['source']}:{metadata['label']}"
            groups.setdefault(key, []).append((index, str(metadata["record_id"])))
    if offsets[split_name] != expected_records:
        raise AssertionError(f"{split_name} cache order does not match canonical rows")
    return _selection_indices_from_groups(groups, expected_records, ratio, seed)


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _balance_flags(mode: str) -> tuple[bool, bool]:
    if mode not in BALANCE_MODES:
        raise ValueError(f"unknown balance mode: {mode}")
    return mode in {"sampler-and-loss", "sampler"}, mode in {"sampler-and-loss", "loss"}


def _class_balanced_loader(dataset: Dataset, labels: np.ndarray, batch_size: int, seed: int, classes: int, pin_memory: bool, balance_mode: str) -> tuple[DataLoader, torch.Tensor]:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if not len(labels) or not np.all(counts[labels] > 0):
        raise ValueError("invalid training labels")
    use_sampler, use_loss_weights = _balance_flags(balance_mode)
    observed = counts > 0
    loss_weights = np.ones(classes, dtype=np.float32)
    if use_loss_weights:
        loss_weights[observed] = np.clip(counts[observed].mean() / counts[observed], 0.25, 4.0)
    generator = torch.Generator().manual_seed(seed)
    if use_sampler:
        sampler_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(labels), replacement=True, generator=generator)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=pin_memory)
    return loader, torch.from_numpy(loss_weights)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_head(model: InkClassifierV1, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loader: DataLoader, loss_weights: torch.Tensor, head: str, device: torch.device, epoch: int, log_every: int) -> dict:
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    amp = device.type == "cuda"
    for batch_index, (features, labels) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=amp)
        labels = labels.to(device, non_blocking=amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(features, head), labels, weight=loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}, {head}, batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if batch_index % log_every == 0:
            print(json.dumps({"event": "train_progress", "epoch": epoch, "head": head, "batch": batch_index, "batches": len(loader), "mean_loss": sum(losses) / len(losses)}, ensure_ascii=False), flush=True)
    return {"head": head, "batches": len(loader), "samples": len(loader.dataset), "loss": sum(losses) / len(losses), "seconds": time.perf_counter() - started}


@torch.inference_mode()
def predict(model: InkClassifierV1, dataset: Dataset, truths: list[dict], labels: list[str], head: str, device: torch.device, batch_size: int) -> dict[str, dict]:
    if len(dataset) != len(truths):
        raise ValueError("prediction dataset/truth count mismatch")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model.eval()
    output: dict[str, dict] = {}
    offset = 0
    for features, _ in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda"), head)
        topk = logits.topk(k=5, dim=1).indices.detach().cpu().tolist()
        for truth, indices in zip(truths[offset:offset + len(topk)], topk, strict=True):
            output[str(truth["record_id"])] = {"topk": [labels[index] for index in indices]}
        offset += len(topk)
    if offset != len(truths):
        raise AssertionError("prediction count mismatch")
    return output


def _read_truth(path: Path) -> list[dict]:
    return list(_json_lines(path))


def _selection_truth(dataset: IndexedDataset, labels: list[str], head: str) -> list[dict]:
    return [
        {"record_id": f"selection:{head}:{int(index)}", "label": labels[int(label)]}
        for index, label in zip(dataset.indices, dataset.labels, strict=True)
    ]


def _selection_score(
    model: InkClassifierV1,
    math_dataset: IndexedDataset,
    math_truth: list[dict],
    math_labels: list[str],
    auxiliary_dataset: IndexedDataset,
    auxiliary_truth: list[dict],
    auxiliary_labels: list[str],
    device: torch.device,
    batch_size: int,
) -> dict:
    math_predictions = predict(model, math_dataset, math_truth, math_labels, "math", device, batch_size)
    auxiliary_predictions = predict(model, auxiliary_dataset, auxiliary_truth, auxiliary_labels, "auxiliary", device, batch_size)
    return {
        "all": score_glyphs_from_rows(math_truth + auxiliary_truth, {**math_predictions, **auxiliary_predictions}),
        "math": score_glyphs_from_rows(math_truth, math_predictions),
        "auxiliary": score_glyphs_from_rows(auxiliary_truth, auxiliary_predictions),
    }


def direct_ownership_dataset(canonical_root: Path, math_index: dict[str, int]) -> tuple[ArrayDataset, list[dict]]:
    path = canonical_root / "character_classifier_v1" / "project_owned_ownership_eval.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing ownership evaluation derivative: {path}")
    features: list[np.ndarray] = []
    labels: list[int] = []
    truths: list[dict] = []
    for row in _json_lines(path):
        label = str(row["label"])
        if label not in math_index:
            raise ValueError(f"direct ownership label outside math vocabulary: {label}")
        features.append(tensorize(row))
        labels.append(math_index[label])
        truths.append({"record_id": str(row["record_id"]), "label": label})
    return ArrayDataset(np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)), truths


def _write_predictions(path: Path, predictions: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record_id in sorted(predictions):
            stream.write(json.dumps({"record_id": record_id, **predictions[record_id]}, ensure_ascii=False, separators=(",", ":")) + "\n")


def _self_test() -> None:
    model = InkClassifierV1(372, 95)
    points = torch.zeros(2, POINTS, len(CHANNELS))
    assert model(points, "math").shape == (2, 372)
    assert model(points, "auxiliary").shape == (2, 95)
    assert _balance_flags("sampler-and-loss") == (True, True)
    assert _balance_flags("sampler") == (True, False)
    assert _balance_flags("loss") == (False, True)
    assert _balance_flags("none") == (False, False)
    assert _head_for_row("uji", "(") == "math"
    assert _head_for_row("uji", "A") == "auxiliary"
    assert _checkpoint_contract(2) == (PILOT_SCHEMA, "two_epoch_checkpoint.pt")
    assert _checkpoint_contract(3) == (EXTERNAL_TRAINING_SCHEMA, "classifier_checkpoint.pt")
    assert _checkpoint_contract(5, True) == (SELECTION_TRAINING_SCHEMA, "selection_checkpoint.pt")
    train_indices, selection_indices = _selection_indices_from_groups({"a": [(0, "a0"), (1, "a1")], "b": [(2, "b0"), (3, "b1"), (4, "b2")]}, 5, 0.5, 7)
    assert len(train_indices) == 3 and len(selection_indices) == 2
    assert len(set(train_indices).intersection({0, 1})) == 1 and len(set(train_indices).intersection({2, 3, 4})) == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--balance-mode", choices=BALANCE_MODES, default="sampler")
    parser.add_argument("--cache-dir", type=Path, help="reuse a compatible D: cache across controlled trials")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--selection-validation-ratio", type=float, default=0.0, help="hold out a source-label stratified external-training slice; this mode never scores the fixed holdout")
    parser.add_argument("--selection-validation-seed", type=int, default=20260813)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.log_every < 1:
        parser.error("epochs, batch sizes, and log interval must be positive")
    if not 0.0 <= args.selection_validation_ratio < 1.0:
        parser.error("--selection-validation-ratio must be in [0, 1)")
    selection_enabled = args.selection_validation_ratio > 0.0
    output = args.output.resolve()
    if output.drive.upper() != "D:":
        parser.error(f"output must remain on D:: {output}")
    cache_dir = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    if cache_dir.drive.upper() != "D:":
        parser.error(f"cache must remain on D:: {cache_dir}")
    run_schema, checkpoint_name = _checkpoint_contract(args.epochs, selection_enabled)
    checkpoint_path = output / checkpoint_name
    if any((output / name).exists() for name in ("two_epoch_checkpoint.pt", "classifier_checkpoint.pt", "selection_checkpoint.pt", "training_report.json")):
        parser.error(f"classifier checkpoint already exists: {output}; choose a fresh --output path")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    math_labels, aux_labels = load_vocabs(args.canonical_root)
    cache = prepare_cache(args.canonical_root, cache_dir, math_labels, aux_labels)
    math_train_full = _dataset(cache_dir, cache["sets"]["math_train"])
    aux_train_full = _dataset(cache_dir, cache["sets"]["auxiliary_train"])
    selection_report: dict = {"enabled": False}
    if selection_enabled:
        label_indices = {"math": {label: index for index, label in enumerate(math_labels)}, "auxiliary": {label: index for index, label in enumerate(aux_labels)}}
        holdout_ids = _holdout_ids(args.canonical_root)
        math_train_indices, math_selection_indices = _selection_split_indices(args.canonical_root, holdout_ids, label_indices, "math_train", len(math_train_full), args.selection_validation_ratio, args.selection_validation_seed)
        aux_train_indices, aux_selection_indices = _selection_split_indices(args.canonical_root, holdout_ids, label_indices, "auxiliary_train", len(aux_train_full), args.selection_validation_ratio, args.selection_validation_seed)
        math_train, math_selection = IndexedDataset(math_train_full, math_train_indices), IndexedDataset(math_train_full, math_selection_indices)
        aux_train, aux_selection = IndexedDataset(aux_train_full, aux_train_indices), IndexedDataset(aux_train_full, aux_selection_indices)
        math_selection_truth = _selection_truth(math_selection, math_labels, "math")
        aux_selection_truth = _selection_truth(aux_selection, aux_labels, "auxiliary")
        selection_report = {
            "enabled": True,
            "ratio": args.selection_validation_ratio,
            "seed": args.selection_validation_seed,
            "stratification": "source_label_within_external_training_after_fixed_holdout",
            "selection_metric": "all_glyph_top1_then_top5_earliest_epoch",
            "fixed_external_holdout_used_for_selection": False,
            "math": {"training_records": len(math_train), "selection_records": len(math_selection), "selection_indices_sha256": _indices_sha256(math_selection.indices)},
            "auxiliary": {"training_records": len(aux_train), "selection_records": len(aux_selection), "selection_indices_sha256": _indices_sha256(aux_selection.indices)},
        }
    else:
        math_train, aux_train = math_train_full, aux_train_full
    math_loader, math_weights = _class_balanced_loader(math_train, math_train.labels, args.batch_size, args.seed, len(math_labels), device.type == "cuda", args.balance_mode)
    aux_loader, aux_weights = _class_balanced_loader(aux_train, aux_train.labels, args.batch_size, args.seed + 1, len(aux_labels), device.type == "cuda", args.balance_mode)
    model = InkClassifierV1(len(math_labels), len(aux_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    math_weights = math_weights.to(device)
    aux_weights = aux_weights.to(device)
    print(json.dumps({"event": "training_start", "device": str(device), "epochs": args.epochs, "learning_rate": args.learning_rate, "balance_mode": args.balance_mode, "math_train": len(math_train), "auxiliary_train": len(aux_train), "selection_validation": selection_report["enabled"], "fixed_external_holdout_scored": False}, ensure_ascii=False), flush=True)
    history = []
    best_selection: dict | None = None
    best_selection_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_report = {"epoch": epoch, "math": train_head(model, optimizer, scaler, math_loader, math_weights, "math", device, epoch, args.log_every), "auxiliary": train_head(model, optimizer, scaler, aux_loader, aux_weights, "auxiliary", device, epoch, args.log_every)}
        if selection_enabled:
            selection_score = _selection_score(model, math_selection, math_selection_truth, math_labels, aux_selection, aux_selection_truth, aux_labels, device, args.eval_batch_size)
            epoch_report["selection_validation"] = selection_score
            current = selection_score["all"]
            if best_selection is None or (current["top1"], current["top5"]) > (best_selection["top1"], best_selection["top5"]):
                best_selection = {"epoch": epoch, "top1": current["top1"], "top5": current["top5"]}
                best_selection_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(json.dumps({"event": "selection_validation", "epoch": epoch, "top1": current["top1"], "top5": current["top5"], "selected_epoch": best_selection["epoch"]}, ensure_ascii=False), flush=True)
        history.append(epoch_report)
    if selection_enabled:
        if best_selection is None or best_selection_state is None:
            raise AssertionError("selection validation did not produce a checkpoint")
        model.load_state_dict(best_selection_state)
        selection_report["selected_epoch"] = best_selection["epoch"]
        selection_report["selected_top1"] = best_selection["top1"]
        selection_report["selected_top5"] = best_selection["top5"]
        external_score = {"status": "not_scored", "reason": "fixed external holdout is reserved for the full-data retrain after selection"}
        direct_score = {"status": "not_scored", "reason": "project ownership is reserved for the full-data retrain after selection"}
    else:
        math_index = {label: index for index, label in enumerate(math_labels)}
        math_eval = _dataset(cache_dir, cache["sets"]["math_eval"])
        aux_eval = _dataset(cache_dir, cache["sets"]["auxiliary_eval"])
        math_truth = _read_truth(cache_dir / cache["sets"]["math_eval"]["truth"])
        aux_truth = _read_truth(cache_dir / cache["sets"]["auxiliary_eval"]["truth"])
        direct_eval, direct_truth = direct_ownership_dataset(args.canonical_root, math_index)
        external_predictions = {}
        external_predictions.update(predict(model, math_eval, math_truth, math_labels, "math", device, args.eval_batch_size))
        external_predictions.update(predict(model, aux_eval, aux_truth, aux_labels, "auxiliary", device, args.eval_batch_size))
        direct_predictions = predict(model, direct_eval, direct_truth, math_labels, "math", device, args.eval_batch_size)
        external_score = score_glyphs_from_rows(math_truth + aux_truth, external_predictions)
        direct_score = score_glyphs_from_rows(direct_truth, direct_predictions)
        _write_predictions(output / "external_holdout_predictions.jsonl", external_predictions)
        _write_predictions(output / "direct_ownership_predictions.jsonl", direct_predictions)
    observed_math = {math_labels[index] for index in np.unique(np.asarray(math_train.labels, dtype=np.int64))}
    report = {
        "schema": run_schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "trained_epochs": args.epochs,
        "device": str(device),
        "seed": args.seed,
        "optimization": {"optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": WEIGHT_DECAY, "balance_mode": args.balance_mode, "gradient_clip_norm": 1.0, "scheduler": "none", "warmup_steps": 0},
        "architecture": {"input": [POINTS, len(CHANNELS)], "projection": [len(CHANNELS), 128], "transformer_blocks": 4, "hidden": 128, "attention_heads": 4, "math_head": len(math_labels), "auxiliary_head": len(aux_labels)},
        "data_admission": {"train_sources": SOURCE_HEAD, "math_transferred_auxiliary_labels": sorted(MATH_TRANSFERRED_AUXILIARY_LABELS), "project_owned_training": False, "bdshwa_training": False, "crohme_training": False, "holdout_fixed_before_training": True},
        "cache": cache,
        "selection_validation": selection_report,
        "history": history,
        "external_holdout": external_score,
        "direct_ownership": {**direct_score, **({"records": len(direct_truth), "formulas": 47, "untrained_math_output_labels": sorted(set(math_labels) - observed_math)} if not selection_enabled else {})},
        "formula_protocols": {
            "crohme2019": {"status": "not_scored", "records": 985, "reason": "this checkpoint is a box-local character classifier; no grouping, relation, or formula decoder is implemented"},
            "direct_canonical_10": {"status": "not_scored", "records": 10, "reason": "formula-exact scoring requires the absent grouping and decision pipeline"},
        },
        "elapsed_seconds": time.perf_counter() - started,
        "product_adopted": False,
    }
    torch.save({"schema": run_schema, "state_dict": model.state_dict(), "math_labels": math_labels, "auxiliary_labels": aux_labels, "report": report}, checkpoint_path)
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"event": "training_complete", "selection_enabled": selection_enabled, "selected_epoch": selection_report.get("selected_epoch", args.epochs), "external_top1": external_score.get("top1"), "external_top5": external_score.get("top5"), "direct_top1": direct_score.get("top1"), "direct_top5": direct_score.get("top5"), "seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
