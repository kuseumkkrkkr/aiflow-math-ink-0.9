#!/usr/bin/env python3
"""Writer-disjoint project punctuation calibration for the V1 box-local classifier.

This first adjusts the ``(``, ``)``, and ``=`` rows, then the other observed
project-symbol rows of an external-only math head. Project rows are split by
their already hashed writer group; external training rows are rehearsal only.
Raw data and the frozen external holdout are never written or used as input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from character_tensor_v1 import _json_lines, tensorize
from replay_evaluate_hwr_v1 import score_glyphs_from_rows
from train_character_classifier_v1 import (
    DEFAULT_CANONICAL_ROOT,
    EXTERNAL_TRAINING_SCHEMA,
    INPUT_MODES,
    InkClassifierV1,
    PILOT_SCHEMA,
    _dataset,
    apply_input_mode,
    input_contract,
    input_mode_for_head,
    load_vocabs,
    prepare_cache,
)


PUNCTUATION = ("(", ")", "=")
SEED = 20260812
CALIBRATION_STEPS = 250
PUNCTUATION_DIRECT_BATCH_SIZE = 16
PROJECT_SYMBOL_DIRECT_BATCH_SIZE = 32
REHEARSAL_BATCH_SIZE = 64
REHEARSAL_NEGATIVE_ROWS = 8192
PUNCTUATION_REHEARSAL_LOSS_WEIGHT = 8.0
PROJECT_SYMBOL_REHEARSAL_LOSS_WEIGHT = 16.0
CALIBRATION_LEARNING_RATE = 1e-3
MAX_EXTERNAL_MATH_TOP1_REGRESSION = 0.01
MIN_DIRECT_TOP1_GAIN = 0.05
SCHEMA = "aiflow-project-symbol-head-calibration/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _d_path(path: Path, kind: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"{kind} must remain on D:: {resolved}")
    return resolved


def _copy_head(head: nn.Linear) -> nn.Linear:
    copied = nn.Linear(head.in_features, head.out_features)
    copied.load_state_dict({key: value.detach().cpu().clone() for key, value in head.state_dict().items()})
    return copied


def _load_base(checkpoint_path: Path, math_labels: list[str], auxiliary_labels: list[str], device: torch.device, input_mode: str) -> InkClassifierV1:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {PILOT_SCHEMA, EXTERNAL_TRAINING_SCHEMA}:
        raise ValueError("base checkpoint must be an external-only training schema")
    if checkpoint.get("math_labels") != math_labels or checkpoint.get("auxiliary_labels") != auxiliary_labels:
        raise ValueError("base checkpoint vocabulary does not match the current canonical corpus")
    report = checkpoint.get("report", {})
    if report.get("data_admission", {}).get("project_owned_training"):
        raise ValueError("base checkpoint already contains project-owned training data")
    checkpoint_input_mode = report.get("input_contract", {}).get("observed_channel_mode", "preserve")
    if checkpoint_input_mode not in INPUT_MODES:
        raise ValueError(f"base checkpoint has an unknown input mode: {checkpoint_input_mode}")
    if checkpoint_input_mode != input_mode:
        raise ValueError(f"base checkpoint input mode {checkpoint_input_mode!r} does not match calibration mode {input_mode!r}")
    model = InkClassifierV1(len(math_labels), len(auxiliary_labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def _encode(model: InkClassifierV1, features: np.ndarray | torch.Tensor, device: torch.device, input_mode: str = "preserve", batch_size: int = 256) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        raw_batch = features[start:start + batch_size]
        if isinstance(raw_batch, torch.Tensor):
            raw_batch = raw_batch.detach().cpu().numpy()
        batch = torch.as_tensor(apply_input_mode(raw_batch, input_mode), device=device)
        chunks.append(model.encode(batch).cpu())
    return torch.cat(chunks)


def _direct_rows(canonical_root: Path, math_index: dict[str, int]) -> tuple[np.ndarray, torch.Tensor, list[str], list[dict]]:
    path = canonical_root / "character_classifier_v1" / "project_owned_ownership_eval.jsonl.gz"
    rows = list(_json_lines(path))
    if not rows:
        raise ValueError(f"no project ownership rows: {path}")
    writers = [str(row.get("writer_group", "")) for row in rows]
    if any(not writer for writer in writers):
        raise ValueError("project ownership derivative lacks a hashed writer group")
    labels = [str(row["label"]) for row in rows]
    if any(label not in math_index for label in labels):
        raise ValueError("project ownership label is outside the math vocabulary")
    return (
        np.stack([tensorize(row) for row in rows]).astype(np.float32),
        torch.tensor([math_index[label] for label in labels], dtype=torch.long),
        writers,
        [{"record_id": str(row["record_id"]), "label": label} for row, label in zip(rows, labels, strict=True)],
    )


def _topk_predictions(head: nn.Linear, embeddings: torch.Tensor, labels: list[str], truths: list[dict]) -> dict[str, dict]:
    with torch.inference_mode():
        topk = head(embeddings).topk(k=5, dim=1).indices.tolist()
    return {
        str(truth["record_id"]): {"topk": [labels[index] for index in indices]}
        for truth, indices in zip(truths, topk, strict=True)
    }


def _punctuation_score(truths: list[dict], predictions: dict[str, dict]) -> dict:
    selected = [truth for truth in truths if truth["label"] in PUNCTUATION]
    top1 = sum(predictions[truth["record_id"]]["topk"][0] == truth["label"] for truth in selected)
    top5 = sum(truth["label"] in predictions[truth["record_id"]]["topk"] for truth in selected)
    return {"records": len(selected), "top1_hits": top1, "top5_hits": top5, "top1": top1 / len(selected), "top5": top5 / len(selected)} if selected else {"records": 0, "top1_hits": 0, "top5_hits": 0, "top1": 0.0, "top5": 0.0}


def _score(head: nn.Linear, embeddings: torch.Tensor, labels: list[str], truths: list[dict]) -> tuple[dict, dict[str, dict]]:
    predictions = _topk_predictions(head, embeddings, labels, truths)
    return {**score_glyphs_from_rows(truths, predictions), "punctuation": _punctuation_score(truths, predictions)}, predictions


def _rehearsal_rows(math_train, punctuation_indices: torch.Tensor) -> np.ndarray:
    labels = np.asarray(math_train.labels, dtype=np.int64)
    punctuation = np.isin(labels, punctuation_indices.numpy())
    positives = np.flatnonzero(punctuation)
    negatives = np.flatnonzero(~punctuation)
    if not len(positives) or not len(negatives):
        raise ValueError("external math training cache cannot provide punctuation rehearsal")
    rng = np.random.default_rng(SEED)
    count = min(REHEARSAL_NEGATIVE_ROWS, len(negatives))
    return np.unique(np.concatenate((positives, rng.choice(negatives, size=count, replace=False))))


def _assert_only_rows_changed(base: nn.Linear, calibrated: nn.Linear, mutable_indices: torch.Tensor) -> None:
    immutable = torch.ones(base.out_features, dtype=torch.bool)
    immutable[mutable_indices] = False
    if not torch.equal(base.weight.detach()[immutable], calibrated.weight.detach()[immutable]):
        raise AssertionError("calibration changed an output row outside its declared project labels")
    if not torch.equal(base.bias.detach()[immutable], calibrated.bias.detach()[immutable]):
        raise AssertionError("calibration changed an output bias outside its declared project labels")


def _calibrate_rows(
    base_head: nn.Linear,
    direct_embeddings: torch.Tensor,
    direct_labels: torch.Tensor,
    train_indices: torch.Tensor,
    target_indices: torch.Tensor,
    rehearsal_embeddings: torch.Tensor,
    rehearsal_labels: torch.Tensor,
    direct_batch_size: int,
    rehearsal_loss_weight: float,
    calibration_steps: int,
    calibration_learning_rate: float,
    fold_seed: int,
) -> tuple[nn.Linear, dict]:
    direct_mask = torch.isin(direct_labels[train_indices], target_indices)
    project_train = train_indices[direct_mask]
    train_labels = direct_labels[project_train]
    missing = set(target_indices.tolist()) - set(train_labels.tolist())
    if missing:
        raise ValueError(f"writer-disjoint project training split is missing label indices: {sorted(missing)}")
    counts = torch.bincount(train_labels, minlength=base_head.out_features).double()
    sample_weights = torch.tensor([1.0 / counts[int(label)] for label in train_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=calibration_steps * direct_batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    loader = DataLoader(TensorDataset(direct_embeddings[project_train], train_labels), batch_size=direct_batch_size, sampler=sampler)
    calibrated = copy.deepcopy(base_head).train()
    optimizer = torch.optim.AdamW(calibrated.parameters(), lr=calibration_learning_rate, weight_decay=0.0)
    immutable = torch.ones(base_head.out_features, dtype=torch.bool)
    immutable[target_indices] = False
    replay_generator = torch.Generator().manual_seed(fold_seed + 100)
    losses: list[float] = []
    for direct_batch, target_batch in loader:
        replay_indices = torch.randint(len(rehearsal_labels), (REHEARSAL_BATCH_SIZE,), generator=replay_generator)
        loss = F.cross_entropy(calibrated(direct_batch), target_batch)
        loss += rehearsal_loss_weight * F.cross_entropy(calibrated(rehearsal_embeddings[replay_indices]), rehearsal_labels[replay_indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite project symbol calibration loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        calibrated.weight.grad[immutable] = 0
        calibrated.bias.grad[immutable] = 0
        optimizer.step()
        losses.append(float(loss.detach()))
    calibrated.eval()
    _assert_only_rows_changed(base_head, calibrated, target_indices)
    return calibrated, {"project_train_records": len(project_train), "target_output_rows": target_indices.tolist(), "direct_batch_size": direct_batch_size, "rehearsal_loss_weight": rehearsal_loss_weight, "learning_rate": calibration_learning_rate, "steps": len(losses), "mean_loss": sum(losses) / len(losses)}


def _calibrate_two_stage(
    base_head: nn.Linear,
    direct_embeddings: torch.Tensor,
    direct_labels: torch.Tensor,
    train_indices: torch.Tensor,
    punctuation_indices: torch.Tensor,
    rehearsal_embeddings: torch.Tensor,
    rehearsal_labels: torch.Tensor,
    calibration_steps: int,
    calibration_learning_rate: float,
    fold_seed: int,
) -> tuple[nn.Linear, dict]:
    punctuation_head, punctuation = _calibrate_rows(
        base_head, direct_embeddings, direct_labels, train_indices, punctuation_indices,
        rehearsal_embeddings, rehearsal_labels, PUNCTUATION_DIRECT_BATCH_SIZE,
        PUNCTUATION_REHEARSAL_LOSS_WEIGHT, calibration_steps, calibration_learning_rate, fold_seed,
    )
    train_labels = direct_labels[train_indices]
    other_indices = torch.unique(train_labels[~torch.isin(train_labels, punctuation_indices)])
    calibrated, other = _calibrate_rows(
        punctuation_head, direct_embeddings, direct_labels, train_indices, other_indices,
        rehearsal_embeddings, rehearsal_labels, PROJECT_SYMBOL_DIRECT_BATCH_SIZE,
        PROJECT_SYMBOL_REHEARSAL_LOSS_WEIGHT, calibration_steps, calibration_learning_rate, fold_seed + 1000,
    )
    _assert_only_rows_changed(base_head, calibrated, torch.unique(torch.cat((punctuation_indices, other_indices))))
    return calibrated, {"punctuation": punctuation, "other_observed_symbols": other}


def _aggregate_direct(folds: list[dict], key: str) -> dict:
    records = sum(fold[key]["records"] for fold in folds)
    top1_hits = sum(round(fold[key]["top1"] * fold[key]["records"]) for fold in folds)
    top5_hits = sum(round(fold[key]["top5"] * fold[key]["records"]) for fold in folds)
    punctuation_records = sum(fold[key]["punctuation"]["records"] for fold in folds)
    punctuation_top1 = sum(fold[key]["punctuation"]["top1_hits"] for fold in folds)
    punctuation_top5 = sum(fold[key]["punctuation"]["top5_hits"] for fold in folds)
    return {
        "records": records,
        "top1": top1_hits / records,
        "top5": top5_hits / records,
        "punctuation": {"records": punctuation_records, "top1_hits": punctuation_top1, "top5_hits": punctuation_top5, "top1": punctuation_top1 / punctuation_records, "top5": punctuation_top5 / punctuation_records},
    }


def _compact_external(score: dict) -> dict:
    return {"top1": score["all"]["top1"], "top5": score["all"]["top5"], "math": {"top1": score["math"]["top1"], "top5": score["math"]["top5"], "punctuation": score["math"]["punctuation"]}}


def _external_score(
    head: nn.Linear,
    math_embeddings: torch.Tensor,
    math_labels: list[str],
    math_truth: list[dict],
    auxiliary_predictions: dict[str, dict],
    auxiliary_truth: list[dict],
) -> dict:
    math_score, math_predictions = _score(head, math_embeddings, math_labels, math_truth)
    combined_predictions = {**math_predictions, **auxiliary_predictions}
    return {"math": math_score, "all": score_glyphs_from_rows(math_truth + auxiliary_truth, combined_predictions)}


def _auxiliary_predictions(head: nn.Linear, embeddings: torch.Tensor, labels: list[str], truths: list[dict]) -> dict[str, dict]:
    return _topk_predictions(head, embeddings, labels, truths)


def _writer_split_indices(writers: list[str]) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    groups = sorted(set(writers))
    if len(groups) < 2:
        raise ValueError("writer-disjoint calibration requires at least two hashed writer groups")
    result = []
    for group in groups:
        train = torch.tensor([index for index, writer in enumerate(writers) if writer != group], dtype=torch.long)
        held = torch.tensor([index for index, writer in enumerate(writers) if writer == group], dtype=torch.long)
        result.append((group, train, held))
    return result


def _write_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "calibration_report.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite calibration report: {path}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _self_test() -> None:
    base = nn.Linear(3, 5)
    calibrated = copy.deepcopy(base)
    calibrated.weight.data[1].add_(1.0)
    calibrated.bias.data[1].add_(1.0)
    _assert_only_rows_changed(base, calibrated, torch.tensor([1]))
    calibrated.weight.data[0].add_(1.0)
    try:
        _assert_only_rows_changed(base, calibrated, torch.tensor([1]))
    except AssertionError:
        pass
    else:
        raise AssertionError("output-row integrity check did not fail")
    splits = _writer_split_indices(["a", "b", "a"])
    assert len(splits) == 2 and len(splits[0][2]) == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=False)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-steps", type=int, default=CALIBRATION_STEPS)
    parser.add_argument("--calibration-learning-rate", type=float, default=CALIBRATION_LEARNING_RATE)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="preserve", help="must match the base checkpoint input contract")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--leave-one-writer-out", action="store_true")
    mode.add_argument("--finalize-all-writers", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); print(json.dumps({"self_test": "pass"})); return 0
    if not args.leave_one_writer_out and not args.finalize_all_writers:
        parser.error("use --leave-one-writer-out or --finalize-all-writers")
    if args.cache_dir is None or args.base_checkpoint is None or args.output is None:
        parser.error("--cache-dir, --base-checkpoint, and --output are required")
    if args.calibration_steps < 1 or args.calibration_learning_rate <= 0:
        parser.error("--calibration-steps and --calibration-learning-rate must be positive")
    canonical_root = _d_path(args.canonical_root, "canonical root")
    cache_dir = _d_path(args.cache_dir, "cache")
    checkpoint_path = _d_path(args.base_checkpoint, "base checkpoint")
    output = _d_path(args.output, "output")
    if (output / "calibration_report.json").exists() or (output / "project_symbol_head_checkpoint.pt").exists():
        parser.error(f"calibration output already exists: {output}")
    if not (cache_dir / "cache_manifest.json").is_file():
        parser.error(f"compatible external training cache is required: {cache_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    math_labels, auxiliary_labels = load_vocabs(canonical_root)
    punctuation_indices = torch.tensor([math_labels.index(label) for label in PUNCTUATION], dtype=torch.long)
    cache = prepare_cache(canonical_root, cache_dir, math_labels, auxiliary_labels)
    math_input_mode = input_mode_for_head(args.input_mode, "math")
    auxiliary_input_mode = input_mode_for_head(args.input_mode, "auxiliary")
    math_train = _dataset(cache_dir, cache["sets"]["math_train"], math_input_mode)
    math_eval = _dataset(cache_dir, cache["sets"]["math_eval"], math_input_mode)
    auxiliary_eval = _dataset(cache_dir, cache["sets"]["auxiliary_eval"], auxiliary_input_mode)
    math_truth = list(_json_lines(cache_dir / cache["sets"]["math_eval"]["truth"]))
    auxiliary_truth = list(_json_lines(cache_dir / cache["sets"]["auxiliary_eval"]["truth"]))
    model = _load_base(checkpoint_path, math_labels, auxiliary_labels, device, args.input_mode)
    math_index = {label: index for index, label in enumerate(math_labels)}
    direct_features, direct_labels, writers, direct_truth = _direct_rows(canonical_root, math_index)
    direct_embeddings = _encode(model, direct_features, device, math_input_mode)
    math_eval_embeddings = _encode(model, math_eval.features, device, math_input_mode)
    auxiliary_embeddings = _encode(model, auxiliary_eval.features, device, auxiliary_input_mode)
    rehearsal_indices = _rehearsal_rows(math_train, punctuation_indices)
    rehearsal_features = np.array(math_train.features[rehearsal_indices], dtype=np.float32, copy=True)
    rehearsal_embeddings = _encode(model, rehearsal_features, device, math_input_mode)
    rehearsal_labels = torch.from_numpy(np.asarray(math_train.labels[rehearsal_indices], dtype=np.int64))
    base_math_head = _copy_head(model.math_head)
    auxiliary_predictions = _auxiliary_predictions(_copy_head(model.latin_aux_head), auxiliary_embeddings, auxiliary_labels, auxiliary_truth)
    base_external = _external_score(base_math_head, math_eval_embeddings, math_labels, math_truth, auxiliary_predictions, auxiliary_truth)
    common = {
        "schema": SCHEMA,
        "base_checkpoint_sha256": _sha256(checkpoint_path),
        "base_checkpoint_name": checkpoint_path.name,
        "seed": SEED,
        "data_policy": {
            "project_owned_training": "only real symbol trajectories from non-held hashed writer groups",
            "project_owned_formula_grouping_training": False,
            "raw_dataset_mutation": False,
            "external_rehearsal": "math_train only; fixed external holdout excluded",
        },
        "input_contract": input_contract(args.input_mode),
        "calibration": {
            "punctuation_labels": list(PUNCTUATION),
            "output_rows_only": True,
            "steps": args.calibration_steps,
            "learning_rate": args.calibration_learning_rate,
            "punctuation_direct_batch_size": PUNCTUATION_DIRECT_BATCH_SIZE,
            "other_symbol_direct_batch_size": PROJECT_SYMBOL_DIRECT_BATCH_SIZE,
            "rehearsal_batch_size": REHEARSAL_BATCH_SIZE,
            "rehearsal_negative_rows": REHEARSAL_NEGATIVE_ROWS,
            "punctuation_rehearsal_loss_weight": PUNCTUATION_REHEARSAL_LOSS_WEIGHT,
            "other_symbol_rehearsal_loss_weight": PROJECT_SYMBOL_REHEARSAL_LOSS_WEIGHT,
            "rehearsal_records": int(len(rehearsal_indices)),
        },
        "base_external": _compact_external(base_external),
    }
    if args.leave_one_writer_out:
        folds: list[dict] = []
        for fold, (writer_group, train_indices, held_indices) in enumerate(_writer_split_indices(writers)):
            calibrated, train_meta = _calibrate_two_stage(base_math_head, direct_embeddings, direct_labels, train_indices, punctuation_indices, rehearsal_embeddings, rehearsal_labels, args.calibration_steps, args.calibration_learning_rate, SEED + fold)
            base_direct, _ = _score(base_math_head, direct_embeddings[held_indices], math_labels, [direct_truth[index] for index in held_indices.tolist()])
            calibrated_direct, _ = _score(calibrated, direct_embeddings[held_indices], math_labels, [direct_truth[index] for index in held_indices.tolist()])
            calibrated_external = _external_score(calibrated, math_eval_embeddings, math_labels, math_truth, auxiliary_predictions, auxiliary_truth)
            folds.append({
                "held_writer_group": writer_group,
                "held_records": len(held_indices),
                "base_direct": base_direct,
                "calibrated_direct": calibrated_direct,
                "calibrated_external": _compact_external(calibrated_external),
                "calibration_train": train_meta,
            })
        baseline_direct = _aggregate_direct(folds, "base_direct")
        calibrated_direct = _aggregate_direct(folds, "calibrated_direct")
        external_math_scores = [fold["calibrated_external"]["math"]["top1"] for fold in folds]
        acceptance = {
            "direct_top1_gain": calibrated_direct["top1"] - baseline_direct["top1"],
            "external_math_top1_worst_regression": base_external["math"]["top1"] - min(external_math_scores),
            "writer_disjoint_direct_gain_gate": calibrated_direct["top1"] - baseline_direct["top1"] >= MIN_DIRECT_TOP1_GAIN,
            "external_math_regression_gate": base_external["math"]["top1"] - min(external_math_scores) <= MAX_EXTERNAL_MATH_TOP1_REGRESSION,
        }
        acceptance["accepted_for_research_candidate"] = acceptance["writer_disjoint_direct_gain_gate"] and acceptance["external_math_regression_gate"]
        report = {
            **common,
            "mode": "leave_one_writer_out",
            "writer_groups": len(_writer_split_indices(writers)),
            "pooled_direct": {"base": baseline_direct, "calibrated": calibrated_direct},
            "folds": folds,
            "acceptance": acceptance,
            "product_adopted": False,
            "limit": "three project writer groups are a small research validation set; formula grouping and decoding remain absent",
        }
    else:
        all_indices = torch.arange(len(direct_labels), dtype=torch.long)
        calibrated, train_meta = _calibrate_two_stage(base_math_head, direct_embeddings, direct_labels, all_indices, punctuation_indices, rehearsal_embeddings, rehearsal_labels, args.calibration_steps, args.calibration_learning_rate, SEED)
        calibrated_external = _external_score(calibrated, math_eval_embeddings, math_labels, math_truth, auxiliary_predictions, auxiliary_truth)
        model.math_head.load_state_dict(calibrated.state_dict())
        report = {
            **common,
            "mode": "finalize_all_writers",
            "calibration_train": train_meta,
            "calibrated_external": _compact_external(calibrated_external),
            "direct_ownership": {"status": "not_scored", "reason": "all project symbol rows were calibration input; scoring them would leak training labels"},
            "product_adopted": False,
            "limit": "use only after the separate leave-one-writer-out report passes; formula grouping and decoding remain absent",
        }
        output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": SCHEMA,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "math_labels": math_labels,
            "auxiliary_labels": auxiliary_labels,
            "report": report,
        }, output / "project_symbol_head_checkpoint.pt")
    _write_report(output, report)
    print(json.dumps({"event": "calibration_complete", "mode": report["mode"], "output": str(output), "product_adopted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
