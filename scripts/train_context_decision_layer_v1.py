#!/usr/bin/env python3
"""Build a candidate-preserving formula-context finalizer over frozen HWR.

The 372-class HWR model owns exact visual identity and supplies Top-5
candidates.  Frozen BERT-Tiny attention and a project-owned role grammar only
decide whether the glyph is acting as a digit, operator, fence, operand, or
other symbol.  The highest HWR candidate inside the selected role is retained.
No token, stroke group, or candidate can be created or removed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from character_tensor_v1 import ROOT, _json_lines
from evaluate_homograph_context_reranker_v1 import _metrics, _role
import train_masked_context_reranker_v1 as masked


SCHEMA = "aiflow-context-role-finalizer/v1"
DEFAULT_INPUT = ROOT / "artifacts" / "homograph_context_20260814"
DEFAULT_PREVIOUS_REPORT = ROOT / "artifacts" / "masked_context_bert_tiny_20260814_r3" / "masked_context_report.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "context_role_finalizer_20260819"
SEED = 20260819

ROLES = ("digit", "operator", "fence", "operand", "other")
ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLES)}
ROLE_ALPHA = 0.5
RARE_LABEL_SUPPORT = 5
RARE_EXTRA_MARGIN = 0.25
CONTEXT_WEIGHTS = (0.25, 0.5, 1.0)
SHAPE_WEIGHTS = (0.5, 1.0, 2.0)
GRAMMAR_WEIGHTS = (0.5, 1.0, 2.0)
MARGINS = (0.25, 0.5, 1.0, 1.5)

OPERATOR_TOKENS = frozenset({
    "+", "-", "=", "/", "<", ">", r"\times", r"\div", r"\pm", r"\mp",
    r"\cdot", r"\ast", r"\star", r"\circ", r"\mid", r"\leq", r"\geq",
    r"\neq", r"\approx", r"\simeq", r"\sim", r"\propto", r"\in", r"\notin",
    r"\subset", r"\supset", r"\subseteq", r"\supseteq", r"\to", r"\rightarrow",
    r"\leftarrow", r"\Rightarrow", r"\Leftarrow", r"\Leftrightarrow",
})
FENCE_TOKENS = frozenset({
    "(", ")", "[", "]", "{", "}", "|", r"\{", r"\}", r"\langle", r"\rangle",
    r"\lceil", r"\rceil", r"\lfloor", r"\rfloor", r"\vert", r"\Vert",
})
RELATION_BREAK_TOKENS = frozenset({
    "=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\simeq", r"\sim",
    r"\propto", r"\in", r"\notin", r"\subset", r"\supset", r"\subseteq",
    r"\supseteq", r"\to", r"\rightarrow", r"\leftarrow", r"\Rightarrow",
    r"\Leftarrow", r"\Leftrightarrow",
})


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False), flush=True)


def _freeze(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _semantic_role(token: str) -> str:
    token = str(token)
    if token in OPERATOR_TOKENS:
        return "operator"
    if token in FENCE_TOKENS:
        return "fence"
    role = _role(token)
    return role if role in ROLE_TO_INDEX else "other"


def _plain_counts(counter: Counter) -> dict[str, int]:
    return {role: int(counter.get(role, 0)) for role in ROLES}


def _fit_role_grammar(rows: list[dict]) -> dict:
    context: dict[str, Counter] = defaultdict(Counter)
    previous: dict[str, Counter] = defaultdict(Counter)
    following: dict[str, Counter] = defaultdict(Counter)
    unigram: Counter = Counter()
    for sequence in masked._formulae(rows).values():
        truth_roles = [_semantic_role(str(row["label"])) for row in sequence]
        for index, role in enumerate(truth_roles):
            left = truth_roles[index - 1] if index else "boundary"
            right = truth_roles[index + 1] if index + 1 < len(truth_roles) else "boundary"
            context[f"{left}\t{right}"][role] += 1
            previous[left][role] += 1
            following[right][role] += 1
            unigram[role] += 1
    return {
        "roles": list(ROLES),
        "alpha": ROLE_ALPHA,
        "context": {key: _plain_counts(value) for key, value in sorted(context.items())},
        "previous": {key: _plain_counts(value) for key, value in sorted(previous.items())},
        "following": {key: _plain_counts(value) for key, value in sorted(following.items())},
        "unigram": _plain_counts(unigram),
        "training_formulas": len(masked._formulae(rows)),
        "training_records": len(rows),
    }


def _smoothed_log_probability(counts: dict[str, int], role: str, alpha: float) -> float:
    denominator = sum(int(value) for value in counts.values()) + alpha * len(ROLES)
    return math.log((int(counts.get(role, 0)) + alpha) / denominator)


def _grammar_score(grammar: dict, left: str, right: str, role: str) -> float:
    empty = {name: 0 for name in ROLES}
    alpha = float(grammar["alpha"])
    return (
        0.4 * _smoothed_log_probability(grammar["context"].get(f"{left}\t{right}", empty), role, alpha)
        + 0.2 * _smoothed_log_probability(grammar["previous"].get(left, empty), role, alpha)
        + 0.2 * _smoothed_log_probability(grammar["following"].get(right, empty), role, alpha)
        + 0.2 * _smoothed_log_probability(grammar["unigram"], role, alpha)
    )


@torch.inference_mode()
def _bert_candidate_logits(model, contract: dict, rows: list[dict], device: torch.device, batch_size: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    packed = masked._pack(rows, contract)
    class_ids = torch.tensor(contract["class_ids"], dtype=torch.long, device=device)
    all_logits = []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        all_logits.append(masked._class_logits(
            model,
            packed["input_ids"][start:stop].to(device),
            packed["attention_mask"][start:stop].to(device),
            packed["mask_positions"][start:stop].to(device),
            class_ids,
        ).cpu())
    logits = torch.cat(all_logits)
    packed_rows = packed["rows"]
    width = max(len(row["final_topk"]) for row in packed_rows)
    candidate_indices = torch.zeros((len(packed_rows), width), dtype=torch.long)
    candidate_mask = torch.zeros((len(packed_rows), width), dtype=torch.bool)
    for row_index, row in enumerate(packed_rows):
        for candidate_index, token in enumerate(row["final_topk"]):
            candidate_indices[row_index, candidate_index] = contract["label_to_index"][token]
            candidate_mask[row_index, candidate_index] = True
    return packed_rows, logits.gather(1, candidate_indices).numpy(), candidate_mask.numpy()


def _normalize_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.full_like(values, -1e9, dtype=np.float32)
    for index in range(len(values)):
        selected = values[index, mask[index]]
        output[index, mask[index]] = (selected - selected.mean()) / (selected.std() + 1e-6)
    return output


def _paired_fence_rows(rows: list[dict]) -> set[str]:
    """Return conservative, nearest non-crossing absolute-value bar pairs."""
    paired: set[str] = set()
    for sequence in masked._formulae(rows).values():
        vertical = []
        for row in sequence:
            if "|" not in row["final_topk"]:
                continue
            geometry = row["geometry"]
            height = float(geometry["height_rel"])
            width = float(geometry["width_rel"])
            if height <= 0.0 or width / height > 0.20:
                continue
            bar_index = row["final_topk"].index("|")
            vertical.append({
                "row": row,
                "x": float(geometry["center_x"]),
                "y": float(geometry["center_y"]),
                "height": height,
                "bar_probability": float(row["final_topk_probabilities"][bar_index]),
                "top_bar": row["final_topk"][0] == "|",
            })
        candidates = []
        for left_index, left in enumerate(vertical):
            for right in vertical[left_index + 1:]:
                low, high = sorted((left["x"], right["x"]))
                inside = [
                    row for row in sequence
                    if low < float(row["geometry"]["center_x"]) < high
                    and row is not left["row"] and row is not right["row"]
                ]
                if not inside or high - low < 0.05:
                    continue
                if any(str(row["final_topk"][0]) in RELATION_BREAK_TOKENS for row in inside):
                    continue
                height_ratio = min(left["height"], right["height"]) / max(left["height"], right["height"])
                if height_ratio < 0.50 or abs(left["y"] - right["y"]) > 0.35 * max(left["height"], right["height"]):
                    continue
                median_inside_height = statistics.median(max(float(row["geometry"]["height_rel"]), 1e-6) for row in inside)
                geometry_support = (
                    left["top_bar"] and right["top_bar"]
                    or min(left["height"], right["height"]) >= 1.20 * median_inside_height
                )
                confidence_support = (
                    left["top_bar"] and right["top_bar"]
                    or left["top_bar"] and right["bar_probability"] >= 0.15
                    or right["top_bar"] and left["bar_probability"] >= 0.15
                )
                if geometry_support and confidence_support:
                    candidates.append((high - low, left, right))
        used: set[str] = set()
        for _, left, right in sorted(candidates, key=lambda value: value[0]):
            left_id, right_id = str(left["row"]["record_id"]), str(right["row"]["record_id"])
            if left_id in used or right_id in used:
                continue
            used.update((left_id, right_id))
            paired.update((left_id, right_id))
    return paired


def _role_components(model, contract: dict, rows: list[dict], grammar: dict, device: torch.device, batch_size: int) -> dict:
    packed_rows, candidate_logits, candidate_mask = _bert_candidate_logits(model, contract, rows, device, batch_size)
    record_to_index = {str(row["record_id"]): index for index, row in enumerate(packed_rows)}
    grammar_by_record: dict[str, list[float]] = {}
    for sequence in masked._formulae(rows).values():
        hwr_roles = [_semantic_role(str(row["final_topk"][0])) for row in sequence]
        for index, row in enumerate(sequence):
            left = hwr_roles[index - 1] if index else "boundary"
            right = hwr_roles[index + 1] if index + 1 < len(hwr_roles) else "boundary"
            grammar_by_record[str(row["record_id"])] = [
                _grammar_score(grammar, left, right, role) for role in ROLES
            ]

    shape_scores = np.full((len(packed_rows), len(ROLES)), -1e9, dtype=np.float32)
    context_scores = np.full_like(shape_scores, -1e9)
    grammar_scores = np.full_like(shape_scores, -1e9)
    role_mask = np.zeros_like(shape_scores, dtype=bool)
    role_choices = np.zeros_like(shape_scores, dtype=np.int64)
    for row_index, row in enumerate(packed_rows):
        probabilities = row["final_topk_probabilities"]
        for role_index, role in enumerate(ROLES):
            candidate_positions = [
                index for index, token in enumerate(row["final_topk"])
                if _semantic_role(str(token)) == role
            ]
            if not candidate_positions:
                continue
            role_mask[row_index, role_index] = True
            role_choices[row_index, role_index] = candidate_positions[0]
            context_scores[row_index, role_index] = max(float(candidate_logits[row_index, index]) for index in candidate_positions)
            shape_scores[row_index, role_index] = math.log(sum(float(probabilities[index]) for index in candidate_positions) + 1e-12)
            grammar_scores[row_index, role_index] = grammar_by_record[str(row["record_id"])][role_index]
    if set(record_to_index) != set(grammar_by_record):
        raise AssertionError("role-context coverage mismatch")
    return {
        "rows": packed_rows,
        "context": _normalize_rows(context_scores, role_mask),
        "shape": shape_scores,
        "grammar": _normalize_rows(grammar_scores, role_mask),
        "role_mask": role_mask,
        "role_choices": role_choices,
        "paired_fence_rows": _paired_fence_rows(rows),
        "candidate_mask": candidate_mask,
    }


def _configuration_grid() -> list[dict | None]:
    output: list[dict | None] = [None]
    for context_weight in CONTEXT_WEIGHTS:
        for shape_weight in SHAPE_WEIGHTS:
            for grammar_weight in GRAMMAR_WEIGHTS:
                for margin in MARGINS:
                    output.append({
                        "context_weight": context_weight,
                        "shape_weight": shape_weight,
                        "grammar_weight": grammar_weight,
                        "margin": margin,
                    })
    return output


def _predict(components: dict, configuration: dict | None, label_support: dict[str, int]) -> tuple[dict[str, str], dict]:
    rows = components["rows"]
    if configuration is None:
        predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in rows}
        return predictions, {"role_switches": 0, "paired_fence_rows": len(components["paired_fence_rows"]), "abstained": True}
    scores = (
        float(configuration["context_weight"]) * components["context"]
        + float(configuration["shape_weight"]) * components["shape"]
        + float(configuration["grammar_weight"]) * components["grammar"]
    )
    predictions, role_switches, fence_forced, rare_margin_kept = {}, 0, 0, 0
    for row_index, row in enumerate(rows):
        top_role = ROLE_TO_INDEX[_semantic_role(str(row["final_topk"][0]))]
        selected_role = int(scores[row_index].argmax())
        record_id = str(row["record_id"])
        fence_index = ROLE_TO_INDEX["fence"]
        if record_id in components["paired_fence_rows"] and components["role_mask"][row_index, fence_index]:
            selected_role, fence_forced = fence_index, fence_forced + 1
        elif selected_role != top_role:
            top_label = str(row["final_topk"][0])
            margin = float(configuration["margin"])
            if int(label_support.get(top_label, 0)) < RARE_LABEL_SUPPORT:
                margin += RARE_EXTRA_MARGIN
            if scores[row_index, selected_role] - scores[row_index, top_role] < margin:
                selected_role = top_role
                rare_margin_kept += int(label_support.get(top_label, 0)) < RARE_LABEL_SUPPORT
        candidate_index = int(components["role_choices"][row_index, selected_role])
        prediction = str(row["final_topk"][candidate_index])
        if prediction not in row["final_topk"]:
            raise AssertionError("context role finalizer invented a candidate")
        predictions[record_id] = prediction
        role_switches += _semantic_role(prediction) != _semantic_role(str(row["final_topk"][0]))
    return predictions, {
        "role_switches": role_switches,
        "paired_fence_rows": len(components["paired_fence_rows"]),
        "fence_forced": fence_forced,
        "rare_margin_kept": rare_margin_kept,
        "abstained": False,
    }


def _selection_key(metrics: dict, configuration: dict | None) -> tuple:
    complexity = -math.inf if configuration is None else -sum(float(value) for value in configuration.values())
    return (
        metrics["all_top1"], metrics["formula_exact"], metrics["strict_macro_top1"],
        metrics["strict_micro_top1"], -metrics["regressed"], -metrics["changed"], complexity,
    )


def _select_configuration(model, contract: dict, rows: list[dict], device: torch.device, batch_size: int, salt: str) -> dict:
    fit_rows, validation_rows = masked._inner_split(rows, salt)
    grammar = _fit_role_grammar(fit_rows)
    label_support = dict(Counter(str(row["label"]) for row in fit_rows))
    components = _role_components(model, contract, validation_rows, grammar, device, batch_size)
    trials = []
    for configuration in _configuration_grid():
        predictions, audit = _predict(components, configuration, label_support)
        trials.append({"configuration": configuration, "metrics": _metrics(validation_rows, predictions), "audit": audit})
    best = max(trials, key=lambda trial: _selection_key(trial["metrics"], trial["configuration"]))
    return {
        "configuration": best["configuration"],
        "selected_metrics": best["metrics"],
        "fit_formulas": len(masked._formulae(fit_rows)),
        "fit_records": len(fit_rows),
        "validation_formulas": len(masked._formulae(validation_rows)),
        "validation_records": len(validation_rows),
        "trials": trials,
    }


def _writer_loo(model, contract: dict, rows: list[dict], device: torch.device, batch_size: int) -> tuple[dict[str, str], list[dict]]:
    predictions, folds = {}, []
    writers = sorted({str(row["writer_group"]) for row in rows})
    for fold_index, writer in enumerate(writers):
        training = [row for row in rows if str(row["writer_group"]) != writer]
        held = [row for row in rows if str(row["writer_group"]) == writer]
        selection = _select_configuration(model, contract, training, device, batch_size, f"role-inner-{writer}")
        grammar = _fit_role_grammar(training)
        label_support = dict(Counter(str(row["label"]) for row in training))
        components = _role_components(model, contract, held, grammar, device, batch_size)
        fold_predictions, audit = _predict(components, selection["configuration"], label_support)
        predictions.update(fold_predictions)
        metrics = _metrics(held, fold_predictions)
        folds.append({
            "held_writer_group": writer,
            "training_formulas": len(masked._formulae(training)),
            "held_formulas": len(masked._formulae(held)),
            "selection": selection,
            "held_metrics": metrics,
            "audit": audit,
        })
        _event(
            "role_writer_fold", fold=fold_index + 1, held_writer=writer,
            configuration=selection["configuration"], held_top1=metrics["all_top1"],
        )
    if set(predictions) != {str(row["record_id"]) for row in rows}:
        raise AssertionError("writer-LOO role-finalizer coverage mismatch")
    return predictions, folds


def _aggregate_configuration(folds: list[dict]) -> dict | None:
    values = [fold["selection"]["configuration"] for fold in folds]
    active = [value for value in values if value is not None]
    if len(active) < 2:
        return None
    return {
        key: float(statistics.median(float(value[key]) for value in active))
        for key in ("context_weight", "shape_weight", "grammar_weight", "margin")
    }


def _save_checkpoint(
    path: Path,
    model_hashes: dict,
    hwr_checkpoint: Path,
    contract: dict,
    grammar: dict,
    configuration: dict | None,
    label_support: dict[str, int],
    rows: list[dict],
) -> str:
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": masked.MODEL_ID,
        "model_revision": masked.MODEL_REVISION,
        "model_file_sha256": model_hashes,
        "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
        "math_labels": contract["labels"],
        "class_tokens": contract["class_tokens"],
        "relation_tokens": contract["relation_tokens"],
        "roles": list(ROLES),
        "role_grammar": grammar,
        "configuration": configuration,
        "label_support": label_support,
        "rare_label_support_threshold": RARE_LABEL_SUPPORT,
        "rare_extra_margin": RARE_EXTRA_MARGIN,
        "training_formulas": len(masked._formulae(rows)),
        "training_records": len(rows),
        "fence_pair_policy": "nearest non-crossing tall narrow bars; relation-containing spans rejected",
        "contract": "frozen HWR Top-5 -> frozen masked attention role evidence -> project role grammar -> HWR-best candidate inside selected role",
    }
    torch.save(payload, path)
    return masked._sha256(path)


def load_context_decision_layer(pretrained: Path, checkpoint_path: Path, device: torch.device):
    pretrained = masked._d_path(pretrained, "pretrained model")
    checkpoint_path = masked._d_path(checkpoint_path, "context-role checkpoint")
    model_hashes = masked._verify_pretrained(pretrained)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("model_id") != masked.MODEL_ID or payload.get("model_revision") != masked.MODEL_REVISION:
        raise ValueError("context-role checkpoint contract mismatch")
    if payload.get("model_file_sha256") != model_hashes or payload.get("roles") != list(ROLES):
        raise ValueError("context-role checkpoint base or role contract mismatch")
    model, _, contract = masked._build_model(pretrained, list(payload["math_labels"]), device, SEED)
    if payload.get("class_tokens") != contract["class_tokens"] or payload.get("relation_tokens") != contract["relation_tokens"]:
        raise ValueError("context-role custom-token contract mismatch")
    _freeze(model)
    return model, contract, payload


def decide_formula_rows(model, contract: dict, checkpoint: dict, rows: list[dict], device: torch.device, batch_size: int = 128) -> tuple[dict[str, str], dict]:
    components = _role_components(model, contract, rows, checkpoint["role_grammar"], device, batch_size)
    return _predict(components, checkpoint["configuration"], checkpoint["label_support"])


def _self_test(pretrained: Path, hwr_checkpoint: Path, device: torch.device) -> None:
    assert _semantic_role("1") == "digit"
    assert _semantic_role("|") == "fence"
    assert _semantic_role(r"\mid") == "operator"
    assert _semantic_role("x") == "operand"
    assert _semantic_role(r"\times") == "operator"
    labels = masked._labels(hwr_checkpoint)
    model, _, contract = masked._build_model(pretrained, labels, device, SEED)
    _freeze(model)
    rows = []
    values = (
        ("|", ["|", "1", r"\mid", "I", "l"], 0.02, 0.8),
        ("x", ["x", r"\times", "X", r"\chi", r"\mathcal{X}"], 0.50, 0.3),
        ("|", ["1", "|", r"\mid", "I", "l"], 0.98, 0.8),
    )
    for index, (truth, candidates, center_x, height) in enumerate(values):
        rows.append({
            "record_id": f"r{index}", "formula_id": "f", "writer_group": "w", "label": truth,
            "final_topk": candidates, "final_topk_probabilities": [0.45, 0.25, 0.15, 0.10, 0.05],
            "geometry": {"center_x": center_x, "center_y": 0.5, "width_rel": 0.02, "height_rel": height},
            "context": {"index": index, "length": 3},
        })
    grammar = _fit_role_grammar(rows)
    components = _role_components(model, contract, rows, grammar, device, 3)
    predictions, _ = _predict(
        components,
        {"context_weight": 0.5, "shape_weight": 1.0, "grammar_weight": 1.0, "margin": 0.5},
        dict(Counter(str(row["label"]) for row in rows)),
    )
    assert set(components["paired_fence_rows"]) == {"r0", "r2"}
    assert all(predictions[str(row["record_id"])] in row["final_topk"] for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=masked.DEFAULT_PRETRAINED)
    parser.add_argument("--hwr-checkpoint", type=Path, default=masked.DEFAULT_PRODUCT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--previous-report", type=Path, default=DEFAULT_PREVIOUS_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    pretrained = masked._d_path(args.pretrained, "pretrained model")
    hwr_checkpoint = masked._d_path(args.hwr_checkpoint, "HWR checkpoint")
    input_root = masked._d_path(args.input, "candidate input")
    output = masked._d_path(args.output, "output")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model_hashes = masked._verify_pretrained(pretrained)
    if args.self_test:
        _self_test(pretrained, hwr_checkpoint, device)
        _event("self_test", status="pass", device=str(device))
        return 0

    report_path = output / "context_role_finalizer_report.json"
    checkpoint_path = output / "context_role_finalizer_product.pt"
    if report_path.exists() or checkpoint_path.exists():
        parser.error(f"refusing to overwrite context-role output: {output}")
    direct_path = input_root / "direct_candidates.jsonl.gz"
    crohme_path = input_root / "crohme_candidates.jsonl.gz"
    if not direct_path.is_file() or not crohme_path.is_file():
        parser.error("candidate caches missing")
    direct_rows, crohme_rows = list(_json_lines(direct_path)), list(_json_lines(crohme_path))
    labels = masked._labels(hwr_checkpoint)
    model, _, contract = masked._build_model(pretrained, labels, device, SEED)
    _freeze(model)

    _event("role_finalizer_start", direct_records=len(direct_rows), direct_formulas=len(masked._formulae(direct_rows)), device=str(device))
    direct_predictions, folds = _writer_loo(model, contract, direct_rows, device, args.batch_size)
    baseline_direct_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in direct_rows}
    direct_baseline = _metrics(direct_rows, baseline_direct_predictions)
    direct_metrics = _metrics(direct_rows, direct_predictions)

    product_configuration = _aggregate_configuration(folds)
    product_grammar = _fit_role_grammar(direct_rows)
    product_label_support = dict(Counter(str(row["label"]) for row in direct_rows))
    direct_components = _role_components(model, contract, direct_rows, product_grammar, device, args.batch_size)
    crohme_components = _role_components(model, contract, crohme_rows, product_grammar, device, args.batch_size)
    direct_refit_predictions, direct_refit_audit = _predict(direct_components, product_configuration, product_label_support)
    crohme_predictions, crohme_audit = _predict(crohme_components, product_configuration, product_label_support)
    baseline_crohme_predictions = {str(row["record_id"]): str(row["final_topk"][0]) for row in crohme_rows}
    crohme_baseline = _metrics(crohme_rows, baseline_crohme_predictions)
    crohme_metrics = _metrics(crohme_rows, crohme_predictions)

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _save_checkpoint(
        checkpoint_path, model_hashes, hwr_checkpoint, contract,
        product_grammar, product_configuration, product_label_support, direct_rows,
    )
    masked._write_prediction_rows(output / "direct_writer_loo_predictions.jsonl.gz", direct_rows, direct_predictions)
    masked._write_prediction_rows(output / "direct_product_refit_predictions.jsonl.gz", direct_rows, direct_refit_predictions)
    masked._write_prediction_rows(output / "crohme_transfer_predictions.jsonl.gz", crohme_rows, crohme_predictions)
    direct_audit = masked._candidate_audit(direct_rows, direct_predictions)
    crohme_candidate_audit = masked._candidate_audit(crohme_rows, crohme_predictions)
    previous = json.loads(args.previous_report.read_text(encoding="utf-8")) if args.previous_report.is_file() else None
    transfer_metrics = ("all_top1", "strict_macro_top1", "formula_exact")
    research_gate = (
        direct_metrics["all_top1"] > direct_baseline["all_top1"]
        and all(crohme_metrics[key] >= crohme_baseline[key] for key in transfer_metrics)
        and direct_audit["candidate_preservation_rate"] == 1.0
        and crohme_candidate_audit["candidate_preservation_rate"] == 1.0
    )
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "visual_identity": "frozen 372-class HWR Top-5; highest HWR candidate inside the selected semantic role",
            "context_attention": "frozen Google BERT-Tiny masked attention over complete formula candidates and seven spatial relations",
            "semantic_roles": list(ROLES),
            "role_grammar": "project-owned smoothed previous/current/next role model; no class-specific output head",
            "structural_guard": "nearest non-crossing absolute-value bar pairs; spans containing relation operators are rejected",
            "uncertainty_guard": f"add {RARE_EXTRA_MARGIN} decision margin when current HWR Top-1 has fewer than {RARE_LABEL_SUPPORT} project examples",
            "candidate_contract": "candidate-only finalization; no token, stroke group, or same-role visual identity mutation",
        },
        "provenance": {
            "model_id": masked.MODEL_ID,
            "model_revision": masked.MODEL_REVISION,
            "model_license": "Apache-2.0",
            "model_file_sha256": model_hashes,
            "hwr_checkpoint_sha256": masked._sha256(hwr_checkpoint),
            "direct_candidate_sha256": masked._sha256(direct_path),
            "crohme_candidate_sha256": masked._sha256(crohme_path),
            "product_checkpoint": str(checkpoint_path),
            "product_checkpoint_sha256": checkpoint_sha256,
        },
        "data_admission": {
            "training": "project-owned accepted ownership only: 47 formulae / 211 glyphs / three writer groups",
            "crohme": "CC BY-NC transfer evaluation only; never used for grammar fitting or configuration selection",
            "canonical_10_and_external_10pct": "not context-evaluable because verified formula sequences and spatial relations are absent",
        },
        "selection": {
            "method": "nested formula-disjoint selection inside outer writer-LOO",
            "context_weight_grid": CONTEXT_WEIGHTS,
            "shape_weight_grid": SHAPE_WEIGHTS,
            "grammar_weight_grid": GRAMMAR_WEIGHTS,
            "margin_grid": MARGINS,
            "writer_loo_folds": folds,
            "product_configuration": product_configuration,
            "bert_parameters_trained": 0,
            "class_specific_head_parameters": 0,
        },
        "evaluation": {
            "direct_writer_loo": {
                "baseline": direct_baseline,
                "context_role_finalizer": direct_metrics,
                "delta": masked._metric_delta(direct_metrics, direct_baseline),
                "candidate_audit": direct_audit,
            },
            "direct_product_refit_resubstitution": {
                "warning": "training-set fit only; not acceptance evidence",
                "metrics": _metrics(direct_rows, direct_refit_predictions),
                "audit": direct_refit_audit,
            },
            "crohme_transfer": {
                "baseline": crohme_baseline,
                "context_role_finalizer": crohme_metrics,
                "delta": masked._metric_delta(crohme_metrics, crohme_baseline),
                "candidate_audit": crohme_candidate_audit,
                "runtime_audit": crohme_audit,
            },
            "previous_full_finetune_mask_reference": previous["evaluation"] if previous else None,
        },
        "decision": {
            "research_gate_passed": research_gate,
            "runtime_status": "shadow context finalizer",
            "automatic_default_replacement": False,
            "reason": "an untouched project-owned formula/writer acceptance set is still required before commercial default promotion",
        },
        "limitations": [
            "The role grammar has only 47 project-owned formulae and three writer groups.",
            "Truth grouping and sequence order are supplied; stroke grouping and LaTeX-equivalent decoding remain outside this evaluation.",
            "CROHME is noncommercial diagnostic evidence and cannot serve as product training data.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _event(
        "role_finalizer_complete", report=str(report_path), checkpoint=str(checkpoint_path),
        direct_top1=direct_metrics["all_top1"], crohme_top1=crohme_metrics["all_top1"],
        crohme_strict_macro=crohme_metrics["strict_macro_top1"], research_gate=research_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
