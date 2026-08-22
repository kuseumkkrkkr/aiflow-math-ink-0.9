#!/usr/bin/env python3
"""실데이터 없이 벡터 기호와 48 Hz 펜 물리로 128x5 온라인 잉크를 만든다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
import numpy as np


POINTS = 128
CHANNELS = ("x", "y", "delta_t", "stroke_start", "observed")
SCHEMA = "aiflow-physics-only-glyph-teacher/v1"
FONTSETS = ("stix", "dejavusans")
RASTER_SIZE = 192
PROCEDURAL_POINT_LABELS = frozenset({r"\cdot", r"\bullet"})

# MathText에 없는 소수 기호만 상업 사용 가능한 로컬 DejaVu/STIX 글리프로
# 치환한다. 치환 내역은 산출물에 그대로 기록되어 감사 가능하게 한다.
FALLBACK_EXPRESSIONS = {
    r"\&": ("&", False),
    r"\Bowtie": (r"\bowtie", True),
    r"\astrosun": ("☉", False),
    r"\celsius": ("℃", False),
    r"\diameter": ("⌀", False),
    r"\dotsc": (r"\cdots", True),
    r"\female": ("♀", False),
    r"\fint": (r"\int", True),
    r"\fullmoon": ("●", False),
    r"\iddots": ("⋰", False),
    r"\leftmoon": ("☾", False),
    r"\lightning": ("↯", False),
    r"\llbracket": ("⟦", False),
    r"\lozenge": ("◊", False),
    r"\male": ("♂", False),
    r"\mars": ("♂", False),
    r"\mathsection": ("§", False),
    r"\ohm": ("Ω", False),
    r"\parr": ("⅋", False),
    r"\pounds": ("£", False),
    r"\rrbracket": ("⟧", False),
    r"\shortrightarrow": (r"\rightarrow", True),
    r"\square": ("□", False),
    r"\sun": ("☼", False),
    r"\varoiint": (r"\oiint", True),
    r"\varsubsetneq": (r"\subsetneq", True),
    r"\venus": ("♀", False),
    r"\with": ("&", False),
}


@dataclass(frozen=True)
class GlyphTemplateV1:
    """한 클래스의 라이선스 고정 벡터-스켈레톤 토폴로지를 보관한다."""

    label: str
    fontset: str
    expression: str
    fallback_used: bool
    strokes: tuple[np.ndarray, ...]
    raster_pixels: int
    components: int


@dataclass(frozen=True)
class WriterPhysicsV1:
    """한 합성 작가가 모든 클래스에서 공유하는 공간·운동 잠재변수다."""

    writer_id: str
    split: str
    slant: float
    rotation_radians: float
    width_scale: float
    height_scale: float
    bow_x: float
    bow_y: float
    hook: float
    natural_frequency_hz: float
    damping_ratio: float
    minimum_jerk_strength: float
    curvature_slowdown: float
    tremor_hz: float
    tremor_amplitude: float
    drift_amplitude: float
    reverse_probability: float


def stable_seed(*values: object) -> int:
    """플랫폼과 실행 순서에 독립적인 63비트 난수 시드를 만든다."""

    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def _expression(label: str) -> tuple[str, bool, bool]:
    """출력 라벨을 MathText 또는 일반 Unicode 렌더 표현으로 바꾼다."""

    if label in FALLBACK_EXPRESSIONS:
        value, math_mode = FALLBACK_EXPRESSIONS[label]
        return value, math_mode, True
    value = label.replace(r"\mathds", r"\mathbb")
    value = value.replace(r"\sqrt{}", r"\sqrt{\ }")
    value = value.replace(r"\checked", r"\checkmark")
    return value, True, value != label


def _render_mask(label: str, fontset: str, size: int = RASTER_SIZE) -> tuple[np.ndarray, dict]:
    """번들 폰트의 벡터 경로를 고해상도 이진 마스크로만 임시 렌더한다."""

    expression, math_mode, fallback_used = _expression(label)
    displayed = f"${expression}$" if math_mode else expression
    properties = FontProperties(family="DejaVu Sans")
    with rc_context({"mathtext.fontset": fontset}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = TextPath((0.0, 0.0), displayed, size=1.0, prop=properties, usetex=False)
    if len(path.vertices) < 2:
        raise ValueError(f"empty vector glyph for {label!r} using {fontset}")
    bounds = path.get_extents()
    width = max(float(bounds.width), 1.0e-6)
    height = max(float(bounds.height), 1.0e-6)
    pad = 0.16 * max(width, height)
    figure = Figure(figsize=(1.0, 1.0), dpi=size, facecolor="white")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(bounds.x0) - pad, float(bounds.x1) + pad)
    axis.set_ylim(float(bounds.y0) - pad, float(bounds.y1) + pad)
    axis.add_patch(PathPatch(path, facecolor="black", edgecolor="black", linewidth=0.0))
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    mask = np.where(rgba[:, :, :3].mean(axis=2) < 160.0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError(f"rasterization produced no foreground for {label!r}")
    return mask, {
        "expression": expression,
        "math_mode": math_mode,
        "fallback_used": fallback_used,
        "font_warnings": [str(item.message) for item in caught],
    }


def _neighbors(node: tuple[int, int], nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """스켈레톤 픽셀의 표준 8방향 이웃을 결정한다."""

    row, column = node
    values: list[tuple[int, int]] = []
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            if delta_row == 0 and delta_column == 0:
                continue
            candidate = (row + delta_row, column + delta_column)
            if candidate not in nodes:
                continue
            values.append(candidate)
    return sorted(values)


def _edge(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """방향 없는 픽셀 간선을 정렬된 키로 만든다."""

    return (first, second) if first <= second else (second, first)


def _trace_component(component: np.ndarray) -> list[list[tuple[int, int]]]:
    """한 연결 성분을 연속 DFS 필기 경로 하나로 바꾸고 분기만 되짚는다."""

    nodes = {tuple(int(value) for value in row) for row in component}
    extent = component.max(axis=0) - component.min(axis=0)
    if len(nodes) == 1 or int(extent.max()) <= 8:
        center = component.mean(axis=0)
        radius = max(2.0, 0.55 * float(extent.max()))
        clock = np.linspace(0.0, math.tau, 12, endpoint=True)
        return [[
            (
                int(round(center[0] + radius * math.sin(value))),
                int(round(center[1] + radius * math.cos(value))),
            )
            for value in clock
        ]]
    adjacency = {node: _neighbors(node, nodes) for node in nodes}
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    endpoints = sorted(node for node, values in adjacency.items() if len(values) == 1)
    start = endpoints[0] if endpoints else min(nodes)
    route = [start]

    def visit(node: tuple[int, int]) -> None:
        """미방문 간선을 따라가고 분기 끝에서는 실제 선을 되짚어 복귀한다."""

        for following in adjacency[node]:
            key = _edge(node, following)
            if key in visited:
                continue
            visited.add(key)
            route.append(following)
            visit(following)
            if route[-1] != node:
                route.append(node)

    visit(start)
    return [route]


def _skeleton_strokes(mask: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
    """이진 글리프를 단일 픽셀 스켈레톤으로 만든 뒤 순서형 획으로 바꾼다."""

    skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    pixel_paths: list[list[tuple[int, int]]] = []
    if not np.any(skeleton):
        foreground = np.argwhere(mask > 0)
        if len(foreground) == 0:
            raise ValueError("glyph mask has no foreground")
        center = foreground.mean(axis=0)
        radius = max(2.0, math.sqrt(len(foreground) / math.pi) * 0.42)
        clock = np.linspace(0.0, math.tau, 24, endpoint=True)
        pixel_paths.append([
            (
                int(round(center[0] + radius * math.sin(value))),
                int(round(center[1] + radius * math.cos(value))),
            )
            for value in clock
        ])
        count = 2
        labels = np.zeros_like(mask, dtype=np.int32)
    else:
        count, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8), connectivity=8)
    for component_id in range(1, count):
        component = np.argwhere(labels == component_id)
        if len(component) == 0:
            continue
        pixel_paths.extend(_trace_component(component))
    if not pixel_paths:
        raise ValueError("glyph skeleton produced no traceable path")
    all_pixels = np.asarray([point for path in pixel_paths for point in path], dtype=np.float64)
    low = all_pixels.min(axis=0)
    high = all_pixels.max(axis=0)
    extent = max(float((high - low).max()), 1.0)
    center = (low + high) * 0.5
    strokes = []
    for path in pixel_paths:
        points = np.asarray(path, dtype=np.float64)
        xy = np.column_stack((points[:, 1] - center[1], points[:, 0] - center[0]))
        xy = xy / extent * 0.84 + 0.5
        if len(xy) >= 3:
            keep = [0]
            for index in range(1, len(xy) - 1):
                before = xy[index] - xy[keep[-1]]
                after = xy[index + 1] - xy[index]
                cross = abs(float(before[0] * after[1] - before[1] * after[0]))
                if cross > 1.0e-7 or np.linalg.norm(xy[index] - xy[keep[-1]]) >= 0.025:
                    keep.append(index)
            keep.append(len(xy) - 1)
            xy = xy[sorted(set(keep))]
        strokes.append(np.clip(xy, 0.0, 1.0).astype(np.float32))
    strokes.sort(key=lambda value: (float(value[:, 0].min()), float(value[:, 1].min()), -len(value)))
    if len(strokes) > 60:
        lengths = np.asarray([_path_length(value) for value in strokes])
        keep = np.argsort(-lengths)[:60]
        strokes = [strokes[int(index)] for index in sorted(keep)]
    return tuple(strokes), count - 1


def build_template(label: str, fontset: str) -> GlyphTemplateV1:
    """한 출력 라벨의 벡터 글리프를 데이터 독립 온라인 획 템플릿으로 만든다."""

    if fontset not in FONTSETS:
        raise ValueError(f"unsupported fontset: {fontset}")
    if label in PROCEDURAL_POINT_LABELS:
        clock = np.linspace(0.0, math.tau, 24, endpoint=True)
        circle = np.column_stack((0.5 + 0.30 * np.cos(clock), 0.5 + 0.30 * np.sin(clock))).astype(np.float32)
        return GlyphTemplateV1(
            label=label,
            fontset=fontset,
            expression="procedural_point_loop",
            fallback_used=True,
            strokes=(circle,),
            raster_pixels=0,
            components=1,
        )
    mask, metadata = _render_mask(label, fontset)
    strokes, components = _skeleton_strokes(mask)
    if not strokes or len(strokes) > 64:
        raise ValueError(f"unsafe stroke count for {label!r}: {len(strokes)}")
    return GlyphTemplateV1(
        label=label,
        fontset=fontset,
        expression=str(metadata["expression"]),
        fallback_used=bool(metadata["fallback_used"]),
        strokes=strokes,
        raster_pixels=int(np.count_nonzero(mask)),
        components=components,
    )


def build_templates(labels: Iterable[str]) -> dict[str, tuple[GlyphTemplateV1, ...]]:
    """모든 라벨에 대해 두 독립 번들 폰트의 절차형 토폴로지를 만든다."""

    result: dict[str, tuple[GlyphTemplateV1, ...]] = {}
    for label in labels:
        values = tuple(build_template(str(label), fontset) for fontset in FONTSETS)
        result[str(label)] = values
    return result


def writer_physics(writer_index: int, split: str, seed: int) -> WriterPhysicsV1:
    """분할별 범위 안에서 클래스와 독립적인 합성 작가 잠재변수를 뽑는다."""

    if writer_index < 0 or split not in {"train", "development", "stress"}:
        raise ValueError("invalid writer profile request")
    random = np.random.default_rng(stable_seed(SCHEMA, seed, split, writer_index))
    strength = 1.0 if split != "stress" else 1.32

    def uniform(low: float, high: float) -> float:
        """현재 작가 난수원에서 닫힌 설정 범위의 실수를 뽑는다."""

        return float(random.uniform(low, high))

    return WriterPhysicsV1(
        writer_id=f"procedural-{split}-{writer_index:03d}",
        split=split,
        slant=uniform(-0.22, 0.22) * strength,
        rotation_radians=math.radians(uniform(-7.0, 7.0) * strength),
        width_scale=uniform(0.78, 1.22) ** strength,
        height_scale=uniform(0.82, 1.18) ** strength,
        bow_x=uniform(-0.026, 0.026) * strength,
        bow_y=uniform(-0.026, 0.026) * strength,
        hook=uniform(-0.020, 0.020) * strength,
        natural_frequency_hz=uniform(5.0, 7.5) / min(strength, 1.2),
        damping_ratio=uniform(0.72, 1.02),
        minimum_jerk_strength=min(0.42, uniform(0.16, 0.30) * strength),
        curvature_slowdown=min(0.78, uniform(0.28, 0.52) * strength),
        tremor_hz=uniform(7.0, 12.0),
        tremor_amplitude=uniform(0.0010, 0.0030) * strength,
        drift_amplitude=uniform(0.0015, 0.0040) * strength,
        reverse_probability=uniform(0.18, 0.62),
    )


def _path_length(points: np.ndarray) -> float:
    """한 획의 누적 유클리드 길이를 반환한다."""

    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """획을 호길이 기준으로 지정 점 수에 보간한다."""

    if count < 2:
        raise ValueError("a procedural stroke needs at least two points")
    if len(points) == 1:
        points = np.vstack((points, points + np.asarray([[0.0, 1.0e-4]])))
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    domain = np.concatenate(([0.0], np.cumsum(distances)))
    if domain[-1] <= 1.0e-9:
        domain = np.arange(len(points), dtype=np.float64)
    grid = np.linspace(float(domain[0]), float(domain[-1]), count)
    return np.column_stack([np.interp(grid, domain, points[:, axis]) for axis in range(2)])


def _allocate(strokes: list[np.ndarray], total: int = POINTS) -> list[int]:
    """각 획에 최소 두 점을 보존하며 총 128점을 길이 비례 배정한다."""

    if len(strokes) * 2 > total:
        raise ValueError(f"{len(strokes)} strokes cannot fit safely in {total} points")
    remaining = total - 2 * len(strokes)
    weights = np.asarray([max(_path_length(value), 0.006) for value in strokes], dtype=np.float64)
    exact = remaining * weights / weights.sum()
    base = np.floor(exact).astype(np.int64)
    for index in np.argsort(-(exact - base))[: remaining - int(base.sum())]:
        base[int(index)] += 1
    return [int(value) + 2 for value in base]


def _affine_strokes(strokes: list[np.ndarray], profile: WriterPhysicsV1, random: np.random.Generator) -> list[np.ndarray]:
    """작가 고정 affine·bow·terminal hook을 각 템플릿 획에 적용한다."""

    cosine, sine = math.cos(profile.rotation_radians), math.sin(profile.rotation_radians)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    output: list[np.ndarray] = []
    for source in strokes:
        points = np.asarray(source, dtype=np.float64).copy()
        center = points - 0.5
        transformed = np.empty_like(center)
        transformed[:, 0] = profile.width_scale * center[:, 0] + profile.slant * (-center[:, 1])
        transformed[:, 1] = profile.height_scale * center[:, 1]
        transformed = transformed @ rotation.T + 0.5
        clock = np.linspace(0.0, 1.0, len(transformed))
        envelope = np.sin(math.pi * clock)
        transformed[:, 0] += profile.bow_x * envelope
        transformed[:, 1] += profile.bow_y * envelope
        if len(transformed) >= 2:
            tangent = transformed[-1] - transformed[max(0, len(transformed) - 5)]
            norm = max(float(np.linalg.norm(tangent)), 1.0e-8)
            normal = np.asarray((-tangent[1], tangent[0])) / norm
            terminal = np.clip((clock - 0.72) / 0.28, 0.0, 1.0) ** 2
            transformed += profile.hook * terminal[:, None] * normal[None]
        if random.random() < profile.reverse_probability:
            transformed = transformed[::-1].copy()
        output.append(transformed)
    if len(output) > 1 and random.random() < 0.45:
        order = random.permutation(len(output))
        output = [output[int(index)] for index in order]
    return output


def _minimum_jerk(points: np.ndarray, strength: float, slowdown: float) -> np.ndarray:
    """획 양끝과 급회전에서 속도가 낮아지는 단조 운동 명령을 만든다."""

    count = len(points)
    if count < 3:
        return points.copy()
    delta = np.diff(points, axis=0)
    lengths = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-8)
    tangents = delta / lengths[:, None]
    curvature = np.zeros(count - 1, dtype=np.float64)
    if count > 2:
        turns = np.minimum(np.linalg.norm(np.diff(tangents, axis=0), axis=1), 2.0)
        curvature[1:] = turns
        curvature[:-1] = np.maximum(curvature[:-1], turns)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths * (1.0 + slowdown * curvature))))
    cumulative /= max(float(cumulative[-1]), 1.0e-8)
    clock = np.linspace(0.0, 1.0, count)
    smooth = 10.0 * clock ** 3 - 15.0 * clock ** 4 + 6.0 * clock ** 5
    progress = (1.0 - strength) * clock + strength * smooth
    return np.column_stack([np.interp(progress, cumulative, points[:, axis]) for axis in range(2)])


def _second_order(points: np.ndarray, profile: WriterPhysicsV1) -> np.ndarray:
    """192 Hz semi-implicit Euler 적분으로 펜 끝의 2차 추종 운동을 계산한다."""

    if len(points) < 2:
        return points.copy()
    oversample = 4
    step = 1.0 / (48.0 * oversample)
    omega = math.tau * profile.natural_frequency_hz
    tip = points[0].copy()
    velocity = np.zeros(2, dtype=np.float64)
    output = points.copy()
    for index in range(1, len(points)):
        previous, following = points[index - 1], points[index]
        for substep in range(1, oversample + 1):
            command = previous + (substep / oversample) * (following - previous)
            acceleration = omega * omega * (command - tip) - 2.0 * profile.damping_ratio * omega * velocity
            velocity += acceleration * step
            tip += velocity * step
        output[index] = tip
    clock = np.linspace(0.0, 1.0, len(points))
    correction = clock * clock * (3.0 - 2.0 * clock)
    output += correction[:, None] * (points[-1] - output[-1])
    output[0], output[-1] = points[0], points[-1]
    return output


def _motor_noise(points: np.ndarray, profile: WriterPhysicsV1, random: np.random.Generator) -> np.ndarray:
    """48 Hz 대역제한 떨림과 양끝 0인 저주파 손 드리프트를 더한다."""

    if len(points) < 3:
        return points.copy()
    tangent = np.gradient(points, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-8)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    clock = np.arange(len(points), dtype=np.float64) / 48.0
    phase = float(random.uniform(0.0, math.tau))
    envelope = np.sin(math.pi * np.linspace(0.0, 1.0, len(points))) ** 2
    tremor = profile.tremor_amplitude * envelope * np.sin(math.tau * profile.tremor_hz * clock + phase)
    controls = max(4, math.ceil(len(points) / 18) + 2)
    control_clock = np.linspace(0.0, 1.0, controls)
    target_clock = np.linspace(0.0, 1.0, len(points))
    noise = random.normal(size=(controls, 2))
    drift = np.column_stack([np.interp(target_clock, control_clock, noise[:, axis]) for axis in range(2)])
    bridge = drift - ((1.0 - target_clock[:, None]) * drift[0] + target_clock[:, None] * drift[-1])
    bridge /= max(float(np.sqrt(np.mean(bridge * bridge))), 1.0e-8)
    return points + tremor[:, None] * normal + profile.drift_amplitude * envelope[:, None] * bridge


def _letterbox(strokes: list[np.ndarray]) -> list[np.ndarray]:
    """모든 획의 종횡비를 보존해 단위 정사각형 안에 고정한다."""

    joined = np.concatenate(strokes, axis=0)
    low, high = joined.min(axis=0), joined.max(axis=0)
    center = (low + high) * 0.5
    extent = max(float((high - low).max()), 1.0e-8)
    return [np.clip((value - center) / extent * 0.90 + 0.5, 0.0, 1.0) for value in strokes]


def synthesize(template: GlyphTemplateV1, profile: WriterPhysicsV1, sample_seed: int) -> tuple[np.ndarray, dict]:
    """템플릿 토폴로지와 작가 운동계만으로 한 128x5 온라인 표본을 만든다."""

    random = np.random.default_rng(stable_seed(SCHEMA, sample_seed, template.label, template.fontset, profile.writer_id))
    strokes = _affine_strokes(list(template.strokes), profile, random)
    allocations = _allocate(strokes)
    generated: list[np.ndarray] = []
    for stroke, count in zip(strokes, allocations, strict=True):
        command = _resample(stroke, count)
        command = _minimum_jerk(command, profile.minimum_jerk_strength, profile.curvature_slowdown)
        physical = _second_order(command, profile)
        physical = _motor_noise(physical, profile, random)
        generated.append(physical)
    generated = _letterbox(generated)
    generated = [np.round(value * 8192.0) / 8192.0 for value in generated]
    features = np.zeros((POINTS, len(CHANNELS)), dtype=np.float32)
    offset = 0
    for stroke in generated:
        end = offset + len(stroke)
        features[offset:end, :2] = stroke.astype(np.float32)
        features[offset, 3] = 1.0
        offset = end
    if offset != POINTS:
        raise AssertionError(f"physics generator emitted {offset} points")
    features[:, 2] = 1.0 / (POINTS - 1)
    features[0, 2] = 0.0
    features[:, 4] = 1.0
    if not np.isfinite(features).all() or np.any(features[:, :2] < 0.0) or np.any(features[:, :2] > 1.0):
        raise AssertionError("physics generator emitted invalid values")
    return features, {
        "teacher_schema": SCHEMA,
        "label": template.label,
        "fontset": template.fontset,
        "writer": asdict(profile),
        "template_strokes": len(template.strokes),
        "output_strokes": int(np.sum(features[:, 3] > 0.5)),
        "sample_seed": int(sample_seed),
    }


def trajectory_descriptor(features: np.ndarray) -> np.ndarray:
    """물리 교사용 점유·방향·곡률·방사·토폴로지 기술자를 계산한다."""

    values = np.asarray(features, dtype=np.float64)
    if values.shape != (POINTS, len(CHANNELS)):
        raise ValueError(f"expected {POINTS}x{len(CHANNELS)} features")
    xy = values[:, :2]
    starts = np.flatnonzero(values[:, 3] > 0.5)
    ends = np.r_[starts[1:], POINTS]
    occupancy, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=6, range=((0.0, 1.0), (0.0, 1.0)))
    occupancy = occupancy.ravel() / max(float(occupancy.sum()), 1.0)
    deltas = []
    turns = []
    closures = []
    lengths = []
    short = 0
    for start, end in zip(starts, ends, strict=True):
        stroke = xy[int(start):int(end)]
        delta = np.diff(stroke, axis=0)
        length = float(np.linalg.norm(delta, axis=1).sum()) if len(delta) else 0.0
        lengths.append(length)
        short += length < 0.045
        closures.append(float(np.linalg.norm(stroke[-1] - stroke[0])) / max(length, 1.0e-8))
        if len(delta):
            deltas.append(delta)
            angle = np.arctan2(delta[:, 1], delta[:, 0])
            if len(angle) > 1:
                turns.append((np.diff(angle) + math.pi) % math.tau - math.pi)
    joined_delta = np.concatenate(deltas, axis=0) if deltas else np.zeros((1, 2))
    direction = np.arctan2(joined_delta[:, 1], joined_delta[:, 0])
    direction_hist, _ = np.histogram(direction, bins=12, range=(-math.pi, math.pi))
    direction_hist = direction_hist / max(float(direction_hist.sum()), 1.0)
    joined_turn = np.concatenate(turns) if turns else np.zeros(1)
    turn_hist, _ = np.histogram(joined_turn, bins=8, range=(-math.pi, math.pi))
    turn_hist = turn_hist / max(float(turn_hist.sum()), 1.0)
    centered = xy - xy.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    radius_hist, _ = np.histogram(radius, bins=8, range=(0.0, math.sqrt(0.5)))
    radius_hist = radius_hist / max(float(radius_hist.sum()), 1.0)
    low, high = xy.min(axis=0), xy.max(axis=0)
    width, height = np.maximum(high - low, 1.0e-8)
    covariance = np.cov(xy.T) if len(xy) > 1 else np.zeros((2, 2))
    scalar = np.asarray([
        float(xy[:, 0].mean()), float(xy[:, 1].mean()),
        float(xy[:, 0].std()), float(xy[:, 1].std()),
        float(covariance[0, 1]), float(width), float(height),
        float(width / max(width + height, 1.0e-8)),
        min(len(starts) / 32.0, 1.0), min(sum(lengths) / 12.0, 1.0),
        float(np.mean(np.asarray(closures) < 0.18)), short / max(len(starts), 1),
    ], dtype=np.float64)
    descriptor = np.concatenate((occupancy, direction_hist, turn_hist, radius_hist, scalar))
    if len(descriptor) != 76 or not np.isfinite(descriptor).all():
        raise AssertionError("unexpected teacher descriptor")
    return descriptor.astype(np.float32)


def template_manifest(templates: dict[str, tuple[GlyphTemplateV1, ...]]) -> dict:
    """템플릿 토폴로지와 fallback 사용량을 직렬화 가능한 요약으로 만든다."""

    rows = [template for values in templates.values() for template in values]
    return {
        "schema": SCHEMA,
        "classes": len(templates),
        "templates": len(rows),
        "fontsets": list(FONTSETS),
        "fallback_templates": sum(value.fallback_used for value in rows),
        "stroke_count": {
            "minimum": min(len(value.strokes) for value in rows),
            "median": float(np.median([len(value.strokes) for value in rows])),
            "maximum": max(len(value.strokes) for value in rows),
        },
        "component_count": {
            "minimum": min(value.components for value in rows),
            "median": float(np.median([value.components for value in rows])),
            "maximum": max(value.components for value in rows),
        },
        "fallback_labels": sorted({value.label for value in rows if value.fallback_used}),
    }


def _self_test(vocabulary: Path) -> None:
    """대표 토폴로지와 128x5 계약의 결정적 재실행을 검사한다."""

    payload = json.loads(vocabulary.read_text(encoding="utf-8"))
    labels = list(payload["labels"])
    if len(labels) != 372 or len(set(labels)) != 372:
        raise AssertionError("vocabulary contract is not 372 unique labels")
    selected = ["0", "A", "x", r"\times", r"\int", r"\sqrt{}", r"\female"]
    templates = build_templates(selected)
    profile = writer_physics(0, "development", 20260822)
    for label in selected:
        first, _ = synthesize(templates[label][0], profile, 17)
        second, _ = synthesize(templates[label][0], profile, 17)
        if not np.array_equal(first, second):
            raise AssertionError(f"non-deterministic procedural replay: {label}")
        if first.shape != (POINTS, len(CHANNELS)) or len(trajectory_descriptor(first)) != 76:
            raise AssertionError(f"invalid procedural contract: {label}")
    print(json.dumps({"self_test": "pass", "classes_checked": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    _self_test(root / "research" / "physics_distillation" / "math_labels_372_v1.json")
