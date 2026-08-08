---
language:
- ko
- en
library_name: pytorch
license: other
pipeline_tag: feature-extraction
base_model: cwLeeDev/aiflow-math-ink-06-intermediate
tags:
- online-handwriting
- mathematical-expression-recognition
- trajectory
- research
---

# AIFlow Math Ink 0.9

AIFlow Math Ink 0.6 seed17 base/online adapter에 결합하는 정규화·class/writer-balanced additive calibrator 연구 후보이다.

## 사용 계약

- 파일: `normalized_balanced_calibrator.pt`
- SHA-256: `46eb2b283d5d61f707572e0295e4001e759fbe1693660fe92104347108dff725`
- 입력 정규화: glyph bbox crop, aspect-preserving letterbox, local character context
- 필압 사용: 아니오
- 출력: 기존 378 labels + `=`
- 필수 base: `cwLeeDev/aiflow-math-ink-06-intermediate`, seed17 `base_378.pt`, `online_adapter.pt`

이 파일만으로 추론할 수 없다. base encoder의 embedding에 additive correction을 적용해야 한다.

## 평가

| 지표 | 동일 baseline | seed31 후보 |
|---|---:|---:|
| 문자 Top-1 | 50.29% | 87.72% |
| 문자 Top-5 | 70.18% | 98.83% |
| 수식 exact | 26.53% | 67.35% |
| 수식 Top-5 oracle | 42.86% | 95.92% |
| `=` Top-1 | 0.00% | 100.00% |

평가 단위는 기존 수동 ownership replay 49개 수식·171개 문자다. 학습 47개 수식과 정확한 수식/prompt 중복은 없지만 replay에 contributor ID가 없어 writer-disjoint를 증명할 수 없다. 후보는 67개 문자를 개선하고 3개를 악화시켰으며, 특히 `f`는 6/6에서 5/6, `g`는 4/5에서 2/5로 회귀했다.

## 제한

- 연구 후보이며 제품 채택 모델이 아니다.
- CROHME 또는 MathWriting에서 0.9 후보를 재평가한 결과가 아니다.
- 원본 수집 데이터는 재배포 동의가 없어 공개하지 않는다.
- 수식 exact 수치는 작은 내부 replay에 한정된다.

세부 수치는 `metrics.json`, 시각 감사는 `normalized_class_means.png`를 참조한다.
