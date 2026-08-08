# AIFlow Math Ink 0.9 연구 릴리스

온라인 수학 필기를 `형태/그리딩 → 문자 Top-k → 보수적 결정`으로 분리하는 AIFlow Math Ink 연구의 0.9 후보입니다.

## 공개물

- `paper/AIFlow_Math_Ink_0_9_Draft.md`: 연구 논문 초안
- `reports/seed31_metrics.json`: 선택 seed의 원시 지표
- `artifacts/normalized_class_means.png`: 학습 문자별 정규화 평균
- `scripts/train_normalized_balanced_touch_calibrator09.py`: 재현용 학습 진입점
- Hugging Face model: https://huggingface.co/cwLeeDev/aiflow-math-ink-0.9
- Hugging Face aggregate data card: https://huggingface.co/datasets/cwLeeDev/aiflow-math-ink-0.9-data-card

## 핵심 결과

동일한 49개 수식·171개 문자 replay에서 seed31 후보는 문자 Top-1 `50.29% → 87.72%`, Top-5 `70.18% → 98.83%`, 수식 exact `26.53% → 67.35%`였다. 다만 replay에 contributor ID가 없어 writer-disjoint 검증이 아니며 `f`, `g` 회귀도 남아 있다. 따라서 연구 후보만 채택했고 제품 모델로 채택하지 않았다.

## 데이터 공개 제한

119개 수집 레코드의 동의 범위는 AIFlow 모델 학습·검증이다. 원본 stroke, 이미지, contributor/session 식별자는 재배포하지 않는다. Hugging Face dataset 저장소에는 행 단위 데이터가 아닌 비식별 집계 통계와 데이터 카드만 제공한다.

## 체크포인트 계약

`normalized_balanced_calibrator.pt`는 독립 모델이 아니다. `cwLeeDev/aiflow-math-ink-06-intermediate`의 seed17 `base_378.pt`와 `online_adapter.pt` 위에 적용하는 additive head이며 379번째 `=` 출력을 추가한다.

## 상태

- `research_adopted=true`
- `product_adopted=false`
- 상용 또는 준상용 성능 주장 없음
- 자동채점 모델이 아니라 필기 인식 연구 후보
