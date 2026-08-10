# AIFlow Math Ink 1.0 연구 Fork

온라인 수학 필기를 `원본 stroke → 형태/그리딩 → 문자 Top-k → 보수적 결정`으로 분리하는 상용 1.0 연구 트랙입니다. 이 저장소는 0.9 릴리스의 독립 Fork이며, 0.9 작업본을 변경하지 않습니다.

## 1.0 데이터 정책

- 데이터 카탈로그: [datasets/DATASET_CATALOG.md](datasets/DATASET_CATALOG.md)
- 2026-08-10 원본 재감사: [research/DATASET_REAUDIT_20260810.md](research/DATASET_REAUDIT_20260810.md)
- HWR 논문·데이터셋 포화점 검토: [research/HWR_PAPER_DATASET_REVIEW_20260810.md](research/HWR_PAPER_DATASET_REVIEW_20260810.md)
- 외부 원본은 `datasets/`에 출처·상태별로 분리한다. 압축 원본은 로컬 보관 대상이며 Git에 넣지 않는다.
- `10_approved_external`만 문자 HWR 보조학습 후보이다.
- `20_reaudit_required`는 재감사 완료 전 학습·평가·모델선택에 사용할 수 없다.
- 이번 재감사에서 ISGL, UCI Character Trajectories, HWRT, BDSHWA는 모두 차단 상태를 유지했다.
- 비상업 데이터는 제품 학습에 사용하지 않는다.

## 공개물

- `paper/AIFlow_Math_Ink_0_9_Draft.md`: 연구 논문 초안
- `reports/public_retrain_seed31_metrics.json`: 공개 데이터 재학습·필기 replay 지표
- `reports/crohme2019_valid.json`: CROHME2019 비상업 검증
- `scripts/build_public_dataset09.py`: 비식별 공개 데이터 생성
- `scripts/validate_public_dataset09.py`: 공개 데이터 무결성 검사
- `scripts/train_normalized_balanced_touch_calibrator09.py`: 공개 데이터 재학습
- Hugging Face model: https://huggingface.co/cwLeeDev/aiflow-math-ink-0.9
- Hugging Face dataset: https://huggingface.co/datasets/cwLeeDev/aiflow-math-ink-0.9-dataset

## 공개 데이터

데이터 관리자는 참여자 공개 동의 완료를 확인했습니다. 119개 수식의 stroke sequence를 공개하되 원본 contributor/session/prompt ID, 기록 시각, 이미지 URL, IP·원본 경로는 제거했습니다. dataset-local writer/session 별칭과 formula-relative 시간만 제공합니다.

- valid: 110건
- pending: 4건, 기본 학습 제외
- reject: 5건, 기본 학습 제외
- 수동 ownership: 47개 수식·211개 문자

## 결과

공개 JSONL 재학습은 기존 seed31을 재현했습니다. 49개 수식·171개 문자 replay에서 Top-1 87.72%, Top-5 98.83%, 수식 exact 67.35%입니다. 고신뢰 baseline 보존 guard는 집계 정확도를 유지하면서 회귀를 3건에서 2건으로 줄였습니다.

CROHME2019 CC BY-NC valid 985식에서는 truth-group 문자 Top-1이 56.96%→74.11%, end-to-end group-token Top-1 recall이 48.20%→62.17%, fully-covered 수식 exact가 7.95%→11.98%였습니다. guarded 후보의 수식 exact는 12.10%입니다.

## 상태

- `research_adopted=true`
- `product_adopted=false`
- CROHME2019은 비상업 검증으로만 사용
- 자동채점 모델이 아니라 필기 인식 연구 후보
- writer-disjoint 실사용 검증과 독립 guard threshold 검증은 아직 필요
