# AIFlow Math Ink 1.0 연구 Fork

온라인 수학 필기를 `원본 stroke → 형태/그리딩 → 문자 Top-k → 보수적 결정`으로 분리하는 상용 1.0 연구 트랙입니다. 이 저장소는 0.9 릴리스의 독립 Fork이며, 0.9 작업본을 변경하지 않습니다.

## 1.0 데이터 정책

- 데이터 카탈로그: [datasets/DATASET_CATALOG.md](datasets/DATASET_CATALOG.md)
- 현재 활용·승인 데이터 목록: [datasets/UTILIZED_DATASETS.md](datasets/UTILIZED_DATASETS.md)
- 2026-08-10 원본 재감사: [research/DATASET_REAUDIT_20260810.md](research/DATASET_REAUDIT_20260810.md)
- HWR 논문·데이터셋 포화점 검토: [research/HWR_PAPER_DATASET_REVIEW_20260810.md](research/HWR_PAPER_DATASET_REVIEW_20260810.md)
- 외부 원본은 `datasets/`에 출처·상태별로 분리한다. 압축 원본은 로컬 보관 대상이며 Git에 넣지 않는다.
- `10_approved_external`만 승인된 외부 HWR 학습 풀이다. 승인과 현재 체크포인트 반영은 구분한다.
- `20_reaudit_required`는 재감사 완료 전 학습·평가·모델선택에 사용할 수 없다.
- ISGL, UCI Character Trajectories, 정제 HWRT, BDSHWA는 2026-08-10 용도 제한 조건으로 채택했다.
- 외부 데이터는 box-local 문자 후보 학습에만 쓰고, 수식 grouping·ownership·관계·최종 결정은 프로젝트 보유 수식 데이터가 담당한다.
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

## 1.0 MASK 문맥 재랭커

Google BERT-Tiny에 372개 문자 token과 7개 공간관계 token을 추가한 후보 보존형 재랭커를 구현했다. 프로젝트 ownership 47식만 학습하고 CROHME는 전이 평가에만 사용했다. 직접 writer-LOO Top-1은 73.46%→84.36%, CROHME 지원 문자 Top-1은 69.50%→76.36%였지만, 프로젝트 학습원에 `|·O·o` 정답이 없어 CROHME strict macro가 42.64%→37.69%로 하락했다.

따라서 체크포인트는 shadow component로 생성했으며 기존 372-head의 기본 출력을 자동 교체하지 않았다. 구조·지표·동형문자별 실패와 재현 방법은 [research/MASKED_CONTEXT_BERT_TINY_20260814.md](research/MASKED_CONTEXT_BERT_TINY_20260814.md)에 기록했다.

## 1.0 역할 기반 문맥 최종 확정기

라벨별 MASK 출력 헤드는 프로젝트에 없는 `|·O·o`를 억제하므로 폐기했다. 새 후단은 HWR Top-5의 정확한 문자 형상을 그대로 두고, 고정 BERT-Tiny attention과 프로젝트 수식의 역할 문법으로 `숫자·연산자·구분자·피연산자·기타` 역할만 확정한다. 같은 역할 안에서는 HWR 순위를 바꾸지 않는다.

직접 수집 writer-LOO Top-1은 73.46%→84.83%, strict macro는 57.94%→74.49%, 수식 exact는 34.04%→51.06%였다. CROHME 진단은 Top-1 69.50%→76.14%, strict macro 42.64%→52.34%, 수식 exact 15.85%→24.09%였다. 모든 9,746건에서 원래 Top-5 밖 출력과 grouping 변경은 0건이다.

체크포인트는 재로딩 예측 불일치 0건을 확인했지만 아직 shadow 상태다. CROHME는 반복 개발에 사용된 CC BY-NC 진단셋이며, 상용 기본 승격에는 새 프로젝트 소유 writer/formula acceptance set이 필요하다. 구현·경계·재현 절차는 [research/CONTEXT_ROLE_FINALIZER_20260819.md](research/CONTEXT_ROLE_FINALIZER_20260819.md)에 기록했다.

## 1.0 증류형 수학 문맥 최종 확정기

모델 저장소가 MIT로 표시된 MathBERTa를 연구용 학습 교사로 사용해 372-class 문맥 분포를 만들고, 이를 경량 Google BERT-Tiny에 증류했다. 실행 시에는 586MB 교사가 필요하지 않으며 33.8MB 학생 체크포인트만 사용한다. 출력은 항상 HWR Top-5 안에서만 선택하고 stroke grouping은 바꾸지 않는다.

직접 수집 writer-LOO Top-1은 88.63%, strict macro는 75.74%, 수식 exact는 68.09%였다. CROHME 진단은 각각 78.69%, 57.68%, 27.44%였다. 이는 기존 역할 기반 r2의 대응 지표를 모두 상회하며, 후보 이탈과 grouping 변경은 9,746건 전체에서 0건이다.

연구 gate는 통과했지만 자동 기본 교체는 보류한다. MathBERTa 학습원인 ArXMLiv·Math StackExchange의 상용 권리 감사도 통과하지 못했고, CROHME는 반복 관찰된 CC BY-NC 진단셋이며 `/·o`는 여전히 약하다. 새 프로젝트 소유 writer/formula acceptance와 상업 이용이 명확한 교사 교체가 모두 필요하다. 구조·모델 감사·재현 절차는 [research/DISTILLED_MATH_CONTEXT_20260819.md](research/DISTILLED_MATH_CONTEXT_20260819.md)에 기록했다.

## 1.0 자체 수식 문맥 확정 레이어

외부 사전학습 가중치와 외부 텍스트 말뭉치를 제거하고, 무작위 초기화 2-layer Transformer를 프로젝트 소유 47식과 저장소 내 결정적 수식 DSL만으로 학습했다. 실행 시 HWR Top-5와 7개 공간관계만 읽고 기존 후보 중 하나를 확정하며, 새 token·삭제·stroke regrouping은 허용하지 않는다.

직접수집 writer-LOO는 Top-1 89.10%, strict macro 79.40%, 수식 exact 65.96%, 기존 정답 회귀 0건이다. CROHME 개발진단은 각각 77.82%, 52.01%, 27.95%다. 체크포인트 재로딩 불일치는 직접·CROHME 모두 0건이며 상업 학습권리 gate는 통과했다.

다만 직접 47식과 CROHME가 반복 개발에 사용됐고 `/·|·o` 전이 비회귀가 확인되지 않아 자동 기본 교체는 하지 않는다. 현재 상태는 `shadow owned formula-context candidate`이며, 구조·권리 경계·실패 문자·재현 절차는 [research/OWNED_FORMULA_CONTEXT_20260822.md](research/OWNED_FORMULA_CONTEXT_20260822.md)에 기록했다.

### 신규 수집 데이터 적용과 실제 추론 경계

2026-08-19 수집분을 포함한 ownership 96식 중 고정 372-class HWR에 없는 `t`가 든 한 식을 통째로 제외하고 95식·387글자를 감사했다. 기존 r6을 새 수집 48식에 재학습 없이 적용한 뒤, Top-5 lattice에서 정확한 산술 등식과 누락된 괄호 짝만 복원하는 결정적 의미 가드를 연결했다. HWR Top-1 65.91%는 최종 84.09%, 수식 exact 31.25%는 60.42%로 올랐고 r6 정답 회귀는 0건이었다. 반면 95식 재학습 challenger, 반복 추론, 등호 없는 산술 강제 규칙과 기호를 숫자로 바꾸는 과도한 등식 보정은 효과 부족 또는 안전성 문제로 기각했다.

실제 추론은 [scripts/finalize_formula_context_v1.py](scripts/finalize_formula_context_v1.py)가 담당한다. 정답 `label` 없이 HWR Top-5·확률·수식 순서·공간관계만 받고, 후보 추가·글자 삭제·stroke regrouping 없이 후보 하나를 확정한다. 산술 어휘 밖 기호가 이미 선택된 식은 등식 보정에서 제외한다. CROHME도 r6 대비 Top-1 77.82%→78.00%, 수식 exact 27.95%→28.46%, 추가 개선/회귀 17/0이었다. r6+의미 가드는 선택 경로지만 아직 opt-in shadow 상태다. 새 수집분에는 `|`와 `o` 정답이 없어 상용 정확도 gate는 유지하며, 전체 감사는 [research/OWNED_CONTEXT_ARRIVAL_AUDIT_20260819.md](research/OWNED_CONTEXT_ARRIVAL_AUDIT_20260819.md)에 기록했다.
