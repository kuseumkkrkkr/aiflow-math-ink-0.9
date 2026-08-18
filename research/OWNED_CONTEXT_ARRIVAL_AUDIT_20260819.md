# 신규 수집 데이터 기반 수식 문맥 확정 레이어 감사

상태: **r6 선택 유지 / 정답 없는 런타임 경계 구현 완료 / 자동 기본 승격 보류**

## 결론

문자 형상 모델을 계속 키우는 대신, HWR가 낸 Top-5 후보를 수식 맥락으로 확정하는 레이어를 실제 추론 경계로 분리했다. 기존 r6은 새 수집 48식·176글자에 재학습 없이 적용해 Top-1을 65.91%에서 82.95%, 수식 exact를 31.25%에서 56.25%로 높였고 기존 정답 회귀는 0건이었다.

확장 데이터로 다시 학습한 모델은 직접 수집과 CROHME 모두 r6보다 약해 교체하지 않았다. 사후 앙상블 규칙도 외부 전이 회귀가 확인되어 넣지 않았다. 따라서 제품 호출 경계는 r6 하나로 고정하되 opt-in shadow로 유지한다. 이번 새 수집분에는 정답 `|`와 `o`가 없어 상용 정확도 gate를 통과했다고 판정할 수 없다.

## 수집·데이터 감사

- 수집기 commit: `e7fffa5677fa091bb78fad012a0ef6c9c51ebb16`
- 운영 URL: `https://math-ink-data-collector.vercel.app`
- 폴러: Windows 작업 `AIFlow Math Ink Data Poller`, 마지막 실행 결과 `0`
- 신규 검수 대기 49건: contact sheet 6장으로 전부 육안 확인, blank·목표 불일치 0건
- 전체 공개 후보: 164건 = valid 159, pending 0, reject 5
- 금지 개인정보 키: 0건
- ownership: 96식·393글자, 각 식의 stroke를 정확히 한 번씩 소유 그룹에 배정
- 구성: 기존 47식 + 과거 시각 검수 35식 + 이번 신규 작성자 10식 + 단일 기호 4식

검수 근거는 로컬 비공개 경계에 둔다.

- 데이터 후보: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\public-candidate-20260819-r2`
- 신규 contact sheet: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\inspection-pending-20260819`
- 신규 ownership overlay: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\ownership-review-20260819\inspection`
- 통합 ownership 원본: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\ownership-annotations-20260819.json`

고정 372-class HWR vocabulary에 없는 정답 `t`가 `f(t)+2` 한 식에서 발견됐다. 임의 문자 치환이나 식 일부 사용을 하지 않고 그 식 전체를 문맥 학습·평가에서 제외했다. 최종 문맥 후보는 95식·387글자·7 writer group이다.

## 기존 r6의 신규 수집 사전학습 없는 평가

HWR 정책은 데이터 도착 뒤 바뀌지 않았다. 기존 작성자는 해당 작성자를 제외하고 학습한 고정 writer-LOO head, 처음 본 작성자는 수집 전에 만들어진 고정 product head를 사용했다. 문맥 r6도 기존 47식만으로 이미 학습된 체크포인트를 그대로 사용했다.

| 범위 | 글자/식 | HWR Top-1 | r6 Top-1 | strict micro | strict macro | HWR 수식 exact | r6 수식 exact | 개선/회귀 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 확장 | 387/95 | 70.03% | **86.30%** | 79.17% | 77.19% | 32.63% | **60.00%** | 63/0 |
| 신규 수집 | 176/48 | 65.91% | **82.95%** | 79.07% | 76.02% | 31.25% | **56.25%** | 30/0 |
| 신규 writer만 | 172/44 | 65.12% | **82.56%** | 78.57% | 75.68% | 25.00% | **52.27%** | 30/0 |

신규 수집 strict 동형군은 `/` 5개, `0` 8개, `1` 16개, `\times` 11개, `x` 3개다. r6은 `0` 5→7, `1` 6→12, `\times` 2→10, `x` 0→2로 개선했고 `/`는 3→3으로 유지했다. 하지만 `|`와 `o` 정답 표본은 0개다.

평가 보고서:

- `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\context-candidates-20260819-r3\r6_pretrain_eval_r2.json`
- 후보 cache SHA-256: `a4936b4cfabd4437d2315d7253d536d8fceb8f8ecc736bfa8dcae9eff42b4d61`
- r6 checkpoint SHA-256: `7e0ff1b04ba1071867bfa6811869f242907ebe017d0996597776c6050cc2fc94`

## 확장 재학습 challenger 판정

95식 전체로 같은 구조를 다시 학습한 challenger는 자체 writer-LOO 연구 gate는 넘었지만 r6을 대체하지 못했다.

| 범위 | 모델 | Top-1 | strict micro | strict macro | 수식 exact | 개선/회귀 |
|---|---|---:|---:|---:|---:|---:|
| 확장 직접 writer-LOO | challenger | 85.27% | 72.92% | 70.94% | 57.89% | 60/1 |
| 신규 48식 | r6 | **82.95%** | **79.07%** | **76.02%** | **56.25%** | 30/0 |
| 신규 48식 | challenger | 81.82% | 72.09% | 69.52% | 54.17% | 29/1 |
| CROHME | r6 | **77.82%** | **63.15%** | 52.01% | **27.95%** | 1,073/280 |
| CROHME | challenger | 76.25% | 57.57% | **54.19%** | 25.81% | 1,032/389 |

신규 48식에서 challenger가 r6 오류를 고친 것은 3건, r6 정답을 망친 것은 5건이다. 체크포인트 교체를 기각했다.

- challenger checkpoint SHA-256: `4e967f37e0cd220e4c1e7aa45462d12d40a6bdf0040827ded035a48707f80ddc`
- checkpoint reload mismatch: 직접·CROHME 모두 0건

사후 규칙도 제품에 넣지 않았다.

- ASCII 영숫자 challenger 선택: 신규 수집은 개선됐지만 CROHME strict macro가 52.0136%→51.9954%로 하락했다.
- 숫자 challenger만 선택: 신규 수집 2건 개선·0건 회귀였지만 CROHME에서 8건 개선·28건 회귀했다. 이 중 `+→4` 회귀가 22건이다.

## 정답 없는 제품 추론 경계

기존 평가 경로는 내부 packing 과정에서 정답 `label` 필드를 요구했다. 모델 입력에는 쓰이지 않았지만 실제 제품 입력 계약으로는 잘못된 구조였다. 이를 수정하고 [../scripts/finalize_formula_context_v1.py](../scripts/finalize_formula_context_v1.py)에 재사용 가능한 `OwnedFormulaContextFinalizer`와 CLI를 추가했다.

입력:

- HWR `final_topk` 최대 5개와 내림차순 확률
- 수식 ID, 수식 내 연속 index/length
- box-local geometry와 후보 간 공간관계 계산에 필요한 값
- 정답 `label` 불필요

불변 계약:

- 출력은 입력 Top-5 중 하나
- 새 token 0
- 글자 삭제 0
- grouping mutation 0
- 체크포인트와 HWR vocabulary/hash 불일치 시 즉시 실패

```powershell
$env:PYTHONNOUSERSITE='1'
python scripts\finalize_formula_context_v1.py `
  --checkpoint artifacts\owned_formula_context_20260822_r6\owned_formula_context_product.pt `
  --input D:\path\to\formula_candidates.jsonl.gz `
  --output D:\path\to\formula_finalized.jsonl.gz `
  --device cpu
```

387글자 전체에서 정답 필드를 제거한 뒤 실행한 결과, 저장된 r6 예측과 불일치 0건, 후보 보존율 100%, 새 token 0, grouping mutation 0을 확인했다.

## 승격 조건

- 현재 선택: r6
- 현재 실행 상태: `opt-in shadow context finalizer`
- `commercial_context_training_rights_gate_passed=true`
- `commercial_accuracy_gate_passed=false`
- `automatic_default_replacement=false`

다음 판정에는 collector가 새로 요구하도록 배포된 `|`와 `o` 프로젝트 소유 수식을 실제로 수집한 뒤, 지금 체크포인트와 정책을 고정한 untouched 평가가 필요하다. 후보 보존 100%, 각 동형군 비회귀, 전체 Top-1 비회귀, 수식 exact 비회귀를 동시에 통과하기 전에는 기본 경로로 승격하지 않는다.
