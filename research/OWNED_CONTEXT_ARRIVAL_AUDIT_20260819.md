# 신규 수집 데이터 기반 수식 문맥 확정 레이어 감사

상태: **r6 + 결정적 의미 가드 선택 / 정답 없는 런타임 구현 완료 / 자동 기본 승격 보류**

## 결론

문자 형상 모델을 계속 키우는 대신, HWR가 낸 Top-5 후보를 수식 맥락으로 확정하는 레이어를 실제 추론 경계로 분리했다. 기존 r6을 새 수집 48식·176글자에 재학습 없이 적용하고 보수적 의미 가드를 연결해 Top-1을 65.91%에서 85.80%, 수식 exact를 31.25%에서 66.67%로 높였고 r6 정답 회귀는 0건이었다.

확장 데이터나 HWR 문맥 노이즈로 다시 학습한 모델은 직접 수집에서 r6을 넘지 못해 교체하지 않았다. 사후 앙상블과 등호 없는 산술 강제 규칙도 외부 전이 회귀가 확인되어 넣지 않았다. 제품 호출 경계는 r6, 정확한 산술 등식 가드, 기존 괄호 잠금 보완 가드, 수평 중위 연산자 가드로 고정하되 opt-in shadow로 유지한다. 이번 새 수집분에는 정답 `|`와 `o`가 없어 상용 정확도 gate를 통과했다고 판정할 수 없다.

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

## 수식 단위 의미 확정 개선

r6은 각 글자를 문맥으로 재정렬하지만 수식 전체가 실제로 성립하는지는 보지 않았다. 정답을 읽지 않고 Top-5 후보 lattice만 사용하는 세 가드를 r6 뒤에 추가했다.

- 산술 등식 가드 v2: x축 순서가 증가하고 인접 세로 구간이 25% 이상 겹치는 3~12글자 평면식만 처리한다. v1은 산술 후보열 안에서 정확한 유리수 등식이 유일하거나 차점 대비 점수 차가 1.5 이상일 때 최대 3글자만 바꾼다. v2는 현재 선택열의 비산술 기호가 정확히 한 슬롯이고 그 슬롯의 산술 후보 중 정확한 등식을 만드는 후보가 정확히 하나일 때만 한 글자를 추가 확정한다. 이 후보도 r6의 프로젝트 검증 확률비 하한 0.006115를 넘어야 한다.
- 괄호 가드: 이미 선택된 방향성 괄호는 잠근다. 기존 출력이 불균형일 때만 최대 64글자 안에서 누락된 짝을 최대 2글자 보충한다.
- 중위 연산자 가드: 양쪽 공간관계가 모두 `right`이고 이웃 역할이 값이며 적어도 한쪽이 숫자일 때만 본다. 중앙이 연산자가 아니고 Top-5에 `+·−·/·×·÷·⋅` 중 하나만 있으며 r6의 프로젝트 검증 확률비 하한 0.006115를 넘을 때 그 후보를 고른다. 세 위치가 모두 숫자이거나 양쪽 이웃이 모두 피연산자면 그대로 둔다.
- 앞의 두 가드는 beam 64다. 세 가드 모두 HWR Top-5 밖 후보, 삭제, grouping 변경은 금지한다.

| 범위 | 지표 | HWR | r6 | r6 + 의미 가드 | 의미 가드 추가 개선/회귀 |
|---|---|---:|---:|---:|---:|
| 전체 95식 | Top-1 | 70.03% | 86.30% | **88.63%** | 9/0 |
| 전체 95식 | 수식 exact | 32.63% | 60.00% | **69.47%** | 9식/0식 |
| 신규 48식 | Top-1 | 65.91% | 82.95% | **85.80%** | 5/0 |
| 신규 48식 | 수식 exact | 31.25% | 56.25% | **66.67%** | 5식/0식 |
| 신규 writer 44식 | Top-1 | 65.12% | 82.56% | **85.47%** | 5/0 |
| 신규 writer 44식 | 수식 exact | 25.00% | 52.27% | **63.64%** | 5식/0식 |
| CROHME 984식 | Top-1 | 69.50% | 77.82% | **78.09%** | 26/0 |
| CROHME 984식 | 수식 exact | 15.85% | 27.95% | **28.56%** | 6식/0식 |

전체 95식 strict micro는 79.17%→81.25%, strict macro는 77.19%→80.18%로 올랐다. CROHME strict micro는 63.15%→63.32%, strict macro는 52.01%→52.61%로 올랐다. 의미 가드가 r6 위에 새로 만든 회귀는 직접·CROHME 모두 0건이다.

산술 등식 v1은 직접 3글자를 복원했다. v2는 `ξ0÷5=6→30÷5=6`, `ℏ+0=5→5+0=5`, `1b÷8=2→16÷8=2`를 추가 복원해 기존 47식 +1, 신규 작성자 +2, CROHME 변경 0, 전 범위 회귀 0이었다. `0×q=0`은 `7`과 `3`이 모두 정확한 식을 만들어 유일하지 않으므로 그대로 둔다. 괄호 가드는 직접 `f ( y ) = 1`의 여는 괄호 1개와 CROHME 17글자를 복원했다. 중위 연산자 가드는 직접 `(b+4)`와 `n/3`의 2글자, CROHME 9글자를 추가 복원했고 회귀는 0건이었다.

추가 탈출구도 같은 고정 평가에서 검토했다. 두 개 이상의 비산술 슬롯을 한꺼번에 숫자로 바꾸는 규칙은 직접 수집을 하나도 고치지 못하고 반복 관찰한 CROHME 한 식만 고쳐 채택하지 않았다. 대소문자 일치 규칙은 `X≠y→x≠y`를 고쳤지만 정당한 대문자 변수와 `0·O·o`를 훼손할 수 있어 채택하지 않았다. `5/0→5-0`은 직접 한 글자를 고쳤지만 0으로 나누는 입력 자체도 사용자가 쓰려는 유효한 문자열이므로 의미를 강제하지 않았다. 문법 유효성만 강제한 기존 가설과 r6 결과를 다시 넣는 2회 추론도 각각 회귀 또는 효과 부족으로 기각했다.

실제 HWR Top-1 오인식을 프로젝트 truth별로 재표집해 합성 문맥에 넣은 노이즈 학습도 시험했다. 신규 48식의 제품 예측은 r6과 완전히 같고 CROHME Top-1은 77.82%→77.84%에 그쳤지만, 프로젝트 writer-LOO Top-1은 89.10%→88.15%, 수식 exact는 65.96%→61.70%로 내려가 체크포인트와 학습 코드 모두 채택하지 않았다. 실험 산출물은 로컬 `artifacts/owned_formula_context_20260819_noise_r1`에만 남겼다.

- 최종 평가: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\context-candidates-20260819-r3\r6_semantic_eval_r8.json`
- 평가 SHA-256: `56608469035a000a09818cc4847730f8f653cdd00c2e226ea6a230c4c8a83a40`
- 정답 없는 런타임 출력: `D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\context-candidates-20260819-r3\runtime-r6-semantic-r9.jsonl.gz`
- 런타임 출력 SHA-256: `686c7f9685757bd4878c132dc7c95db02563341655ff8a2970c871d8dfaca57e`
- 런타임·평가 예측 불일치 0건, CPU·GPU 예측 불일치 0건, 런타임 `label` 필드 0건

## r8 잔여 오답 탈출구 감사

최종 387글자 중 오답은 44개다. 이 가운데 25개는 정답이 HWR Top-5에 없어 후보 보존 문맥 레이어가 복구할 수 없다. 나머지 19개에는 길이 1의 고립 기호 2개와 `x/χ`, `b/h`, `x/X`처럼 수식 문법만으로 확정할 수 없는 같은 역할의 형상 동형이 남는다. 전체 Top-5 ceiling은 93.54%다.

- 완성 수식 경계의 `/·÷·×·⋅`를 숫자로 바꾸는 규칙은 기존 47식에서만 3개를 고쳤다. 세 후보 모두 r6 확률비 하한보다 낮고 신규·CROHME 적용 사례가 0개라 채택하지 않았다.
- `5 4 4→5+4` 규칙은 같은 슬롯 Top-5에 `+`와 `÷`가 함께 있어 문맥 후보가 유일하지 않으므로 채택하지 않았다.
- 길이 1을 HWR Top-1로 강제 복귀하면 직접 수집 정답 1개가 깨졌다. 예측은 유지하고 맥락 부재 상태만 노출한다.

따라서 등식 v2 이후 현재 후보 집합 안에서 직접·신규·CROHME 비회귀와 일반 수학 의미 보존을 함께 만족하는 추가 Top-1 규칙은 없다. 다음 정확도 상승은 HWR Top-5 밖 25개를 줄이는 형상 후보 개선과, `|·o`를 포함한 새 프로젝트 소유 untouched 수식 맥락이 필요하다.

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

출력 v4는 호환 `finalized_top1`에 더해 `context_available`, `decision_status`, `decision_source`를 제공한다. 수식 길이가 1이고 후보가 둘 이상이면 `ambiguous_no_formula_context`이며, 소비자는 이를 문맥 확정 결과로 취급하면 안 된다.

```powershell
$env:PYTHONNOUSERSITE='1'
python scripts\finalize_formula_context_v1.py `
  --checkpoint artifacts\owned_formula_context_20260822_r6\owned_formula_context_product.pt `
  --input D:\path\to\formula_candidates.jsonl.gz `
  --output D:\path\to\formula_finalized.jsonl.gz `
  --device cpu
```

387글자 전체에서 정답 필드를 제거한 뒤 실행한 결과, r8 최종 평가 예측과 불일치 0건, CPU·GPU 예측 및 결정 출처 불일치 0건, 후보 보존율 100%, 새 token 0, 삭제 0, grouping mutation 0을 확인했다. 374글자는 `finalized`, 맥락이 없는 단일 글자 13개는 `ambiguous_no_formula_context`다.

## 승격 조건

- 현재 선택: r6 + 산술 등식 가드 v2 + 잠금 괄호 가드 + 수평 중위 연산자 가드
- 현재 실행 상태: `opt-in shadow context finalizer`
- `commercial_context_training_rights_gate_passed=true`
- `commercial_accuracy_gate_passed=false`
- `automatic_default_replacement=false`

다음 판정에는 collector가 새로 요구하도록 배포된 `|`와 `o` 프로젝트 소유 수식을 실제로 수집한 뒤, 지금 체크포인트와 정책을 고정한 untouched 평가가 필요하다. 후보 보존 100%, 각 동형군 비회귀, 전체 Top-1 비회귀, 수식 exact 비회귀를 동시에 통과하기 전에는 기본 경로로 승격하지 않는다.
