# 원시 타점 식 분할·문맥 배치 루프

## 결론

- 원시 타점의 자동 문자 분할은 기존 0.9 기하 기준 `68/96`(70.83%)에서 새 writer 완전분리 평가 `90/96`(93.75%)로 상승했다.
- 시간순으로 앞 47식을 학습하고 뒤 49식을 평가한 경로에서는 분할 완전일치가 `45/49`에서 `49/49`(100%)로 상승했고, 기존 정분할 식의 회귀는 0건이었다.
- 같은 49식의 식 완전일치는 raw HWR `15/49`(30.61%), masked context `29/49`(59.18%), 기존 식 배치·guard 최종 `38/49`(77.55%), 관계기호 병합과 후보 보존형 문맥 융합 `41/49`(83.67%), 두 단계 숫자 문맥 gate 포함 최종 `43/49`(87.76%)였다.
- 관계기호 병합은 `aiflow_0010`의 분리된 `=, 1` stroke를 HWR Top-5의 `\neq`로 복원했다. 기존 문맥 융합과 숫자 gate의 개선까지 합쳐 5식이 복원됐고 기존 정답 회귀는 0건이었다.
- 선택 분할 HWR Top-5 식 oracle은 `42/49`, 기존·보조 Top-5 합집합 oracle은 `47/49`, 제한형 Top-10 합집합 oracle은 `48/49`(97.96%)다. 남은 병목은 Top-10 후보 누락 1식과 후보 안 선택 실패 5식이다.
- 이 결과는 개발 shadow다. 49식 오류를 본 뒤 보수 gate를 정했고 보조 HWR head가 현재 96식과 겹친다. 새 collector 데이터는 `Listed=164, Downloaded=0, Total=164`로 없었으며 제품 기본값은 꺼져 있다.

## 고정 구조

```text
48 Hz 온라인 타점
  -> label-free 타점 그룹 격자
  -> 기하 그룹 selector
  -> 상위 5개 exact-cover 분할
  -> 각 그룹 372-class HWR Top-5
  -> masked-context 분할 ranker
  -> 병합 전용 보수 gate
  -> 겹친 관계기호 본체+슬래시를 Top-3 exact-cover 안에서만 재결합
  -> old/new HWR 0.4:0.6 융합으로 기존 Top-5 순서만 재배치
  -> 2D 식 배치 관계
  -> masked-context 문자 확정
  -> semantic / sequence / syntax guard
  -> 단일식 `/`, `\times` 후보 보존형 shape rescue
  -> 기존·보조 Top-5 합집합 안의 숫자 피연산자 문맥 rescue
  -> 깨진 숫자 문법의 유일한 최소 복원만 Top-10에서 선택
  -> 후보 보존 식 출력
```

고정 계약은 다음과 같다.

- 모든 원시 stroke를 정확히 한 번 사용한다.
- 정답 문자, writer ID, 목표 문자 수를 추론 입력으로 받지 않는다.
- 일반 문맥 확정은 기존 HWR Top-5를 벗어나지 않는다.
- 숫자 rescue만 기존·보조 HWR Top-5 합집합에서 최대 2개 피연산자를 숫자로 바꿀 수 있다.
- Top-10 rescue도 숫자 피연산자만 바꾸며 유일한 최소 복원이 아니면 보류한다.
- Top-10 rescue의 `=` 교정 경로는 꺼져 있다.
- Top-5 숫자 gate는 연산자·관계기호·괄호를 바꾸지 않는다. Top-10 gate는 오인식된 슬롯을 숫자로만 바꾸고 `=` 교정은 하지 않는다.
- 문자 삽입·삭제·재배치와 산술 정답 계산을 하지 않는다.
- 분할 ranker는 상위 5개 exact-cover 밖으로 임의 regrouping하지 않는다.
- 관계기호 재결합은 rank 3 이내, 정확히 3 stroke, 본체·슬래시 박스 겹침, 병합 HWR Top-5의 부정 관계기호, 좌우 피연산자 조건을 모두 만족하는 유일 후보만 허용한다.
- 현재 posthoc gate는 명시적 `--allow-posthoc-shadow` 없이는 실행되지 않는다.

## 평가 결과

### 타점 그룹핑

| 실험 | 식 분할 exact | 판단 |
|---|---:|---|
| 0.9 기하 기준 | 68/96 (70.83%) | 교체 |
| 새 기하 selector, nested writer-LOO | 90/96 (93.75%) | 선택 |
| 폭 0.65 hard gate | 89/96 (92.71%) | 기각 |
| 개별 HWR confidence 결합 | 87/96 (90.63%) | 기각 |

새 격자는 truth group `393/393`, truth partition `96/96`을 포함한다. 따라서 현재 6식 오류는 격자 누락이 아니라 selector 순위 오류다.

### 시간순 47→49 원시 입력

| 단계 | 식 exact |
|---|---:|
| 기하 Top-1 분할 | 45/49 (91.84%) |
| 문맥 분할 보수 gate | 48/49 (97.96%) |
| 관계기호 exact-cover 재결합 | **49/49 (100%)** |
| raw HWR 식 | 15/49 (30.61%) |
| masked context 식 | 29/49 (59.18%) |
| 식 배치·guard 최종 | **38/49 (77.55%)** |
| 관계기호 병합+후보 보존형 문맥 융합 최종 | **41/49 (83.67%)** |
| 숫자 문맥 gate 포함 최종 | **42/49 (85.71%)** |
| 제한형 Top-10 숫자 문법 gate 포함 최종 | **43/49 (87.76%)** |
| 선택 분할 Top-5 식 oracle | 42/49 (85.71%) |
| 기존·보조 Top-5 합집합 식 oracle | 47/49 (95.92%) |
| 제한형 Top-10 합집합 식 oracle | 48/49 (97.96%) |

분할 gate는 `aiflow_0003`, `aiflow_0007`, `aiflow_0008`을 rank 2로 복원했다. 추가 관계기호 gate는 `aiflow_0010`만 rank 3으로 바꿔 `x \neq y`를 복원했고 회귀는 없었다.

`43/49`는 원시 stroke부터 자동 분할을 포함한 수치다. 기존 `85/95`는 검수된 glyph ownership을 입력으로 준 조건부 수치이므로 직접 비교하면 안 된다.

실제 raw runtime으로 유효 원시식 159개를 다시 실행했다. 관계기호 병합까지 합치면 시간순 뒤 49식은 `42→43`, ownership 96식은 `87→88`, 전체 prompt 159식은 `129→130`으로 올랐고 회귀는 0건이었다. r9 대비 변경은 `aiflow_0010` 한 건뿐이다. 159식 모두 stroke exact-cover, 정답·writer·목표 글자 수 입력 없음, 삽입·삭제 0, 산술 계산 없음, 제품 기본값 OFF를 확인했다. 최종 출력 SHA-256은 `b2139fdb0a749811ea4d4587d87348a87660082eb17699bb5137fb73a712bc07`이다.

## 남은 병목

- 제한형 Top-10 합집합 누락 1식: `aiflow_0064`.
- 합집합 안 선택 실패 5식: `aiflow_0004`, `aiflow_0046`, `aiflow_0056`, `aiflow_0061`, `aiflow_0072`.
- 이 중 `aiflow_0072`의 단일 `\times↔x`는 식 문맥이 없어 현재 구조만으로 확정할 수 없다.
- 현재 49식은 반복 검토된 개발 집합이며 관계기호 gate도 남은 오류를 본 뒤 선택했다. 따라서 이 수치는 상용 acceptance가 아니다.
- Top-20 old/new product 융합은 oracle을 `47/49`까지 올렸지만 승인된 보수 규칙의 실제 개선은 0식이었다. 현재 런타임에는 넣지 않았다.
- 새 교정 writer-LOO HWR head는 최종 `35/49`, Top-5 oracle `40/49`로 선택본보다 낮아 기각했다.

따라서 다음 정확도 향상은 epoch 추가보다 다음 순서가 타당하다.

1. 새 writer·새 식의 상업 이용 가능 원시 ink를 untouched acceptance로 확보한다.
2. `/·1`, `x·\times`, `S·5`, `\neq` 양성 쌍을 writer/formula-disjoint로 수집한다.
3. Top-20 전체를 무조건 확정하지 말고 후보 회수용 보조 학습과 abstention gate를 새 acceptance에서 선택한다.

## 선택 산출물

- 그룹핑 구현: `scripts/stroke_grouping_v1.py`
- 그룹핑 학습·nested writer-LOO: `scripts/train_project_owned_grouping_v1.py`
- HWR 그룹핑 기각 실험: `scripts/evaluate_joint_hwr_grouping_v1.py`
- 시간순 분할·문맥 평가: `scripts/evaluate_partition_context_ranker_v1.py`
- 원시 타점 shadow 런타임: `scripts/raw_formula_context_runtime_v1.py`
- 후보 보존형 문맥 융합 설정: `artifacts/raw_candidate_context_fusion_20260819_r1_shadow/candidate_context_fusion_runtime_config.json`
- 후보 보존형 문맥 융합 결과: `artifacts/raw_candidate_context_fusion_20260819_r1_shadow/selected_summary.json`
- 그룹핑 모델 SHA-256: `4a6b3fa461ca15ab5872f9648541904c04f032fe096d6185ab6c501b2ca02ebd`
- 분할·문맥 모델 SHA-256: `0d283d40b40f60d58131dec213efb84074f9119ac82b32fcb893ca931aeb0582`
- 후보 보존형 문맥 융합 설정 SHA-256: `7ea7b11a2fa7d3a4eaa7dc1bb215803c46bc263689e7b856026e4b45fa504f84`
- HWR checkpoint SHA-256: `1fed74b4580d3acc9ff48544a7d715c52170037bcb0842921b1fe0339f0abd1d`
- context checkpoint SHA-256: `7e0ff1b04ba1071867bfa6811869f242907ebe017d0996597776c6050cc2fc94`

## 실행 예

```powershell
$env:PYTHONNOUSERSITE='1'
python scripts\raw_formula_context_runtime_v1.py `
  --partition-ranker artifacts\partition_context_ranker_20260819_r12_selected_shadow\partition_context_ranker.joblib `
  --hwr-checkpoint artifacts\unified_head_20260814\uniform_time_final_all_writers\project_symbol_head_checkpoint.pt `
  --context-checkpoint artifacts\owned_formula_context_20260822_r6\owned_formula_context_product.pt `
  --formula-sequence-config artifacts\formula_sequence_guard_20260819_r6_runtime\formula_sequence_guard_runtime_config.json `
  --formula-syntax-rescue-config artifacts\formula_syntax_rescue_20260819_r2_shadow\formula_syntax_rescue_runtime_config.json `
  --candidate-context-fusion-config artifacts\raw_candidate_context_fusion_20260819_r1_shadow\candidate_context_fusion_runtime_config.json `
  --candidate-context-auxiliary-hwr-checkpoint D:\AIFlow-Workspace\PrivateData\math-ink-data-collector\derived\hwr-head-calibration-20260819-r1\final_all_writers_steps250_lr1e-3\project_symbol_head_checkpoint.pt `
  --input D:\AIFlow-Workspace\PrivateData\runtime-input.json `
  --output D:\AIFlow-Workspace\PrivateData\runtime-output.json `
  --allow-posthoc-shadow
```

승격 조건은 새로운 상업 이용 가능 project-owned writer/formula-disjoint 데이터에서 분할·문자·식 exact와 동형 문자 비회귀를 모두 다시 통과하는 것이다.

## 2026-08-20 교차 획·공식 배치 개선 결과

이 절은 위의 `130/159` 결과를 대체하는 후속 shadow 결과다. 두 개의 단일 획이 `x` 후보로 합쳐지는 경우만 허용하는 exact-cover 교차 획 병합과, 기존 Top-20 후보 안에서만 작동하는 공식 배치 보정을 런타임에 연결했다. 공식 배치 단계는 앞선 숫자 문맥 보정 결과를 입력으로 받도록 합성 순서도 바로잡았다.

| 평가 집합 | r11 | 이번 결과 |
|---|---:|---:|
| 시간순 ownership 49식 | 43/49 (87.76%) | 43/49 (87.76%) |
| accepted ownership 96식 | 88/96 (91.67%) | **89/96 (92.71%)** |
| writer_012 제외 public 110식 | 102/110 (92.73%) | **103/110 (93.64%)** |
| writer_012 휴대폰 재생 49식 | 28/49 (57.14%) | **35/49 (71.43%)** |
| 전체 유효 159식 | 130/159 (81.76%) | **138/159 (86.79%)** |

- 개선 8식: `aiflow_0092`, `aiflow_0093`, `aiflow_0097`, `aiflow_0098`, `aiflow_0102`, `aiflow_0103`, `aiflow_0112`, `aiflow_0142`
- 회귀: 0식
- 단계별 변경: 관계기호 병합 1식, `x` 교차 획 병합 11식, 병합 `x` 후보 잠금 1글자, 공식 배치 1식, 직선형 등호 0식
- 계약 검증: 모든 획 정확히 1회 사용, 정답·writer·목표 글자 수 입력 없음, 후보 밖 문자 생성 없음, 문자 삽입·삭제 없음, 산술 계산 없음, 제품 기본값 OFF
- 최종 159식 출력 SHA-256: `ac4325a05d445f4d28578de0b118ca5488f4a6aadeb102d688142bfe2a3c79a6`

현재 결과는 writer_012 및 `aiflow_0142`를 관찰해 gate를 정한 posthoc shadow 결과다. 상용 승격 근거로 쓰려면 새 writer/formula-disjoint acceptance가 필요하다.

### 후속 실행 인자

기존 실행 예에 다음 두 설정을 함께 전달한다.

```powershell
  --formula-placement-config artifacts\formula_placement_rescue_20260819_r1_shadow\formula_placement_rescue_runtime_config.json `
  --straight-equality-config artifacts\straight_equality_slot_rescue_20260819_r1_shadow\straight_equality_slot_rescue_runtime_config.json `
```

## 2026-08-20 공식 역할·오탈자 보정 루프

이 절은 바로 위의 `138/159` 결과를 대체한다. 기존 masked-attention 문맥 모델의 출력을 그대로 확정하지 않고, product-fused HWR Top-20 안에서 식 역할과 타점 기하가 동시에 맞을 때만 다시 고르는 `formula_role_typo_rescue_v1` 단계를 공식 배치 레이어에 추가했다. 새 문자를 만들거나 순서를 바꾸는 신경망이 아니라, 이미 학습된 HWR·문맥 후보를 보수적으로 재판정하는 오탈자 보정층이다.

허용한 경우는 다음뿐이다.

- `f`, `g`, `h` 후보가 뒤의 `(`와 함께 함수호출 머리를 이루는 경우
- `(`와 `)` 사이 단일 `x`, `y`, `z` 후보
- 이항연산자 앞에서 fused HWR Top-1이 숫자인 경우의 문맥 오버라이드 철회
- 수직 1획 `1`, 2획 십자형 `+`, 두 후보가 붙어 있는 `+1` 쌍
- 균형 잡힌 함수호출 좌변 뒤 `= 변수`의 fused 알파벳 Top-1
- 숫자 양옆의 넓고 수평인 2획 `=` 후보

식의 계산 결과는 사용하지 않았다. 예를 들어 `30 ÷ 6 = 5`에서 가운데 글자를 `0`으로 고르는 산술 역산은 금지되어 그대로 남겼다.

| 평가 집합 | 이전 결과 | 공식 역할 보정 후 |
|---|---:|---:|
| 시간순 ownership 49식 | 43/49 (87.76%) | **46/49 (93.88%)** |
| accepted ownership 96식 | 89/96 (92.71%) | **92/96 (95.83%)** |
| writer_012 제외 public 110식 | 103/110 (93.64%) | **106/110 (96.36%)** |
| writer_012 휴대폰 재생 49식 | 35/49 (71.43%) | **41/49 (83.67%)** |
| 전체 유효 159식 | 138/159 (86.79%) | **147/159 (92.45%)** |

- 개선 9식: `aiflow_0004`, `aiflow_0056`, `aiflow_0061`, `aiflow_0095`, `aiflow_0100`, `aiflow_0105`, `aiflow_0110`, `aiflow_0111`, `aiflow_0116`
- 회귀: 0식
- 공식 역할 보정: 8식·12글자, 직선형 등호 보정: 1식·1글자
- 계약: 모든 획 정확히 1회 사용, 정답·writer·목표 글자 수 입력 없음, Top-20 밖 문자 생성 없음, 삽입·삭제·재배치 없음, 산술 계산 없음, 제품 기본값 OFF
- 시간순 49식 evaluator SHA-256: `58b64880775040bc1379ce853288fdebf490813f72c68bf30b1ad1d789a6d33b`
- 최종 159식 출력 SHA-256: `37e2829c7134f758d16d1752ae9754088c2d5058127c09e562c5fe8253ff22e3`

CROHME 반복 비상업 진단 984식에서는 문자 Top-1이 `7516/9535 → 7555/9535`, 식 exact가 `288/984 → 295/984`로 증가했고 문자·식 회귀는 모두 0건이었다. 이 결과는 제품 검증이나 상업 승격 근거가 아니다.

남은 12식은 그룹 수 불일치 3식(`aiflow_0038`, `aiflow_0101`, `aiflow_0109`), 현재 Top-20 정답 후보 부재 3식(`aiflow_0036`, `aiflow_0064`, `aiflow_0104`), 문맥이 없는 단일 문자 4식(`aiflow_0032`, `aiflow_0044`, `aiflow_0046`, `aiflow_0072`), 산술 역산 없이는 안전하게 못 고르는 숫자 혼동 2식(`aiflow_0023`, `aiflow_0113`)이다.

이 결과도 현재 159식을 관찰해 임계값을 정한 posthoc shadow다. 상용 승격 조건은 새 writer/formula-disjoint acceptance에서 동일 개선과 비회귀를 재현하는 것이다.
