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

## 2026-08-20 반복형 그룹·단일 등호·라틴 t 문맥 보정

바로 위 `147/159` 결과에서 안전하게 근거를 만들 수 있었던 세 오류만 추가로 처리했다.

- 반복형 exact-cover: `값 = 값` 세 슬롯에서 양끝 문자가 같고, 한쪽만 두 단일 획으로 갈라졌으며, HWR·문맥 토큰이 일치하고 정규화 DTW 거리가 0.1 이하일 때만 병합한다. `aiflow_0101`의 `7 = 7` 한 식을 복구했다.
- 단일 등호: 한 글자 식이라도 두 구성 획이 수평·평행·정렬 조건을 모두 만족할 때만 `=` 후보를 허용했다. `aiflow_0044` 한 식을 복구했다.
- 라틴 `t` 보조 헤드: 기존 372클래스에 빠진 소문자 `t`를 승인된 95클래스 레거시 보조 헤드에서 가져오되, `f/g/h ( 단일 인자 )` 슬롯에서만 Top-2·확률 0.25·Top-1 대비 비율 0.5를 모두 넘을 때 교체한다. `aiflow_0064`의 `f(t)+2` 한 식을 복구했다.

| 평가 집합 | 공식 역할 보정 후 | 이번 결과 |
|---|---:|---:|
| 시간순 ownership 49식 | 46/49 (93.88%) | **47/49 (95.92%)** |
| accepted ownership 96식 | 92/96 (95.83%) | **93/96 (96.88%)** |
| writer_012 제외 public 110식 | 106/110 (96.36%) | **107/110 (97.27%)** |
| writer_012 휴대폰 재생 49식 | 41/49 (83.67%) | **43/49 (87.76%)** |
| 전체 유효 159식 | 147/159 (92.45%) | **150/159 (94.34%)** |

- 개선: `aiflow_0044`, `aiflow_0064`, `aiflow_0101`
- 회귀: 0식
- r22와 r23 최종 토큰 차이: 0식. r23은 갱신된 설정·증거 해시를 다시 검증한 재현 실행이다.
- 시간순 49식 evaluator SHA-256: `a34d196aee0a9ca42103c8793ad059ac5bfcb4d94efb574a73e0f041070442e9`
- 최종 159식 출력 SHA-256: `7294fb873b50698659ef77cdeacbbce0eba011f947dc965cd0b16479494a2edf`

CROHME 원시 궤적은 중복 제거 전 985식 중 지원 라벨·완전 그룹·2점 이상 획 조건을 통과한 722식을 재생했다. 새 세 단계는 모두 0회 발동했고 그룹 exact `331→331`, 식 exact `77→77`, 회귀 0건이었다. 진단 SHA-256은 `7a7af2d9c2dfa02a46e0435fbc879d30f9a437ecc2d60f9af2485835299f67cf`이다. CROHME는 반복 비상업 진단일 뿐 제품 검증이 아니다.

남은 오류는 9식이다. `aiflow_0023`, `aiflow_0113`은 산술 역산 없이 확정하기 어려운 숫자 혼동이고, `aiflow_0032`, `aiflow_0036`, `aiflow_0046`, `aiflow_0072`는 단일 글자 또는 문맥 부족 동형 문자다. `aiflow_0038`, `aiflow_0104`, `aiflow_0109`는 현재 안전 임계값과 후보·그룹 계약 안에서 복구할 수 없다. 이 9식을 억지 교정하는 규칙은 추가하지 않는다.

모든 결과는 현재 159식을 관찰해 정한 posthoc shadow이며 제품 기본값은 계속 OFF다. 다음 승격 판단에는 새 상업 이용 가능 project-owned writer/formula-disjoint acceptance가 필요하다.

## 2026-08-20 중첩식 문맥·배치 보정 루프

바로 위 r23의 `150/159`를 고정 기준선으로 두고, 남은 오류 중 문맥과 배치 증거가 모두 있는 `aiflow_0109`만 좁게 복구했다. 새 신경망이 문자를 생성하거나 식을 다시 쓰는 방식은 사용하지 않았다. 기존 r12 분할 선택기, HWR 두 헤드, 소유 수식 문맥 모델의 후보를 그대로 두고 다음 조건을 전부 만족할 때만 exact-cover 분할과 문자 후보를 재선택한다.

- 바깥 괄호를 포함한 중첩 flat 식 전체가 유효할 것
- 두 단일 획을 합친 `x`가 product-fused Top-3 안에 있을 것
- 바로 다음 2획 교차 문자가 `+` Top-4와 기하 조건을 만족할 것
- 그 다음 문자가 여는 괄호 Top-1일 것
- 모든 획을 정확히 한 번 사용하고, 후보 밖 문자·삽입·삭제·순서 변경이 없을 것

`aiflow_0109`는 분할 rank 3에서 `x` rank 3, `+` rank 3, 여는 괄호 rank 1 증거가 동시에 성립했다. 최종 토큰은 `( 7 \subset 4 ( y - 1 ) )`에서 정답 `( x + ( y - 1 ) )`로 바뀌었다.

| 평가 집합 | r23 | r25 중첩식 보정 |
|---|---:|---:|
| 시간순 ownership 49식 | 47/49 (95.92%) | 47/49 (95.92%) |
| accepted ownership 96식 | 93/96 (96.88%) | 93/96 (96.88%) |
| writer_012 제외 public 110식 | 107/110 (97.27%) | 107/110 (97.27%) |
| writer_012 휴대폰 재생 49식 | 43/49 (87.76%) | **44/49 (89.80%)** |
| 전체 유효 159식 | 150/159 (94.34%) | **151/159 (94.97%)** |

- 개선: `aiflow_0109` 1식
- 회귀: 0식
- 최종 토큰이 바뀐 식: 위 1식뿐
- 계약 검사: 획 누락·중복 0, 문자 삽입·삭제 0, 정답·writer·목표 글자 수 입력 0, 산술 계산 0
- 동형 문자 비회귀: 기존 정답 출력은 하나도 바뀌지 않았고, 바뀐 `x`와 `+`는 모두 해당 식의 정답이었다.

CROHME 원시 궤적 722식에서는 이전 r23 동작을 기준선으로 새 중첩 병합과 `값 + (` 보정만 끄고 켜서 비교했다. 중첩 병합은 0식, `+` 보정은 2식에서 발동했다. 그룹 exact는 `331→331`, 식 exact는 `77→77`, 그룹·식 회귀는 모두 0건이었다. 두 식은 전체 오답 상태가 유지되어 제품 양성 근거로 세지 않았다. CROHME는 반복 비상업 진단이며 상업 승격 근거가 아니다.

넓은 모델 변경은 채택하지 않았다.

- 기존 BERT-Tiny masked-context 모델은 필요한 `x` 후보를 낮게 평가했고 과거 strict homograph macro도 퇴행했으므로 런타임에 연결하지 않았다.
- 함수 DSL을 확장해 재학습한 소유 문맥 r7은 writer-LOO Top-1 `89.10%→88.15%`, CROHME 식 exact `27.95%→26.22%`, CROHME 회귀 `280→385`로 악화되어 기각했다.
- 매 실행 CUDA 특징을 다시 계산한 r37 분할 재학습은 새 규칙이 0회 발동한 49식에서 관련 없는 3개 분할이 달라졌다. 쌍대 비교에는 재학습 모델을 쓰지 않고 고정 r12 SHA-256 `0d283d40b40f60d58131dec213efb84074f9119ac82b32fcb893ca931aeb0582`만 사용했다.

재현 산출물은 다음과 같다.

- 중첩식 fusion 설정 SHA-256: `341a71e66beeb8fbac36a96ba930be7d2a8a7b98732a19a8250c7a8e6c418606`
- 공식 배치 설정 SHA-256: `fd5d4ccb6ec138aa8506aaafdb68b12d26b3acee1114d2006bd109d52b4e3d15`
- CROHME 진단 SHA-256: `6184427b35c13a416e8290507ddc51893d760db195789f58fd907d13f9b0bc2f`
- 최종 159식 출력 SHA-256: `5217c07a24ac907074f469aeee5ba74e34bbd95dc40d59738cad992d61d173f1`

남은 오류는 8식이다. 문맥 없는 단일 글자·동형 문자 `aiflow_0032`, `aiflow_0036`, `aiflow_0046`, `aiflow_0072`, 안전한 그룹·후보 근거가 부족한 `aiflow_0038`, `aiflow_0104`, 산술 역산 없이는 확정하기 어려운 `aiflow_0023`, `aiflow_0113`이다. 이번 수치는 현재 159식을 관찰한 posthoc shadow 결과이므로 제품 기본값은 계속 OFF다.

## 2026-08-20 rank-2 단일 `×` 후보 보정 루프

r26의 단일식 38개를 다시 감사했다. `aiflow_0072`는 주 HWR Top-5 안에 `\times`가 있고, old/new product-fused 분포에서도 `X` 다음 rank 2·확률 0.402473이었다. 기존 단일기호 gate는 fused Top-1만 보아 이 후보를 버리고 있었다.

단일 글자 식에 한해 `/`는 기존 rank 1 조건을 유지하고, `\times`만 fused rank 2·확률 0.40 이상까지 허용했다. fused 후보에 있더라도 주 HWR 원본 Top-5에 없으면 선택하지 않는다. 0.20부터 0.45까지 임계값을 훑은 결과 0.40에서는 38식 중 `aiflow_0072`만 바뀌었고 기존 정답 회귀는 0건이었다.

| 평가 집합 | r26 | rank-2 단일기호 보정 |
|---|---:|---:|
| 시간순 ownership 49식 | 47/49 (95.92%) | **48/49 (97.96%)** |
| accepted ownership 96식 | 93/96 (96.88%) | **94/96 (97.92%)** |
| writer_012 제외 public 110식 | 107/110 (97.27%) | **108/110 (98.18%)** |
| writer_012 휴대폰 재생 49식 | 44/49 (89.80%) | 44/49 (89.80%) |
| 전체 유효 159식 | 151/159 (94.97%) | **152/159 (95.60%)** |

- 개선: `aiflow_0072`의 `x → \times` 1식
- 회귀·그 외 최종 토큰 변경: 0식
- 획 누락·중복, 문자 삽입·삭제·순서 변경, 정답·writer·목표 글자 수 입력, 산술 계산: 모두 0
- CROHME 원시 722식: 새 단일기호 gate 발동 0식, 그룹 exact `331→331`, 식 exact `77→77`, 회귀 0식

후속 `aiflow_0104` probe에서는 `(`가 fused rank 6, `y`가 rank 21이었다. 두 후보를 역할 기반 Top-5에 넣으면 기존 문맥 신경망은 `L→(`는 고쳤지만 `q`를 계속 선택해 식 exact는 오르지 않았다. `f(x)+g(y)`를 규칙으로 강제하면 정상적인 `f(x)+g(x)`까지 바꿀 수 있으므로 채택하지 않았다. 다음 개선에는 새 writer의 `y/q` 온라인 궤적과 두 형태의 함수식 양성·음성 데이터로 형상·문맥 모델을 다시 학습해야 한다.

재현 해시는 다음과 같다.

- rank-2 singleton fusion 설정: `312c9f2bd278319d6232f4d0d3d901c1ecfe014140047232c35aca2703e8f963`
- CROHME 진단: `767164f7778ccbd0a2f60802dfb5163ce76830d3a44646a979a924f0f272e5f8`
- 수정 평가기 재실행: `584a71fe13bf133ae13bed84c215820491979cfe7e37ca19b35a64d2c21f61ca` (생성 시각 제외 의미 내용 일치)
- 최종 159식 출력: `ea095520d0671d6ddcdf41f134af9a6d4e1093101b408ec7e8816acfa819f3d6`

남은 오류는 7식이다. 단일식 `aiflow_0032`, `aiflow_0036`, `aiflow_0046`은 정답 rank가 각각 20, 98, 8이고 문맥이 없다. `aiflow_0038`은 두 획 병합 `x`가 rank 8이지만 `x/\times` 의미 문맥이 없다. `aiflow_0104`는 위의 `y/q` 재학습이 필요하다. `aiflow_0023`, `aiflow_0113`은 산술 역산 없이 각각 `0/6`, `8/p`를 확정할 근거가 부족하다. 제품 기본값은 계속 OFF다.

## 2026-08-20 잔여 7식 탈출구 감사

rank-2 단일기호 루프 뒤 남은 7식을 원시 획으로 다시 그려 확인했다. 7건 모두 획 누락이나 확정 가능한 오라벨로 볼 근거가 없어 평가에서 삭제하지 않았다. 시각 감사 파일은 `D:/AIFlow-Workspace/PrivateData/remaining-errors-r27-raw-strokes.png`, SHA-256은 `7975589f999b01a7a73070e93342b1b7b5bdd425840bcf9663afd97c98d8ca3e`이다.

추가 네 가설은 다음 이유로 채택하지 않았다.

- 외부 167,088글자의 128차원 HWR 임베딩 클래스 중심은 `0/6`, `8/p`, `5/π`, `y/q`를 무회귀로 분리하지 못했다.
- 폐곡선 역방향·순환 시프트 앙상블은 충돌 제거 외부 18,330글자 중 1,735개 폐곡선에서 개선과 회귀가 동시에 생겼다. `p→8`만 좁힌 외부 5건의 실제 정답은 `P` 3, `\rho` 1, `p` 1이고 `8`은 0건이었다.
- 기존 BERT-Tiny masked-attention은 `aiflow_0104`의 `y`를 raw 문맥 rank 74, 여는 괄호 보정 뒤 rank 78로 평가했다. HWR와 결합 강도를 높이면 `y`보다 `r`이 먼저 승격되어 함수 인자 gate로도 채택할 수 없었다.
- 두 획 `5` 기하와 승인된 Latin 보조 헤드를 결합하면 현재 159식의 후보 22개는 모두 실제 `5`였다. 그러나 외부의 넓은 동일 조건 27개에는 `5` 10개 외에 `6`, `J`, `\Gamma`, `\Sigma`, `\hbar`, `\mathcal{T}`, `\sum`이 섞였다. `aiflow_0046`만 잡는 극단 조건은 외부 적용 표본이 0개라 일반화 근거가 없었다.

따라서 이번 루프는 변경 0, 회귀 0으로 종료했고 공식 exact는 `152/159 (95.60%)`를 유지한다. 현재 159식에만 맞춘 규칙으로 `153/159`를 만들지는 않는다. 다음 실질 개선에는 새 writer의 분리 윗가로획 `5`, 폐곡선 `0/8`, 필기체 `y/q`, 단독 `x/\times` 양성·음성 궤적이 필요하며 제품 기본값은 계속 OFF다.

기계 판독 감사 증거 SHA-256: `e630bebfe4fd399370b42c1938e51e4b7bebbb00086d4e426a1d3bdd47a4bcd3`.

## 2026-08-20 외부 임베딩 `8/p` 보정기 shadow 루프

위 감사에서 기각한 것은 클래스 중심·기하 규칙이었다. 후속 루프에서는 기존 승인 외부 train의 `8` 325개와 `p` 389개만 사용해 frozen production HWR encoder의 128차원 임베딩 위에 class-balanced 이진 logistic expert를 학습했다. 식 정답, writer, 식 길이, 산술 결과는 학습·추론 입력에 넣지 않았다.

충돌 제거 외부 holdout의 `8/p` 78개에서 production HWR의 두 클래스 간 판정은 78/78, expert 단독 판정은 77/78이었다. 유일한 `p→8` 오분류 margin은 `0.082169`였다. 따라서 최종 `p`, expert의 `8` margin 0.10 이상, 원본 production HWR Top-20 안의 `8`, `P(8)/P(p) ≥ 0.05`를 모두 요구했다. 이 게이트를 외부 18,330개 전체에 적용하면 변경·개선·회귀가 모두 0개였다.

159식 raw runtime에서는 기존 문맥·배치·교차획 lock 뒤에 이 레이어를 연결했다. 원본 HWR 후보 밖 문자를 만들지 않으며 grouping, 문자 수, 순서를 바꾸지 않는다.

| 평가 집합 | r27 | `8/p` expert shadow |
|---|---:|---:|
| public web 110식 | 108/110 (98.18%) | 108/110 (98.18%) |
| owned phone replay 49식 | 44/49 (89.80%) | **45/49 (91.84%)** |
| 전체 유효 159식 | 152/159 (95.60%) | **153/159 (96.23%)** |

- 변경·개선: `aiflow_0113` 한 식의 첫 토큰 `p→8`
- expert margin: `0.583145`
- 원본 production HWR `8` rank: 13
- 원본 HWR `P(8)/P(p)`: `0.058589`
- 회귀 및 그 외 최종 토큰 변경: 0식
- 획 exact-cover, 삽입·삭제·순서 변경, 정답·writer·목표 글자 수 입력, 산술 계산: 모두 기존 안전 계약 유지
- CROHME 원시 722식: 보정 발동 0식, grouping exact `331→331`, formula exact `77→77`, 회귀 0식
- expert/config 인자를 생략한 기본값 OFF 재실행: r27 대비 159식 최종 토큰 차이 0, grouping 차이 0

이 결과는 `153/159`를 제품 정확도로 승격하지 않는다. 임계값이 현재 159식을 관찰해 정해졌고 외부 게이트의 실제 양성 적용 표본이 0개이므로, `retain_as_shadow_only_not_product`로 고정했다. 상용 승격에는 새 writer/formula-disjoint 상업 이용 가능 `8/p` 양성·음성 acceptance가 필요하다.

재현 해시는 다음과 같다.

- expert JSON: `0f07993e4ccc71630290e1516fa132d35ad9e754ffffce88426b399ee7c21061`
- gate 설정: `a67452835709baa2a22f3a959b419ff203fa626c2bdb2d451828cc2f3a098a45`
- 외부 학습·평가 보고서: `f571b4ede20b56ac1750a84e0e424ec59554bdde774a5ff8a61be039de0aded4`
- 159식 runtime 출력: `601239ffabff9b2ad2616c1157a04bd20382306f2fb8ba784e44188bff6943b0`
- CROHME 반복 진단: `a21e150e841592ea2f0abab8c756a232445f007bdc98159ce0cda16c8e3bd167`
- 기본값 OFF 호환 출력: `57203d01337b6dd54981bcecdee530838c85a451729946f4fffd3105577d672e`
- 통합 runtime 감사 manifest: `f62d12b9896f045e117f1ead295475a882b2f76cebd77ceeb92fde8947d9b304`

## 2026-08-20 식 단위 선택적 수락 경계 shadow 루프

남은 6식을 직접 바꾸는 추가 가설을 먼저 재감사했다. `aiflow_0038`의 두 획 병합 후보는 `\mathfrak{X}`가 1위이고 평문 `x`는 8위였으며, 외부 14-class 임베딩 분류기들도 모두 `\mathfrak{X}`를 선택했다. `aiflow_0032`의 폐곡선은 random forest에서는 `0` 우세였지만 가장 가까운 외부 3개가 모두 `\circlearrowright`여서 교정기를 기각했다. `aiflow_0104`의 `g(q)`는 문법적으로도 유효하므로 정답식 패턴 없이 `y`로 강제할 수 없었다.

따라서 토큰을 더 바꾸지 않고, 기존 HWR·BERT-Tiny 문맥·배치 레이어가 남긴 충돌 증거를 식 단위 `REVIEW_REQUIRED`로 전달하는 경계를 추가했다. 다음 다섯 조건만 사용한다.

- 문맥 교정이 단일문자식의 HWR Top-1을 바꾼 경우
- 문맥 교정된 숫자가 인접 숫자와 다자리 수를 만든 경우
- 두 글자 이상 식의 방향 괄호가 닫히지 않은 경우
- 문맥 없는 단일문자식이 승인된 폐곡선 동형군 `0/O/o/\circlearrowright`에 속한 경우
- 두 단일 획과 하나의 2획 교차문자 병합 후보가 충돌하고 대안 분할 확률비가 0.35 이상인 경우

정답, writer, 목표 글자 수, 산술 계산은 입력하지 않았다. 최종 토큰, 획 그룹, 문자 수와 순서는 전혀 바꾸지 않고, 검토 시 기존 후보와 원시 획으로 되돌아가도록 상태만 추가한다.

| 평가 집합 | 전체 식 exact | 자동수락 | 자동수락 exact | 검토 필요 |
|---|---:|---:|---:|---:|
| public web 110식 | 108/110 (98.18%) | 88/110 (80.00%) | **88/88 (100%)** | 22식 |
| owned phone replay 49식 | 45/49 (91.84%) | 39/49 (79.59%) | **39/39 (100%)** | 10식 |
| 전체 159식 | 153/159 (96.23%) | 127/159 (79.87%) | **127/127 (100%)** | 32식 |

현재 남은 오분류 6식은 모두 검토로 분리됐지만 이 임계값은 같은 159식을 관찰해 정했으므로 독립 상용 정확도가 아니다. 전체 식 Top-1 exact도 `153/159` 그대로이며, 선택적 수락률을 전체 정확도로 바꿔 부르지 않는다.

CROHME 원시 722식 반복 진단에서는 전체 exact가 `77/722 (10.66%)`, 자동수락이 `530/722 (73.41%)`, 자동수락 exact가 `69/530 (13.02%)`였다. 선택적 정밀도는 소폭 올랐지만 절대 정확도가 낮고 CC BY-NC 자료이므로 제품 검증이 아니다. 결론은 `retain_as_shadow_only_not_product`, 제품 기본값 OFF다. 승격에는 새 상업 이용 가능 writer/formula-disjoint 식으로 자동수락 정밀도와 최소 커버리지를 먼저 고정해 통과해야 한다.

재현 해시는 다음과 같다.

- 수락 경계 설정: `4cf8359750f7df44f97143c4a896fc3b4367380ad298762843c5ba4e8f6aaf88`
- 159식 주석 출력: `93cb257f3b9a0a364af378b901e0d03eede668646a30b78580415034d844753c`
- CROHME 반복 진단: `8895965f7e46abaadcf4ef8d39eb3a9b2c08f95fac4b3e5bc88623d1b927c071`
- 통합 평가 manifest: `162da471669de0033f06ec5fecd807dab0ed7e1ea8930ac252a7e997d8308f61`

## 2026-08-20 수락 경계 검토량 축소 가설 기각

r1이 검토로 보낸 32식 중 22식은 “문맥 교정된 숫자가 다자리 수를 만듦” 조건이었다. 원시 HWR Top-5와 확률을 결과에 보존해 다시 감사한 결과, 실제 오류 `aiflow_0023`만 선택된 `6` 외에 대체 숫자 `0`이 의미 있게 경쟁했다. `P(0)/P(6)=0.529288`이었고, 나머지 21식에서 대체 숫자 최대 확률비는 `0.015908`이었다.

이를 일반 조건 `대체 숫자 확률 / 선택 숫자 확률 ≥ 0.5`로 좁힌 r2를 별도 shadow로 검증했다. 정답, writer, 목표 글자 수, 산술 결과는 사용하지 않았고 토큰·그룹 변화는 0이다.

| 지표 | 선택 r1 | 축소 r2 |
|---|---:|---:|
| 159식 자동수락 | 127/159 (79.87%) | **148/159 (93.08%)** |
| 159식 자동수락 exact | 127/127 (100%) | 148/148 (100%) |
| CROHME 자동수락 | 530/722 (73.41%) | **643/722 (89.06%)** |
| CROHME 자동수락 exact | **69/530 (13.02%)** | 77/643 (11.98%) |

r2는 커버리지를 높였지만 고정 비교 기준인 CROHME 선택적 정밀도가 `1.04%p` 하락했다. 따라서 `reject_r2_keep_r1_selected`로 기각했으며 실제 선택 설정은 계속 r1이다. r2도 같은 159식을 관찰한 posthoc이고 CROHME도 비상업 반복 진단이므로 제품 승격 근거가 아니다.

후보 증거 확장 runtime SHA-256은 `47033bd2eefe2abf25d2958824974d5a69c46e704e62cf24f516e8625e62c0af`, r2 설정은 `c6d2c385c4e77f2509ff83e2d2c14894cc0a1b74e977d02cd77a11302bb1d606`, r2 평가 manifest는 `7f993b3168bb593a3881ff2e7f6dd8ff462a39afa5c3fc933ef7a4aa23a736f6`, 선택 manifest는 `366afc18d6a96bc1b9cda136087d0de51ebb5eaafc822d97726de5e39ebc7d14`이다.
