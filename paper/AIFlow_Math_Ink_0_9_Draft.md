# AIFlow Math Ink 0.9: 구조 분리형 온라인 수식 필기 인식과 소규모 사용자 데이터 보정

> 연구 논문 초안 · 2026-08-08 · 제품 성능 주장 아님

## 초록

본 연구는 온라인 수식 필기 인식을 형태/그리딩, 문자 후보 판정, 보수적 결정의 세 단계로 분리하고, 원본 stroke 순서와 raw fallback을 보존하는 AIFlow Math Ink 구조를 제안한다. 0.9 단계에서는 glyph bbox 중심 정규화, 종횡비 보존 letterbox, class/contributor 균형 sampling을 적용한 additive calibrator를 기존 378-label trajectory encoder에 결합하고 `=` label을 추가하였다. 47개 수식의 211개 문자로 학습하고, 정확한 수식 및 prompt 중복이 없는 49개 수식·171개 문자 replay에서 평가했다. 선택 seed31은 동일 baseline 대비 문자 Top-1을 50.29%에서 87.72%, Top-5를 70.18%에서 98.83%, 수식 exact를 26.53%에서 67.35%로 높였다. 그러나 replay의 contributor ID가 없어 writer-disjoint를 증명할 수 없고 `f`, `g` 회귀가 남아 있으므로 연구 후보로만 채택한다.

## 1. 연구 목적

목표는 자동채점이나 정답 저장이 아니라 사용자가 순차 입력한 수학 필기를 가능한 한 안정적으로 문자와 수식 형태로 복원하는 것이다. 완전한 LaTeX 일치만을 성공으로 보지 않고 `3X4`, `3x4`, `3*4`처럼 의미상 동형인 출력도 후속 계층이 처리할 수 있도록 후보와 원본을 보존한다.

## 2. 관련 연구

CROHME는 온라인 필기 수식 인식의 symbol segmentation, recognition, structural analysis를 함께 평가하는 대표 벤치마크다 [1]. SCAN은 stroke sequence와 spatial 구조를 결합해 온라인 수식을 인식한다 [2]. MathWriting은 대규모 온라인 수식 데이터와 공식 train/validation/test 분할을 제공한다 [3]. 본 연구는 대형 end-to-end decoder 자체보다, 실시간 입력에서 계층별 책임과 실패 복구 계약을 명확히 하는 데 초점을 둔다.

## 3. 시스템 구조

### 3.1 형태/그리딩

stroke의 시간 순서, pen-up 경계, 기하 관계를 보존해 문자 후보 group을 만든다. 이 계층은 문자를 확정하지 않고 `□ op □ = □` 같은 레이아웃 가능성과 stroke ownership 후보를 전달한다. 경계가 불확실하면 단일 partition을 강제하지 않고 후보 또는 raw stroke를 유지한다.

### 3.2 문자 판정

각 후보 group을 bbox crop과 종횡비 보존 letterbox로 정규화한 뒤 trajectory encoder에서 Top-k 문자 후보를 얻는다. 0.9 calibrator는 base encoder를 고정하고, 관측된 label의 logit만 additive correction한다. `=`는 379번째 출력으로 추가했다.

### 3.3 결정

결정기는 문자 Top-1을 무조건 덮어쓰지 않는다. 수식 위치와 인접 후보가 충분한 증거를 제공할 때만 동형 문자를 보정하고, 확신이 없으면 classifier 후보 또는 raw 결과로 abstain한다. 이는 과거 단일 문자에서 올바른 `g`, `)`를 잘못 바꾼 사례를 막기 위한 계약이다.

## 4. 데이터와 윤리

전체 수집 기록은 119개이며 valid 110, pending 4, reject 5다. 0.9 학습에는 검수된 47개 수식, 211개 문자, 25개 관측 label, 3 contributors를 사용했다. replay는 기존 수동 ownership 49개 수식, 171개 문자다. 학습과 replay 사이 exact formula 및 prompt 중복은 없지만 replay에 contributor ID가 없어 writer-disjoint split은 증명할 수 없다.

동의 문구는 AIFlow 상업용 필기 인식 모델의 학습·검증을 허용하지만 공개 재배포는 명시하지 않는다. 따라서 원본 stroke, 이미지, contributor/session ID는 공개하지 않고 비식별 집계만 배포한다.

## 5. 실험 설정

base는 AIFlow Math Ink 0.6 seed17의 378-label encoder와 online adapter다. 입력은 glyph-local 정규화 후 수식 상대 위치 채널을 고정해 문자 형태 경로에서 제거했다. additive head만 20 epoch 학습했고 class와 contributor 조합의 역빈도 sampling을 사용했다. seed 17, 31, 47을 비교했으며 모든 평가는 동일한 정규화 baseline과 같은 replay에서 수행했다.

## 6. 결과

| 지표 | baseline | seed17 | seed31 | seed47 | 3-seed 평균 |
|---|---:|---:|---:|---:|---:|
| 문자 Top-1 | 50.29% | 85.96% | 87.72% | 85.96% | 86.55% |
| 문자 Top-5 | 70.18% | 98.25% | 98.83% | 98.25% | 98.44% |
| 수식 exact | 26.53% | 67.35% | 67.35% | 65.31% | 66.67% |
| 수식 Top-5 oracle | 42.86% | 93.88% | 95.92% | 93.88% | 94.56% |
| `=` Top-1 | 0.00% | 100% | 100% | 100% | 100% |

seed31은 171개 문자 중 67개를 개선하고 3개를 악화시켰다. `f`는 6/6에서 5/6, `g`는 4/5에서 2/5로 하락했다. `/`와 `\sqrt{}`는 각각 단 하나의 replay 표본에서 여전히 실패했다.

## 7. 논의

결과는 정규화와 제한적 additive calibration이 소규모 실제 필기 분포 적응에 효과가 있을 가능성을 보여준다. 하지만 표본 수가 작고 writer-disjoint가 아니므로 일반 사용자 성능이나 상용 수준을 증명하지 않는다. 높은 Top-5는 후보 보존 전략의 유용성을 뒷받침하지만, 수식 exact와 동일하지 않다. 다음 검증은 contributor 식별자가 있는 writer-disjoint split, `f/g` 및 희소 기호 보강, 그리딩 ownership 평가를 동시에 보고해야 한다.

0.9 실험은 CROHME나 MathWriting에서 수행하지 않았다. 과거 CROHME 연구 수치는 구조 탐색의 역사적 근거일 뿐 본 결과와 직접 비교하지 않는다. MathWriting도 본 실험에 포함하지 않았다.

## 8. 결론

AIFlow Math Ink 0.9는 단계별 책임, Top-k 보존, raw fallback이라는 구조 위에 실제 필기 분포용 경량 calibrator를 추가해 내부 replay를 개선했다. 현재 근거로는 연구 후보 채택이 타당하지만 제품 채택은 이르다. 후속 연구의 통과 조건은 writer-disjoint 재현, 희소 문자 회귀 통제, 동일 split에서 그리딩·문자·결정·수식 지표의 동시 개선이다.

## 참고문헌

1. H. Mouchère et al., “ICFHR 2014 Competition on Recognition of On-line Handwritten Mathematical Expressions (CROHME 2014),” ICFHR 2014. DOI: 10.1109/ICFHR.2014.42. https://www.cs.rit.edu/~rlaz/files/Crohme2014FinalVersion.pdf
2. J. Wang, J. Du, and J. Zhang, “Stroke Constrained Attention Network for Online Handwritten Mathematical Expression Recognition,” 2020. https://arxiv.org/abs/2002.08670
3. P. Gervais, A. Fadeeva, and A. Maksai, “MathWriting: A Dataset For Handwritten Mathematical Expression Recognition,” 2024. https://arxiv.org/abs/2404.10690

## 재현성과 공개 상태

원시 지표는 `reports/seed31_metrics.json`, 학습 진입점은 `scripts/train_normalized_balanced_touch_calibrator09.py`, 시각 감사는 `artifacts/normalized_class_means.png`에 있다. private 입력 경로와 원본 데이터가 필요하므로 제3자가 공개물만으로 전체 학습을 완전히 재현할 수는 없다. 공개 체크포인트는 base 의존성을 명시한 연구 artifact다.
