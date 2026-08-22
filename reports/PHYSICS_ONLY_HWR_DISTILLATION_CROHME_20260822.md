# AIFlow Math Ink 1.0 물리 지식 단독 증류와 CROHME 단회 평가

## 결론

- **단독 상용 채택 불가**다. 물리 엔진만으로 만든 학생 HWR는 CROHME test 문자 Top-1 **4.23%**, Top-5 **16.04%**에 그쳤다.
- 물리 엔진은 속도·감쇠·떨림·드리프트·양자화 같은 **필기 동역학**은 전달했지만, 372개 문자의 사람 중심선·획 순서·필체 의미를 생성하지 못했다.
- Vercel, PrivateData, 직접수집, 기존 온라인 궤적, 기존 HWR 가중치는 학습 입력에서 모두 **0행/0개**였다.
- CROHME는 checkpoint SHA를 동결한 뒤 test split을 **한 번만** 평가했다. 평가 뒤 gradient update, variant 선택, threshold 변경은 모두 0회다.
- 따라서 이 결과는 “물리 엔진 지식만으로 기존 HWR를 대체할 수 있다”는 가설을 기각한다. 물리 엔진은 실제 상업권리 온라인 중심선 데이터에 붙는 보조 증강·불변성 교사로만 남길 가치가 있다.

## 실험 경계

| 항목 | 계약 |
|---|---|
| 입력 | 128점 × 5채널 (`x`, `y`, `delta_t`, `stroke_start`, `observed`) |
| 출력 | 고정 372-class 단일 math head |
| 학생 | 무작위 초기화, 4-block Transformer, 858,613 parameters |
| 문자 토폴로지 | Matplotlib 번들 STIX·DejaVu 벡터 글리프를 임시 스켈레톤화 |
| 운동 교사 | 48 Hz, 192 Hz 내부 적분, 최소-저크, 감쇠 2차 추종, 떨림, 드리프트, 13-bit 양자화 |
| 증류 신호 | hard class + 물리 중심 연성 혼동분포 + 76차원 물리 기술자 회귀 |
| 학습 분할 | 절차형 작가 16명, 클래스당 16행, 총 5,952행 |
| 선택 분할 | unseen 절차형 작가 4명 1,488행 + 강한 물리 stress 작가 4명 1,488행 |
| 평가 | CROHME2019 test, truth stroke-group 문자 분류; raw grouping/LgEval 아님 |

벡터 글리프를 중심선으로 바꾸는 과정에서 래스터 마스크는 메모리 안의 임시 변환 단계로만 사용했다. 모델 입력·저장 코퍼스는 전부 순서형 온라인 궤적이다.

## CROHME 노출 전 합성 게이트

| 분할 | 표본 | Top-1 | Top-5 | 최악 작가 Top-1 |
|---|---:|---:|---:|---:|
| unseen development | 1,488 | 87.57% | 99.93% | 87.10% |
| stronger physics stress | 1,488 | 87.50% | 99.80% | 86.29% |

- 40 epoch 상한 중 합성 dev/stress 선택 점수의 최고점은 **37 epoch**였다.
- r1 30 epoch와 r2의 앞 30 epoch 로그는 동일 seed에서 주요 metric이 재현됐다.
- 이 높은 수치는 생성 분포 내부의 자기 일관성일 뿐, 실제 사람 필기 성능 근거가 아님을 CROHME 결과가 확인했다.

## CROHME test 단회 결과

| 지표 | 결과 |
|---|---:|
| 원본 InkML | 1,199개 |
| 평가 수식 | 1,198개 |
| 지원 문자 | 11,991개 |
| 관측 truth 클래스 | 92종 |
| 문자 Top-1 | **4.23%** (507/11,991) |
| 문자 Top-5 | **16.04%** (1,923/11,991) |
| Top-5 밖 | **10,068개** |
| strict macro Top-1 / Top-5 | 7.42% / 17.92% |
| 평균 / 중앙 truth rank | 87.61 / 37 |
| Top-5 밖 평균 truth rank | 103.77 |
| 완전 지원 수식 | 849개 |
| 완전 지원식 truth-group exact | **0.00%** |
| 완전 지원식 Top-5 oracle | **0.12%** (1/849) |
| GPU glyph당 추론 | 0.200 ms |
| checkpoint 크기 | 3,739,528 bytes |

카테고리별 Top-1/Top-5는 핵심 연산자 7.20%/34.05%, 숫자 1.51%/11.11%, 영문 3.50%/7.60%, 기타 수학 5.77%/14.90%였다. 속도와 크기는 충분하지만 정확도가 사용 불가능한 수준이다.

## 동형문자 판단

| 군 | 표본 | exact Top-1 | exact Top-5 | family Top-1 | family Top-5 |
|---|---:|---:|---:|---:|---:|
| `1 · | · /` | 1,008 | 0.20% | 6.05% | 0.20% | 6.15% |
| `0 · O · o` | 280 | 1.07% | 5.00% | 1.07% | 14.64% |
| `x · \times` | 824 | 2.55% | 5.22% | 5.70% | 15.41% |

동형 family 안으로라도 들어오는 비율이 낮다. 따라서 현재 실패는 문맥 모델이 exact identity만 고르지 못한 문제가 아니다. 형태 후보 자체가 대부분 Top-5 밖이므로 문맥 레이어로 넘겨 해결할 수 없다.

## 실패 원인

1. **물리 지식과 문자 지식은 다르다.** 운동계는 주어진 경로를 사람답게 흔들 수 있지만 `1`, `x`, `\int`의 올바른 중심선과 획 순서를 발명하지 못한다.
2. **폰트 외곽선의 중심선 변환이 실제 필기와 다르다.** 분기 구조를 잃지 않기 위한 retrace가 이중 궤적을 만들고, 실제 사용자의 단일 pen-down stroke와 다른 topology를 학습시켰다.
3. **letterbox 뒤 기호가 충돌했다.** `- → \dots` 826건, `1 → \vdots` 175건, `= → \equiv` 82건처럼 작은 점·반복 기호와 선 기호가 같은 크기로 확대되어 구분 경계가 뒤집혔다.
4. **372-class 사전 보정이 없다.** CROHME truth는 92종인데 Top-1 예측은 314종으로 흩어졌다. 학습에 실제 필기 빈도나 topology가 없어서 희귀 수학 기호가 흔한 숫자·영문을 대체했다.
5. **출력 계약의 빈 클래스가 남았다.** `\sin`, 쉼표, `t`, `\cos`, `\lim`, `\log`, `\tan`, 마침표, 느낌표 542개 truth group이 372 출력에 없어 완전 지원 수식이 849/1,198개로 줄었다.

## 채택 판단

- **폐기:** 물리 엔진·폰트 토폴로지만으로 만든 독립 HWR checkpoint의 제품 runtime 채택.
- **보존:** 48 Hz 운동계, 합성 작가 latent, topology/경로길이 안전 게이트, 물리 기술자 보조 loss.
- **향후 사용 조건:** Vercel/직접수집을 쓰지 않더라도 UJI·ISGL·UCI·HWRT 등 권리 검토가 끝난 실제 온라인 중심선의 문자 topology를 교사로 두고, 물리 엔진은 그 경로를 변형하는 역할만 맡긴다.
- **CROHME 금지:** 이 결과의 오류 라벨·빈도·threshold를 다음 학습이나 모델 선택에 사용하지 않는다. 새 후보는 프로젝트 소유 신규 작가 또는 별도 상업권리 holdout에서 먼저 동결한다.
- **제품 변경:** 없음. runtime switch, Vercel 호출·배포, 기존 HWR/문맥 checkpoint 교체를 하지 않았다.

## 증거

- 최종 checkpoint: `artifacts/physics_only_hwr_distillation_20260822_r2/physics_only_hwr_checkpoint.pt`
- checkpoint SHA-256: `b9e16a587818e813f53e5fa14efca189269b4c33707f92e30fdacff8a9bf6cc3`
- 절차 코퍼스 SHA-256: `b4ed3712dee627a82a838d3776bca1ac0f5d475e872cd3cc1fc129d65d180e72`
- 단회 평가 보고서 SHA-256: `ff26d7e3182047014f2906a699bbcd9dd078174fcf8a81fd54084276a76457bd`
- 평가 전·후 checkpoint SHA와 mtime: 동일
- 평가 영수증: `artifacts/physics_only_hwr_distillation_20260822_r2/crohme_evaluation_receipt.json`

STIX는 번들 `LICENSE_STIX`의 SIL Open Font License 1.1, DejaVu는 번들 `LICENSE_DEJAVU`를 계보에 고정했다. 이는 기술적 권리 근거 기록이며 최종 법률 의견은 아니다.
