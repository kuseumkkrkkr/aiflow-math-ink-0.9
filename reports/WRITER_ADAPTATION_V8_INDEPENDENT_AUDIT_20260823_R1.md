# Writer adaptation v8 독립 감사

## 판정

**AUTO_REJECT_CONFIRMED** — v8은 작가별 보정 가능성은 보였지만 제품 채택 조건은 충족하지 못했다. 원본 `report.json`은 당시 실행 영수증으로 보존되며, 최종 채택 판정은 `adoption_decision.json`의 `AUTO_REJECTED`가 우선한다.

| 구간 | Glyph Top-1 | Formula exact | Top-5 | 판정 |
|---|---:|---:|---:|---|
| Legacy baseline | 90.33% | 65.75% | 98.33% | 기준 |
| Legacy prototype_200 | 92.67% | 72.60% | 98.33% | writer_004 회귀로 거부 |
| Known replay baseline | 84.66% | 57.78% | 94.48% | 진단 전용 |
| Known replay prototype_200 | 86.50% | 60.00% | 94.48% | 승격 근거 아님 |

## 독립 확인

- v7 원시 텐서 393행과 동결 체크포인트에서 `prototype_200` 결과를 재계산했다.
- Legacy 300 glyph/73 formula의 집계와 작가별 수치가 원 보고서와 일치했다.
- writer_004는 Top-1 -3.23pp, formula exact -12.50pp 회귀했다.
- Known replay 원장 163 glyph/45 formula의 ownership truth, evaluation-only 소속, 지표를 재계산했다.
- Replay glyph pairing은 5개 개선/2개 회귀, exact McNemar p=0.453125; formula pairing은 3개 개선/2개 회귀, p=1.0이다.
- 모든 adapted Top-5 집합은 frozen baseline Top-5와 동일하며 후보 위반은 0건이다.
- 현재 구현은 identity를 포함하고, Legacy 전 작가 비회귀와 양의 집계 개선을 함께 만족하는 후보가 없어 replay를 열기 전에 `PRE_REPLAY_FAIL_CLOSED`로 종료한다.
- HWR, 체크포인트, 제품 runtime, CROHME, MathWriting, v7은 변경되지 않았다.

## 해석 경계

- Legacy p=0.0390625는 같은 개발 grid에서 선택된 설정의 값이므로 확증 통계로 사용할 수 없다.
- Known replay p=0.453125는 약하고, 이미 알려진 writer replay라 promotion evidence가 아니다.
- 다음 확장은 새로운 untouched 작가/수식 acceptance set을 확보한 뒤 사전 고정된 단일 설정으로만 평가해야 한다.
