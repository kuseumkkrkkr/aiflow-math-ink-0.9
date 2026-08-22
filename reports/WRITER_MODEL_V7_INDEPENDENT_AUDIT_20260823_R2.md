# 계층형 작가모델 v7 독립 감사

## 판정

- 최종 판정: **RAW_WRITER_SHADOW_CANDIDATE**
- raw catalog: 96수식 / 7작가 / 393glyph / 28클래스.
- 동결 원본 재구성: tensor exact=True, token exact=True, source hash mismatch=0건.
- 생성식 원천: raw 61 / external fallback 11.
- 생성 작가별 raw 부모 최대: 1명; provenance 실패 0건.
- known replay 재계산: Top-1 85.48%, Top-5 95.16%, 수식 exact 60.38%, Top-5 oracle 86.79%.
- replay 영수증 불일치: 0건; legacy writer 중첩: 0명.

## 경계

- v7은 합성 전용 v6에서 실제 legacy raw writer 증거를 우선하는 구조로 진전했다.
- replay 53수식은 writer-disjoint이지만 이미 알려진 자료이므로 튜닝·promotion 근거가 아니다.
- 소문자 t는 raw 증거 1건만 있고 372-class 출력부에는 없어 evidence-only다.
- 상용 승격에는 새로 수집하고 사전 동결한 작가/수식 acceptance가 필요하다.
