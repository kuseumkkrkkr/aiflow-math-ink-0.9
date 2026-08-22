# 계층형 작가모델 독립 감사 (aiflow-hierarchical-writer-model/v6)

## 판정

- 최종 판정: **AUGMENTATION_SHADOW_CANDIDATE**
- 구조·provenance 스모크: **통과**
- 작가별 동결 HWR Top-5 최저: **100.00%**
- 현재 입력은 모두 DTW/physics 합성은행이므로 `관측 allograph`가 아니라 `합성 allograph plan`으로만 표현해야 한다.
- 원시 인간 writer/session의 미접촉 작가 분리 검증이 없어 상용 작가모델 승격 근거는 없다.
- 평가 전용 project bank는 현재 run의 catalog 입력에서 격리됐다.

## 입력 증거

| 구분 | 행 | 합성 행 | 원시 인간 관측 | cross-writer 합성 | 고유 부모 작가 지문 |
|---|---:|---:|---:|---:|---:|
| approved_external_cleanroom | 4096 | 4096 | 0 | 0 | 0 |

## 생성 구조

- 수식×작가 행: 12; glyph: 72; stroke: 123.
- source join/partition/time 실패: 0건.
- 인접 baseline glyph 겹침 후보: 0건(진단 지표, 자동 실패 gate 아님).
- 선택 glyph 중 합성 원천: 72건; 원시 인간 관측 원천: 0건.

## 왜 Top-5 게이트만으로 부족한가

- 생성 원천과 동결 HWR이 같은 승인 데이터 계보를 공유한다. 이는 행 단위 누수로 단정할 수 없지만 독립 일반화 평가는 아니다.
- HWR Top-5로 allograph를 고르면 문자 정체성 보존에는 도움이 되지만, 분류기가 이미 아는 필체만 남기는 순환 선택이 된다.
- 따라서 이 값은 증강 안전성 지표로만 사용하고 인간 필체 다양성 또는 새 작가 성능으로 해석하지 않는다.

## 다음 승격 조건

- 평가 전용 387행과 겹치지 않는 프로젝트 소유 training/calibration writer 세트
- 원시 인간 writer/session ID가 보존된 숫자·A-Z·a-z·수학기호 온라인 궤적
- 동결 생성기와 동결 HWR 선택이 끝난 뒤의 미접촉 작가 분리 평가
- 누락 소문자 t 및 t/f/+ 동형군의 프로젝트 소유 다작가 표본
- 위 조건 전에는 기존 HWR/runtime/checkpoint를 변경하지 않고 shadow 구조 스모크로 유지한다.
