---
language:
- ko
- en
license: other
task_categories:
- image-classification
tags:
- online-handwriting
- mathematical-expression-recognition
- aggregate-only
pretty_name: AIFlow Math Ink 0.9 Aggregate Data Card
---

# AIFlow Math Ink 0.9 Aggregate Data Card

이 저장소는 데이터셋 원본이 아니라 AIFlow Math Ink 0.9 수집·선별 데이터의 **비식별 집계 카드**다.

## 원본을 공개하지 않는 이유

수집 동의 범위는 `commercial_model_training:aiflow_math_ink`, 즉 AIFlow 상업용 필기 인식 모델의 학습·검증이다. 공개 재배포 동의는 받지 않았다. 따라서 다음 항목은 포함하지 않는다.

- stroke 좌표·순서·timestamp·필압
- 필기 PNG/수식 이미지
- contributor ID, session ID, IP 또는 원본 파일 경로
- 행 단위 prompt/answer 레코드

## 집계

- 수집 레코드: 119
- 검수 상태: valid 110, pending 4, reject 5
- 출처 유형: public web collection 70, owned phone replay 49
- 0.9 학습 subset: 47개 수식, 211개 문자, 25개 관측 label, 3 contributors
- replay: 49개 수식, 171개 문자

`aggregate_statistics.json`은 비식별 집계만 담는다. `normalized_class_means.png`는 선택된 211개 문자 feature의 class 평균으로, 개별 필기를 복원하는 row-level sample이 아니다.

## 허용 범위

이 카드의 집계 수치와 설명은 연구 재현성 검토에 사용할 수 있다. 원본 필기 데이터에 대한 접근권이나 재배포 라이선스를 부여하지 않는다.
