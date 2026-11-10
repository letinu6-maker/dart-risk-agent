# ML 가점 대응 계획 (최대 +20)

ML 모델을 쓰는 팀만 가점을 받고, 노트북(`notebooks/ml_model.ipynb`)을 제출해야 합니다.

| 가점 항목 (+5씩) | 계획 | 노트북 위치 |
|---|---|---|
| 수업 모델 활용 | Logistic Regression, Decision Tree, Random Forest | 4장 |
| 모델 선택 근거 | 금융 업무는 판단 이유 설명이 중요 → 설명 가능한 Logistic·Decision Tree를 기본, Random Forest는 성능 비교용. 문제·데이터 특성에 연결해 서술 | 3장, 4장 결과 요약 |
| 검증 | 연도 기준 train/test 분리로 누수 방지. 불균형이라 Recall, PR-AUC 사용. class_weight로 가중치 조정 | 2장, 4장 |
| 실험·해석 | 모델 3종 비교 + 변수 중요도. "문구 변화 변수를 넣으면 성능이 오르는가" (넣은 모델 vs 뺀 모델) | 5~8장 |

## 체크리스트

- [x] 라벨 정의를 노트북 맨 앞에 명시
- [x] train 마지막 연도 < test 첫 연도 (assert로 확인)
- [x] 사건 이후 공시가 변수에 없는지 확인
- [x] Accuracy 대신 Recall·PR-AUC를 쓰는 이유 서술
- [x] 데이터 출처 표 삽입
- [x] 새 환경에서 처음부터 끝까지 재실행

## 실데이터 결과 (2026-11-10, `notebooks/ml_model.ipynb`)

| 모델 | Recall | Precision | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic (기본) | 0.672 | 0.360 | 0.439 | 0.788 |
| Decision Tree | 0.664 | 0.334 | 0.376 | 0.752 |
| Random Forest | 0.607 | 0.400 | 0.436 | 0.803 |

평가 2022~2024, 1,592행, 위험 비율 15.3% (무작위 PR-AUC 기준선 0.153). 문구 변수 추가 효과 PR-AUC +0.005 (95% 구간 -0.001 ~ +0.010).
