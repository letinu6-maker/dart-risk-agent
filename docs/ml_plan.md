# ML 가점 대응 계획 (최대 +20)

ML 모델을 쓰는 팀만 가점을 받고, 노트북(`notebooks/ml_model.ipynb`)을 제출해야 합니다.

| 가점 항목 (+5씩) | 계획 | 노트북 위치 |
|---|---|---|
| 수업 모델 활용 | Logistic Regression, Decision Tree, Random Forest | 4장 |
| 모델 선택 근거 | 금융 업무는 판단 이유 설명이 중요 → 설명 가능한 Logistic·Decision Tree를 기본, Random Forest는 성능 비교용. 문제·데이터 특성에 연결해 서술 | 3장 |
| 검증 | 연도 기준 train/test 분리로 누수 방지. 불균형이라 Recall, PR-AUC 사용. class_weight로 가중치 조정 | 2장, 5장 |
| 실험·해석 | 모델 3종 비교 + 변수 중요도. "문구 변화 변수를 넣으면 성능이 오르는가" (넣은 모델 vs 뺀 모델) | 6장, 7장 |

## 체크리스트

- [ ] 라벨 정의를 노트북 맨 앞에 명시
- [ ] train 마지막 연도 < test 첫 연도 (assert로 확인)
- [ ] 사건 이후 공시가 변수에 없는지 확인
- [ ] Accuracy 대신 Recall·PR-AUC를 쓰는 이유 서술
- [ ] 데이터 출처 표 삽입
- [ ] 새 환경에서 처음부터 끝까지 재실행
