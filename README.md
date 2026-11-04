# 공시 변화 기반 상장기업 재무위험 조기경보 AI Agent

> 2026 D&A ML Competition (KIC AI Agent 공모전)
> "숫자만 보지 않고, 공시 문구가 어떻게 바뀌었는지까지 읽어서 위험 기업을 먼저 알려주는 Agent"

질문에 답만 하는 챗봇이 아니라, OpenDART API를 직접 호출해 공시를 가져오고 → 전년 대비 재무·문구 변화를 계산하고 → ML로 위험 점수를 매긴 뒤 → LLM이 근거 리포트를 쓰는 **4단계 자동화 Agent**입니다.

## 일정

| 항목 | 마감 |
|---|---|
| 기획서 KIC 제출 | 11/12(목) 10:00 |
| D&A 내부 제출 (각 팀 멘토) | 11/12(목) 23:59 |
| 본선 MVP · 데모 URL · ML 노트북 | 본선 일정 확인 후 기입 |

## Agent 파이프라인

| 단계 | Agent 행동 | 모듈 | 도구 |
|---|---|---|---|
| ① 수집 | 재무제표, 감사의견, 공시 목록·원문 수집 | `src/dart_agent/collect.py` | OpenDART API |
| ② 변화 탐지 | 전년 대비 재무지표 변화, 주석·감사보고서 문구 변화량 | `features.py`, `text_diff.py` | pandas, TF-IDF 코사인 유사도 |
| ③ 위험 예측 | 위험 점수와 기여 변수 | `model.py` | Logistic, Decision Tree, Random Forest |
| ④ 리포트 | 점수와 근거를 읽기 쉬운 문장으로 | `report.py` | LLM API (키 없으면 템플릿 문장) |

`agent.py`가 네 단계를 순서대로 호출합니다.

## 폴더 구조

```
dart-risk-agent/
├── src/dart_agent/      # Agent 4단계 코드
├── notebooks/           # ML 가점 제출용 노트북 (ml_model.ipynb)
├── data/
│   ├── raw/             # API 원본 (git 제외)
│   ├── processed/       # 학습용 테이블 (git 제외)
│   └── labels/          # 라벨 원천: KRX KIND 상장폐지·관리종목 목록 (직접 다운로드)
├── docs/
│   ├── proposal.md      # 기획 요약 (원본: 김나형 제안서)
│   ├── data_sources.md  # 데이터 출처 표 — 미기재 시 -10점
│   └── ml_plan.md       # 가점 4항목 대응 계획
├── tests/               # 합성 데이터로 돌아가는 테스트
└── scripts/             # 수집·학습 실행 스크립트
```

## 시작하기

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # DART_API_KEY 입력 (https://opendart.fss.or.kr → 인증키 신청)
pytest                  # API 키 없이 합성 데이터로 전체 파이프라인 확인
```

한 기업에 대해 Agent 실행:

```bash
python scripts/run_agent.py --corp 005930 --year 2025
```

## 반드시 지킬 것

- **데이터 누수 금지**: t년 공시로 t+1년 사건을 예측. 사건 이후 제출된 공시는 변수에서 제외. train/test는 연도 기준으로 분리 (`model.time_split`).
- **불균형 지표**: Accuracy 대신 Recall, PR-AUC. 이유를 노트북에 적기.
- **데이터 출처**: `docs/data_sources.md` 표를 계속 갱신. 수집일 기록.
- **API 키 커밋 금지**: `.env`는 git에서 제외되어 있음.
- **노트북 재실행 확인**: 제출 전 새 환경에서 처음부터 끝까지 실행.

## 팀

| 이름 | 역할 |
|---|---|
| 김나형 | 주제 제안 |
| 박준영 | |
| | |
