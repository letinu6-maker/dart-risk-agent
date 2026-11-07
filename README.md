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
│   ├── raw/cache/       # 기업×연도 수집 캐시 (끊겨도 이어서 수집)
│   └── labels/          # events.csv(위험 사건), corp_list.csv(학습 대상 기업)
├── models/              # 학습된 모델 (risk_model.joblib만 커밋 — 데모 배포용)
├── streamlit_app.py     # 데모 웹 화면
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

## 실데이터로 돌리는 순서

| # | 명령 | 하는 일 | API 호출 |
|---|---|---|---|
| 0 | `python scripts/check_api.py` | OpenDART 응답이 코드 가정과 맞는지 점검 | 약 10회 |
| 1 | `python scripts/build_labels.py --start 2016 --end 2025 --n-normal 600` | 위험 사건(events.csv)과 학습 대상 기업(corp_list.csv) | 약 150회 |
| 2 | `python scripts/build_dataset.py --start 2016 --end 2024` | 기업×연도 재무 변수 + 라벨 → `data/processed/panel.csv` | 기업 수 × 9년 × 3회 |
| 3 | `python scripts/build_dataset.py --start 2016 --end 2024 --text` | 문구 변화 변수 추가 (캐시된 재무 변수는 다시 안 부름) | 위 행 수 × 4회 |
| 4 | `python scripts/train.py --data data/processed/panel.csv --test-start 2022` | 모델 3종 비교, `models/risk_model.joblib` 저장 | 0 |
| 5 | `notebooks/ml_model.ipynb`에서 `USE_SYNTHETIC = False` | 가점 제출용 노트북 실데이터로 재실행 | 0 |

- OpenDART는 **키당 하루 호출 한도**가 있어요. 2~3단계는 한도를 넘으면 스스로 멈추고, 다음 날 같은 명령을 다시 실행하면 이어서 수집해요.
- 팀원끼리 나눠 돌리려면 각자 키로 `--shard 0/3`, `--shard 1/3`, `--shard 2/3`을 실행하고 `data/raw/cache/` 폴더를 합치면 돼요.
- GitHub·터미널이 처음인 팀원은 `TEAM_GUIDE.md`대로 `team_collect.py`를 더블클릭하면 돼요. 설치·키 입력·2018~2024년 수집(문구 변화 포함)·결과 압축까지 한 번에 해요.
- 1단계 뒤에 `events.csv`의 `report_nm`을 훑어보고 부실과 무관한 사건이 있으면 `src/dart_agent/labels.py`의 `EXCLUDE_TITLE`에 키워드를 추가하세요.

## 데모 화면

```bash
streamlit run streamlit_app.py
```

기업명(또는 종목코드)과 사업연도를 넣으면 Agent 4단계 진행 상황, 위험 점수·등급, 기여 변수 그래프, 리포트가 나와요. 학습된 모델이 없으면 합성 데이터로 임시 모델을 만들고 화면에 경고를 띄워요.

**데모 URL 만들기 (Streamlit Community Cloud, 무료)**
1. 4단계에서 만든 `models/risk_model.joblib`를 커밋·푸시 (이 파일만 git에 올라가도록 설정되어 있음)
2. https://share.streamlit.io → GitHub 로그인 → **Create app** → 저장소 `letinu6-maker/dart-risk-agent`, 파일 `streamlit_app.py`
3. **Advanced settings → Secrets**에 입력:
   ```toml
   DART_API_KEY = "발급받은키"
   ANTHROPIC_API_KEY = "선택"
   ```
4. 비공개 저장소도 배포되지만, 앱 공개 범위는 Streamlit 설정에서 따로 정해요.

터미널에서 한 기업만 돌리려면:

```bash
python scripts/run_agent.py --corp 005930 --year 2024
```

## 반드시 지킬 것

- **데이터 누수 금지**: t년 사업보고서 제출일을 예측 시점으로 두고 그 뒤 1년의 사건을 예측. 제출 직후 30일 안 사건은 학습에서 제외. train/test는 연도 기준으로 분리 (`model.time_split`). 라벨 정의는 `docs/data_sources.md`.
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
