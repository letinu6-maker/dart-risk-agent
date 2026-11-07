"""데모 화면: 기업과 사업연도를 넣으면 Agent가 4단계를 돌려 위험 점수와 근거 리포트를 보여준다.

    streamlit run streamlit_app.py

키는 .env(로컬) 또는 Streamlit Cloud의 Secrets(배포)에서 읽는다.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # Streamlit Cloud는 pip install -e 를 하지 않음

from dart_agent import collect, report  # noqa: E402
from dart_agent.agent import RiskAgent  # noqa: E402

MODEL_PATH = Path(os.getenv("DART_AGENT_MODEL", ROOT / "models" / "risk_model.joblib"))

st.set_page_config(page_title="공시 위험 조기경보 Agent", page_icon="📑", layout="wide")


def load_secrets() -> None:
    """Streamlit Cloud Secrets → 환경변수. 로컬은 .env를 collect.get_client가 읽는다."""
    try:
        for k in ("DART_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL"):
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass  # secrets.toml이 없으면 그냥 넘어감


@st.cache_resource(show_spinner="모델 준비 중…")
def ensure_model(path: str) -> str:
    """학습된 모델이 없으면 합성 데이터로 임시 모델을 만든다 (화면에 경고 표시)."""
    import joblib

    model_path = Path(path)
    if model_path.exists():
        return joblib.load(model_path).get("trained_on", "unknown")
    from dart_agent import model as M
    from dart_agent.synthetic import make_panel

    df = make_panel()
    feats = M.FINANCIAL_FEATURES + M.TEXT_FEATURES
    m = M.build_models()["Logistic"].fit(df[feats], df[M.LABEL])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "features": feats, "trained_on": "synthetic"}, model_path)
    return "synthetic"


@st.cache_resource(show_spinner="OpenDART 연결 중… (처음엔 기업 목록을 받느라 20초쯤 걸려요)")
def get_dart():
    return collect.get_client()


def demote_headings(md: str) -> str:
    """리포트의 ##, ### 제목을 두 단계 낮춰 화면의 섹션 제목보다 작게 보이게 한다."""
    import re

    return re.sub(r"^(#{1,4}) ", lambda m: "#" * min(len(m.group(1)) + 2, 6) + " ", md, flags=re.M)


def contributions_chart(contrib: list[dict]) -> alt.Chart:
    df = pd.DataFrame(contrib)
    df["변수"] = df["feature"].map(lambda f: report.FEATURE_KO.get(f, f))
    df["방향"] = df["contribution"].map(lambda c: "위험 증가" if c > 0 else "위험 감소")
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="점수 기여도"),
            y=alt.Y("변수:N", sort=alt.EncodingSortField("contribution", op="sum", order="descending"), title=None,
                    axis=alt.Axis(labelOverlap=False, labelLimit=220)),
            color=alt.Color("방향:N", scale=alt.Scale(domain=["위험 증가", "위험 감소"],
                                                      range=["#d9534f", "#5b8def"]), legend=alt.Legend(orient="bottom")),
            tooltip=["변수", alt.Tooltip("value:Q", title="값", format=",.3f"),
                     alt.Tooltip("contribution:Q", title="기여도", format="+.3f")],
        )
        .properties(height=alt.Step(34))
    )


def main():
    load_secrets()
    trained_on = ensure_model(str(MODEL_PATH))

    st.title("📑 공시 변화 기반 재무위험 조기경보 Agent")
    st.caption("OpenDART에서 공시를 직접 가져와 → 전년 대비 재무·문구 변화를 계산하고 → "
               "ML로 위험 점수를 매긴 뒤 → 근거 리포트를 씁니다.")

    with st.sidebar:
        st.header("설정")
        has_key = bool(os.getenv("DART_API_KEY"))
        if not has_key:
            try:
                from dotenv import load_dotenv

                load_dotenv(ROOT / ".env")
                has_key = bool(os.getenv("DART_API_KEY"))
            except ImportError:
                pass
        st.write("OpenDART 키:", "✅" if has_key else "❌ 없음")
        st.write("LLM 리포트:", "✅ 사용" if report.llm_enabled() else "템플릿 문장 (키 없음)")
        st.write("모델:", "⚠️ 합성 데이터 임시 모델" if trained_on == "synthetic" else f"✅ {trained_on}")
        use_text = st.toggle("공시 문구 변화 분석", value=True,
                             help="사업보고서 원문 2개 연도를 받아 비교합니다. 30초~1분 더 걸려요.")

    if trained_on == "synthetic":
        st.warning("지금 모델은 **합성 데이터로 만든 임시 모델**이라 점수는 화면 시연용입니다. "
                   "실데이터로 `python scripts/train.py --data data/processed/panel.csv` 를 돌리면 바뀝니다.")

    with st.form("query"):
        c1, c2, c3 = st.columns([3, 1, 1])
        corp = c1.text_input("기업명 또는 종목코드", placeholder="예: 삼성전자, 005930")
        year = c2.number_input("사업연도", min_value=2016, max_value=date.today().year - 1,
                               value=date.today().year - 2, step=1)
        c3.write("")
        submitted = c3.form_submit_button("분석", type="primary", width="stretch")

    if not submitted:
        st.info("기업과 사업연도를 넣고 **분석**을 누르세요. 사업연도 t의 사업보고서는 보통 t+1년 3월에 나옵니다.")
        return
    if not corp.strip():
        st.error("기업명이나 종목코드를 입력하세요.")
        return
    if not has_key:
        st.error("OpenDART 키가 없습니다. `.env` 또는 Streamlit Secrets에 `DART_API_KEY`를 넣어주세요.")
        return

    try:
        agent = RiskAgent(str(MODEL_PATH), dart=get_dart(), use_text=use_text)
        with st.status("Agent 실행 중…", expanded=True) as status:
            res = agent.run(corp.strip(), int(year), corp.strip(), on_step=lambda line: st.write("✔︎ " + line))
            status.update(label="완료", state="complete", expanded=False)
    except Exception as e:
        st.error(f"분석하지 못했습니다: {e}")
        return

    grade = report.risk_grade(res.score)
    m1, m2, m3 = st.columns(3)
    m1.metric("위험 점수", f"{res.score:.2f}")
    m2.metric("등급", grade)
    m3.metric("올해 처음 등장한 경고 문구", len(res.new_warning_terms))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("리포트")
        st.markdown(demote_headings(res.report))
    with right:
        st.subheader("점수에 기여한 변수")
        st.altair_chart(contributions_chart(res.contributions))
        if res.new_warning_terms:
            st.subheader("새 경고 문구")
            st.write(" · ".join(res.new_warning_terms))

    with st.expander("계산된 지표 전체"):
        rows = [{"변수": report.FEATURE_KO.get(k, k), "코드": k, "값": v} for k, v in res.feature_row.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Agent 실행 로그"):
        for line in res.log:
            st.text(line)
    st.caption("데이터 출처: 금융감독원 OpenDART. 본 화면은 공시 데이터 기반 참고 자료이며 투자 권유가 아닙니다.")


main()
