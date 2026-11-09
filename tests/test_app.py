"""Streamlit 데모 화면을 가짜 OpenDART로 끝까지 실행한다."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path):
    from dart_agent import collect
    from fakes import FakeDart

    monkeypatch.setenv("DART_API_KEY", "fake")
    monkeypatch.setenv("DART_AGENT_MODEL", str(tmp_path / "m.joblib"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(collect, "get_client", lambda *a, **k: FakeDart())
    at = st_testing.AppTest.from_file(APP, default_timeout=60)
    return at


def test_initial_screen(app):
    app.run()
    assert not app.exception
    assert any("분석" in i.value for i in app.info)


def test_analysis_flow(app):
    app.run()
    app.text_input[0].input("테스트기업")
    app.number_input[0].set_value(2024)
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    labels = [m.label for m in app.metric]
    assert labels == ["위험 점수", "등급", "올해 처음 등장한 경고 문구"]
    assert any("투자 권유가 아닙니다" in md.value for md in app.markdown)
