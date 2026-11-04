"""가짜 OpenDART로 Agent 4단계 전체를 끝까지 돌린다."""

import joblib

from dart_agent import model as M
from dart_agent.agent import RiskAgent
from dart_agent.synthetic import make_panel
from fakes import FakeDart


def test_agent_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # 템플릿 리포트
    df = make_panel(n_firms=200)
    feats = M.FINANCIAL_FEATURES + M.TEXT_FEATURES
    m = M.build_models()["Logistic"].fit(df[feats], df[M.LABEL])
    path = tmp_path / "m.joblib"
    joblib.dump({"model": m, "features": feats}, path)

    res = RiskAgent(str(path), dart=FakeDart()).run("000001", 2025, "테스트기업")

    assert 0 <= res.score <= 1
    assert res.feature_row["audit_non_clean"] == 1.0
    assert res.feature_row["audit_text_change"] > 0
    assert "테스트기업" in res.report and "투자 권유가 아닙니다" in res.report
    assert "계속기업" in res.report  # 새 경고 문구가 리포트 근거로 들어감
    assert [l.split(":")[0] for l in res.log] == ["수집", "문구 변화", "위험 점수"]
