import pytest

from dart_agent import model as M
from dart_agent.synthetic import make_panel


def test_time_split_has_no_overlap():
    df = make_panel(n_firms=50)
    train, test = M.time_split(df, 2022)
    assert train["year"].max() < test["year"].min()


def test_time_split_rejects_empty():
    with pytest.raises(AssertionError):
        M.time_split(make_panel(n_firms=10), 2100)


def test_three_models_beat_random():
    df = make_panel(n_firms=300)
    feats = M.FINANCIAL_FEATURES + M.TEXT_FEATURES
    train, test = M.time_split(df, 2022)
    res = M.evaluate(M.build_models(), train[feats], train[M.LABEL], test[feats], test[M.LABEL])
    assert list(res.index) == ["Logistic", "DecisionTree", "RandomForest"]
    # PR-AUC는 무작위면 양성 비율 수준
    assert (res["pr_auc"] > res["positive_rate_test"] * 2).all()


def test_explain_row_returns_top_features():
    df = make_panel(n_firms=200)
    feats = M.FINANCIAL_FEATURES
    m = M.build_models()["Logistic"].fit(df[feats], df[M.LABEL])
    out = M.explain_row(m, df.iloc[[0]], feats, top=3)
    assert len(out) == 3 and {"feature", "value", "contribution"} <= set(out[0])
