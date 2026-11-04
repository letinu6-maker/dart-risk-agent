"""③ 위험 예측: 수업 모델 3종(Logistic, Decision Tree, Random Forest), 시간 기준 분할, 불균형 지표."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FINANCIAL_FEATURES = [
    "capital_impairment",
    "debt_to_assets",
    "roa",
    "operating_margin",
    "ocf_to_assets",
    "net_loss",
    "ocf_negative",
    "revenue_growth",
    "roa_change",
    "debt_to_assets_change",
    "audit_non_clean",
    "n_major_holder_change",
    "n_convertible_bond",
    "n_rights_offering",
    "n_correction",
]
TEXT_FEATURES = ["audit_text_change", "notes_text_change", "warning_terms_added"]
LABEL = "label"
YEAR = "year"


def time_split(df: pd.DataFrame, test_start_year: int, year_col: str = YEAR):
    """test_start_year 이전 = train, 이후 = test. 무작위 분할은 미래 정보가 섞이므로 쓰지 않는다."""
    train = df[df[year_col] < test_start_year]
    test = df[df[year_col] >= test_start_year]
    assert len(train) and len(test), "train/test 중 하나가 비었습니다. test_start_year를 확인하세요."
    assert train[year_col].max() < test[year_col].min(), "시간 순서가 섞였습니다 (데이터 누수)."
    return train, test


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """불균형 대응으로 class_weight='balanced'. 결측은 중앙값 대체."""
    impute = ("impute", SimpleImputer(strategy="median"))
    return {
        "Logistic": Pipeline([impute, ("scale", StandardScaler()),
                              ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        "DecisionTree": Pipeline([impute, ("clf", DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=random_state))]),
        "RandomForest": Pipeline([impute, ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=random_state, n_jobs=-1))]),
    }


def evaluate(models: dict[str, Pipeline], X_train, y_train, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """모델별 Recall·Precision·PR-AUC·ROC-AUC. 학습도 여기서 한다."""
    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        pred = (p >= threshold).astype(int)
        rows.append({
            "model": name,
            "recall": recall_score(y_test, pred, zero_division=0),
            "precision": precision_score(y_test, pred, zero_division=0),
            "pr_auc": average_precision_score(y_test, p),
            "roc_auc": roc_auc_score(y_test, p),
            "positive_rate_test": float(np.mean(y_test)),
        })
    return pd.DataFrame(rows).set_index("model")


def feature_importance(model: Pipeline, features: list[str]) -> pd.Series:
    """Logistic은 표준화 계수, 트리 계열은 불순도 기반 중요도."""
    clf = model.named_steps["clf"]
    vals = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    return pd.Series(vals, index=features).sort_values(key=np.abs, ascending=False)


def explain_row(model: Pipeline, row: pd.DataFrame, features: list[str], top: int = 5) -> list[dict]:
    """한 기업의 점수에 기여한 변수 (Logistic 기준: 계수 × 표준화 값).

    리포트 단계에 넘기는 근거. Logistic이 아니면 중요도 × 값의 부호로 근사한다.
    """
    clf = model.named_steps["clf"]
    x = model.named_steps["impute"].transform(row[features])
    if "scale" in model.named_steps:
        x = model.named_steps["scale"].transform(x)
    if hasattr(clf, "coef_"):
        contrib = clf.coef_[0] * x[0]
    else:
        contrib = clf.feature_importances_ * np.sign(x[0])
    raw = row[features].iloc[0]
    order = np.argsort(-np.abs(contrib))[:top]
    return [{"feature": features[i], "value": float(raw.iloc[i]) if pd.notna(raw.iloc[i]) else None,
             "contribution": float(contrib[i])} for i in order]
