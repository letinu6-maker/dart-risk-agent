"""합성 패널 데이터. 실제 데이터 수집 전에 노트북·테스트를 끝까지 돌려보기 위한 것.

실제 결과로 절대 보고하지 말 것. 노트북에서 실데이터로 바꾸면 이 모듈은 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FINANCIAL_FEATURES, LABEL, TEXT_FEATURES, YEAR


def make_panel(n_firms: int = 400, years: range = range(2016, 2025), event_rate: float = 0.05,
               seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_firms * len(years)
    risk = rng.normal(size=n)  # 잠재 위험도
    df = pd.DataFrame({
        "corp_code": np.repeat([f"{i:06d}" for i in range(n_firms)], len(years)),
        YEAR: np.tile(list(years), n_firms),
        "capital_impairment": np.clip(0.3 * risk + rng.normal(0, 0.3, n), 0, None),
        "debt_to_assets": np.clip(0.5 + 0.15 * risk + rng.normal(0, 0.15, n), 0, 2),
        "roa": -0.03 * risk + rng.normal(0.02, 0.05, n),
        "operating_margin": -0.05 * risk + rng.normal(0.05, 0.1, n),
        "ocf_to_assets": -0.03 * risk + rng.normal(0.04, 0.05, n),
        "revenue_growth": -0.1 * risk + rng.normal(0.05, 0.2, n),
        "roa_change": -0.02 * risk + rng.normal(0, 0.03, n),
        "debt_to_assets_change": 0.03 * risk + rng.normal(0, 0.05, n),
        "audit_non_clean": (risk + rng.normal(0, 1, n) > 2.3).astype(float),
        "n_major_holder_change": rng.poisson(np.exp(0.5 * risk - 2)),
        "n_convertible_bond": rng.poisson(np.exp(0.6 * risk - 1.5)),
        "n_rights_offering": rng.poisson(np.exp(0.4 * risk - 1.8)),
        "n_correction": rng.poisson(np.exp(0.3 * risk - 1)),
        # 문구 변화 변수: 위험과 약하게 연관 → "넣으면 오르는가" 실험이 의미 있게
        "audit_text_change": np.clip(0.05 * risk + rng.normal(0.1, 0.05, n), 0, 1),
        "notes_text_change": np.clip(0.06 * risk + rng.normal(0.15, 0.06, n), 0, 1),
        "warning_terms_added": rng.poisson(np.exp(0.5 * risk - 0.5)),
    })
    df["net_loss"] = (df["roa"] < 0).astype(float)
    df["ocf_negative"] = (df["ocf_to_assets"] < 0).astype(float)
    # 결측 일부 (실데이터처럼)
    for c in ["revenue_growth", "operating_margin"]:
        df.loc[rng.random(n) < 0.05, c] = np.nan
    thr = np.quantile(risk + rng.normal(0, 0.7, n), 1 - event_rate)
    df[LABEL] = ((risk + rng.normal(0, 0.7, n)) > thr).astype(int)
    return df[["corp_code", YEAR] + FINANCIAL_FEATURES + TEXT_FEATURES + [LABEL]]
