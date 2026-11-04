"""② 변화 탐지 (정형): 재무제표 → 재무지표, 감사의견·공시 이벤트 → 변수.

OpenDART 전체 재무제표(finstate_all) 컬럼을 가정한다:
account_id, account_nm, thstrm_amount(당기), frmtrm_amount(전기)
"""

from __future__ import annotations

import math

import pandas as pd

# 표준 계정: (account_id 후보, account_nm 후보)
ACCOUNTS = {
    "assets": (["ifrs-full_Assets"], ["자산총계"]),
    "liabilities": (["ifrs-full_Liabilities"], ["부채총계"]),
    "equity": (["ifrs-full_Equity"], ["자본총계"]),
    "issued_capital": (["ifrs-full_IssuedCapital"], ["자본금"]),
    "revenue": (["ifrs-full_Revenue"], ["매출액", "수익(매출액)", "영업수익"]),
    "operating_income": (["dart_OperatingIncomeLoss"], ["영업이익", "영업이익(손실)"]),
    "net_income": (["ifrs-full_ProfitLoss"], ["당기순이익", "당기순이익(손실)"]),
    "operating_cf": (["ifrs-full_CashFlowsFromUsedInOperatingActivities"], ["영업활동현금흐름", "영업활동으로 인한 현금흐름"]),
}

# 공시 제목 키워드 → 이벤트 변수
EVENT_KEYWORDS = {
    "n_major_holder_change": ["최대주주변경", "최대주주 변경"],
    "n_convertible_bond": ["전환사채", "신주인수권부사채", "교환사채"],
    "n_rights_offering": ["유상증자"],
    "n_correction": ["[기재정정]", "정정"],
}

NON_CLEAN_OPINIONS = ("한정", "부적정", "거절")


def _to_number(v) -> float:
    if v is None:
        return math.nan
    s = str(v).replace(",", "").strip()
    if s in ("", "-", "nan", "None"):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def extract_accounts(fs: pd.DataFrame, column: str = "thstrm_amount") -> dict[str, float]:
    """재무제표에서 표준 계정 금액을 뽑는다. 없는 계정은 NaN."""
    out = {}
    for key, (ids, names) in ACCOUNTS.items():
        val = math.nan
        if len(fs) and "account_id" in fs.columns:
            hit = fs[fs["account_id"].isin(ids)]
            if len(hit):
                val = _to_number(hit.iloc[0][column])
        if math.isnan(val) and len(fs) and "account_nm" in fs.columns:
            nm = fs["account_nm"].astype(str).str.replace(" ", "")
            hit = fs[nm.isin([n.replace(" ", "") for n in names])]
            if len(hit):
                val = _to_number(hit.iloc[0][column])
        out[key] = val
    return out


def _div(a: float, b: float) -> float:
    if b is None or math.isnan(a) or math.isnan(b) or b == 0:
        return math.nan
    return a / b


def compute_ratios(cur: dict[str, float], prev: dict[str, float] | None = None) -> dict[str, float]:
    """재무지표. prev(전기)가 있으면 변화율도 계산."""
    impairment = _div(cur["issued_capital"] - cur["equity"], cur["issued_capital"])
    r = {
        # 자본잠식률: (자본금 - 자본총계) / 자본금. 음수면 잠식 없음 → 0
        "capital_impairment": max(impairment, 0.0) if not math.isnan(impairment) else math.nan,
        "debt_to_assets": _div(cur["liabilities"], cur["assets"]),
        "roa": _div(cur["net_income"], cur["assets"]),
        "operating_margin": _div(cur["operating_income"], cur["revenue"]),
        "ocf_to_assets": _div(cur["operating_cf"], cur["assets"]),
        "net_loss": float(cur["net_income"] < 0) if not math.isnan(cur["net_income"]) else math.nan,
        "ocf_negative": float(cur["operating_cf"] < 0) if not math.isnan(cur["operating_cf"]) else math.nan,
    }
    if prev is not None:
        r["revenue_growth"] = _div(cur["revenue"] - prev["revenue"], abs(prev["revenue"]))
        r["roa_change"] = r["roa"] - _div(prev["net_income"], prev["assets"])
        r["debt_to_assets_change"] = r["debt_to_assets"] - _div(prev["liabilities"], prev["assets"])
    return r


def audit_non_clean(opinion: str | None) -> float:
    """감사의견 비적정(한정·부적정·의견거절)이면 1, 적정이면 0, 정보 없으면 NaN."""
    if opinion is None or str(opinion).strip() in ("", "-", "nan"):
        return math.nan
    return float(any(k in str(opinion) for k in NON_CLEAN_OPINIONS))


def count_events(filings: pd.DataFrame) -> dict[str, float]:
    """공시 목록에서 이벤트 건수."""
    titles = filings["report_nm"].astype(str) if len(filings) and "report_nm" in filings.columns else pd.Series([], dtype=str)
    return {
        name: float(sum(any(k in t for k in kws) for t in titles))
        for name, kws in EVENT_KEYWORDS.items()
    }


def build_feature_row(fs: pd.DataFrame, audit_opinion: str | None, filings: pd.DataFrame) -> dict[str, float]:
    """한 기업·한 연도의 변수 한 줄. 전기 금액은 같은 재무제표의 frmtrm_amount를 쓴다."""
    cur = extract_accounts(fs, "thstrm_amount")
    prev = extract_accounts(fs, "frmtrm_amount") if "frmtrm_amount" in fs.columns else None
    row = compute_ratios(cur, prev)
    row["audit_non_clean"] = audit_non_clean(audit_opinion)
    row.update(count_events(filings))
    return row
