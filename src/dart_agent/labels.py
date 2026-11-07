"""라벨: OpenDART 거래소공시(pblntf_ty='I')에서 위험 사건을 뽑고, 기업×연도 행에 붙인다.

데이터 출처를 OpenDART 하나로 통일한다. 거래소가 기업 대신 올리는 공시
(관리종목 지정, 상장폐지 사유 발생 등)가 DART에 남아 있어서 KRX KIND를 따로 긁지 않아도 된다.

누수 방지 원칙
- 예측 시점 = t년 사업보고서 제출일(rcept_no 앞 8자리).
- 라벨 창 = (제출일 + gap, 제출일 + gap + horizon]. 기본 gap 30일, horizon 365일.
- 제출일 직후 gap 안에 사건이 나면 그 사건은 '이미 공개된 감사의견 때문에 바로 생긴 것'이라
  예측 문제가 아니다. 그런 행은 0도 1도 아니므로 버린다 (label = NaN → 학습 제외).
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

EVENT_PATTERNS = {
    "admin_issue": re.compile(r"관리종목\s*지정"),
    "delisting": re.compile(r"상장폐지\s*(?:사유\s*발생|결정)|상장적격성\s*실질\s*심사\s*대상"),
}
# 부실이 아닌 폐지(스팩 존속기간 만료, 합병, 자진 상장폐지 등)와 해제·취소 공시는 뺀다
EXCLUDE_TITLE = re.compile(r"해제|취소|철회|유예|합병|자진|지주회사|완전자회사|존속기간|예고|안내")
EXCLUDE_NAME = re.compile(r"스팩|기업인수목적|SPAC", re.I)


def month_ranges(start_year: int, end_year: int):
    """OpenDART 공시검색은 기업을 지정하지 않으면 기간을 3개월 이하로 묶어야 한다. 한 달씩 자른다."""
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            first = date(y, m, 1)
            last = (pd.Timestamp(first) + pd.offsets.MonthEnd(0)).date()
            yield first.isoformat(), last.isoformat()


def classify_events(filings: pd.DataFrame) -> pd.DataFrame:
    """거래소공시 목록 → 위험 사건만 남기고 event_type을 붙인다."""
    if filings is None or len(filings) == 0:
        return pd.DataFrame(columns=["corp_code", "corp_name", "stock_code", "event_date",
                                     "event_type", "report_nm", "rcept_no"])
    df = filings.copy()
    title = df["report_nm"].astype(str)
    df["event_type"] = None
    for name, pat in EVENT_PATTERNS.items():
        df.loc[df["event_type"].isna() & title.str.contains(pat), "event_type"] = name
    keep = df["event_type"].notna() & ~title.str.contains(EXCLUDE_TITLE)
    if "corp_name" in df.columns:
        keep &= ~df["corp_name"].astype(str).str.contains(EXCLUDE_NAME)
    df = df[keep].copy()
    df["event_date"] = pd.to_datetime(df["rcept_dt"].astype(str), format="%Y%m%d").dt.date.astype(str)
    cols = ["corp_code", "corp_name", "stock_code", "event_date", "event_type", "report_nm", "rcept_no"]
    return df[[c for c in cols if c in df.columns]].drop_duplicates(["corp_code", "event_date", "event_type"])


def sample_universe(corp_codes: pd.DataFrame, events: pd.DataFrame, n_normal: int, seed: int = 42) -> pd.DataFrame:
    """학습 대상 기업 = 사건 기업 전부 + 현재 상장 기업 중 무작위 n_normal개.

    사건 기업을 따로 넣는 이유: 상장폐지된 기업은 현재 상장 목록에 없어서,
    상장 목록만 쓰면 살아남은 기업만 남는다 (생존 편향).
    """
    listed = corp_codes[corp_codes["stock_code"].fillna("").astype(str).str.strip() != ""]
    listed = listed[~listed["corp_name"].astype(str).str.contains(EXCLUDE_NAME)]
    ev_codes = set(events["corp_code"].astype(str))
    normal = listed[~listed["corp_code"].astype(str).isin(ev_codes)]
    normal = normal.sample(n=min(n_normal, len(normal)), random_state=seed)
    ev = events.drop_duplicates("corp_code")[["corp_code", "corp_name"]].assign(source="event")
    return pd.concat([ev, normal[["corp_code", "corp_name"]].assign(source="normal")], ignore_index=True)


def attach_labels(panel: pd.DataFrame, events: pd.DataFrame, gap_days: int = 30,
                  horizon_days: int = 365) -> pd.DataFrame:
    """panel(corp_code, report_date='YYYYMMDD') 각 행에 label 0/1/NaN을 붙인다."""
    ev = events.assign(corp_code=events["corp_code"].astype(str),
                       event_date=pd.to_datetime(events["event_date"]))
    by_corp = ev.groupby("corp_code")["event_date"].apply(lambda s: np.sort(s.values)).to_dict()
    labels, first_event = [], []
    for corp, rd in zip(panel["corp_code"].astype(str), pd.to_datetime(panel["report_date"].astype(str))):
        dates = by_corp.get(corp)
        if dates is None:
            labels.append(0.0)
            first_event.append(None)
            continue
        after = dates[dates > np.datetime64(rd)]
        gap_end = np.datetime64(rd + pd.Timedelta(days=gap_days))
        win_end = np.datetime64(rd + pd.Timedelta(days=gap_days + horizon_days))
        if len(after) and after[0] <= gap_end:
            labels.append(np.nan)  # 이미 공개된 정보로 바로 생긴 사건 → 학습 제외
        elif len(after) and after[0] <= win_end:
            labels.append(1.0)
        else:
            labels.append(0.0)
        first_event.append(str(pd.Timestamp(after[0]).date()) if len(after) else None)
    return panel.assign(label=labels, next_event_date=first_event)
