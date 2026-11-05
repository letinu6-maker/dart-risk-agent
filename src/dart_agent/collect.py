"""① 수집: OpenDART API에서 재무제표, 감사의견, 공시 목록, 원문을 가져온다.

모든 함수는 OpenDartReader 객체(`dart`)를 첫 인자로 받는다.
실제 API 응답 형태는 첫 수집 때 꼭 확인하고, 다르면 이 파일만 고치면 된다.
"""

from __future__ import annotations

import os
import re

import pandas as pd

ANNUAL_REPORT = "11011"  # 사업보고서


def get_client(api_key: str | None = None):
    """OpenDartReader 클라이언트. 키는 인자 → 환경변수 DART_API_KEY 순으로 찾는다."""
    # 지연 import: 테스트는 API 없이 돌아가야 함
    try:
        from opendartreader import OpenDartReader  # 0.3 이상
    except ImportError:
        import OpenDartReader  # 0.2.x (모듈 자체가 호출 가능)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = api_key or os.getenv("DART_API_KEY")
    if not key:
        raise RuntimeError("DART_API_KEY가 없습니다. .env에 인증키를 넣어주세요.")
    return OpenDartReader(key)


def fetch_financials(dart, corp: str, year: int) -> pd.DataFrame:
    """전체 재무제표. 연결(CFS)이 없으면 별도(OFS)로 다시 요청한다."""
    for fs_div in ("CFS", "OFS"):
        df = dart.finstate_all(corp, year, reprt_code=ANNUAL_REPORT, fs_div=fs_div)
        if df is not None and len(df) > 0:
            df = df.copy()
            df["fs_div"] = fs_div
            return df
    return pd.DataFrame()


def fetch_audit_opinion(dart, corp: str, year: int) -> str | None:
    """사업보고서의 '회계감사인의 명칭 및 감사의견'에서 당기 감사의견 문자열."""
    df = dart.report(corp, "회계감사", year, reprt_code=ANNUAL_REPORT)
    if df is None or len(df) == 0 or "adt_opinion" not in df.columns:
        return None
    # 첫 행이 당기인 경우가 일반적. bsns_year 컬럼이 있으면 그걸로 고른다.
    if "bsns_year" in df.columns:
        cur = df[df["bsns_year"].astype(str).str.contains(str(year))]
        if len(cur):
            df = cur
    return str(df.iloc[0]["adt_opinion"])


def fetch_filings(dart, corp: str, start: str, end: str) -> pd.DataFrame:
    """기간 내 공시 목록 (최대주주 변경, CB 발행, 유상증자 등 건수 집계용). 날짜는 'YYYY-MM-DD'."""
    df = dart.list(corp, start=start, end=end, final=False)
    return df if df is not None else pd.DataFrame()


def find_annual_report_rcept_no(dart, corp: str, year: int) -> str | None:
    """t년 사업보고서 접수번호. 사업보고서는 보통 t+1년 3월에 제출된다."""
    df = dart.list(corp, start=f"{year + 1}-01-01", end=f"{year + 1}-12-31", kind="A", final=True)
    if df is None or len(df) == 0:
        return None
    hit = df[df["report_nm"].str.contains(rf"사업보고서\s*\({year}\.12\)", regex=True)]
    if len(hit) == 0:
        hit = df[df["report_nm"].str.contains("사업보고서")]
    return str(hit.iloc[0]["rcept_no"]) if len(hit) else None


def fetch_document_text(dart, rcept_no: str) -> str:
    """공시 원문(XML/HTML)을 태그 없는 텍스트로."""
    raw = dart.document(rcept_no)
    return strip_markup(raw or "")


def strip_markup(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()
