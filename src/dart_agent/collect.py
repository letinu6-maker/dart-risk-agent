"""① 수집: OpenDART API에서 재무제표, 감사의견, 공시 목록, 원문을 가져온다.

모든 함수는 OpenDartReader 객체(`dart`)를 첫 인자로 받는다.
응답 형태는 scripts/check_api.py로 확인했다 (opendartreader 0.3.3, 2026-11).
"""

from __future__ import annotations

import os
import re

import pandas as pd

ANNUAL_REPORT = "11011"  # 사업보고서
TITLE_RE = re.compile(r"<TITLE[^>]*>(.*?)</TITLE>", re.S | re.I)


HTTP_TIMEOUT = 60  # 초


def install_default_timeout(seconds: float = HTTP_TIMEOUT) -> None:
    """OpenDartReader는 requests.get에 timeout을 주지 않는다.

    노트북을 덮거나 와이파이가 끊기면 응답 없는 요청을 영원히 기다리며 멈춘다.
    모든 requests 호출에 기본 timeout을 걸어, 끊기면 오류로 넘어가게 한다
    (그 행은 저장되지 않아 다음 실행 때 다시 받는다).
    """
    import requests

    if getattr(requests.sessions.Session.request, "_dart_timeout", False):
        return
    original = requests.sessions.Session.request

    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = seconds
        return original(self, method, url, **kwargs)

    request._dart_timeout = True
    requests.sessions.Session.request = request


def get_client(api_key: str | None = None):
    """OpenDartReader 클라이언트. 키는 인자 → 환경변수 DART_API_KEY 순으로 찾는다."""
    install_default_timeout()
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


def fetch_audit_info(dart, corp: str, year: int) -> dict:
    """'회계감사인의 명칭 및 감사의견'에서 당기 행의 감사의견과 강조사항.

    실제 응답의 bsns_year는 '제56기\\n(당기)' 형태라 연도가 아니라 '당기'로 고른다.
    """
    df = dart.report(corp, "회계감사", year, reprt_code=ANNUAL_REPORT)
    if df is None or len(df) == 0 or "adt_opinion" not in df.columns:
        return {"opinion": None, "emphasis": None}
    if "bsns_year" in df.columns:
        cur = df[df["bsns_year"].astype(str).str.contains("당기")]
        if len(cur):
            df = cur
    r = df.iloc[0]
    emphasis = " ".join(str(r[c]) for c in ("emphs_matter", "adt_reprt_spcmnt_matter")
                        if c in df.columns and pd.notna(r[c]))
    return {"opinion": str(r["adt_opinion"]), "emphasis": emphasis}


def fetch_audit_opinion(dart, corp: str, year: int) -> str | None:
    return fetch_audit_info(dart, corp, year)["opinion"]


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


def _document_files(dart, rcept_no: str) -> list[str]:
    """document()는 본문 파일 하나만 주고 감사보고서는 첨부 파일에 있으므로 document_all()을 쓴다."""
    if hasattr(dart, "document_all"):
        return [f for f in (dart.document_all(rcept_no) or []) if f]
    return [dart.document(rcept_no) or ""]


def split_sections(raw: str) -> list[tuple[str, str]]:
    """DART 원문 XML을 <TITLE> 기준 (제목, 본문 텍스트) 목록으로 나눈다."""
    parts = TITLE_RE.split(raw)
    out = []
    if strip_markup(parts[0]):
        out.append(("", strip_markup(parts[0])))
    for i in range(1, len(parts), 2):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((strip_markup(parts[i]), strip_markup(body)))
    return out


def fetch_document_sections(dart, rcept_no: str) -> list[tuple[str, str]]:
    """본문과 첨부(감사보고서 등)를 모두 섹션으로 나눠 이어 붙인다."""
    secs: list[tuple[str, str]] = []
    for raw in _document_files(dart, rcept_no):
        secs += split_sections(raw)
    return secs


def fetch_document_text(dart, rcept_no: str) -> str:
    """원문 전체를 태그 없는 텍스트 하나로."""
    return strip_markup("\n".join(_document_files(dart, rcept_no)))


def strip_markup(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()
