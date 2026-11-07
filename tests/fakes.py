"""OpenDartReader를 흉내 내는 가짜 클라이언트. API 키 없이 Agent 전체를 테스트한다.

응답 컬럼·형태는 실제 OpenDART 응답(scripts/check_api.py 결과)을 따른다.
"""

import pandas as pd


def fake_finstate(equity=50.0, capital=100.0, net_income=-30.0, ocf=-10.0):
    rows = [
        ("ifrs-full_Assets", "자산총계", 1000, 1100),
        ("ifrs-full_Liabilities", "부채총계", 950, 900),
        ("ifrs-full_Equity", "자본총계", equity, 200),
        ("ifrs-full_IssuedCapital", "자본금", capital, 100),
        ("ifrs-full_Revenue", "매출액", 500, 800),
        ("dart_OperatingIncomeLoss", "영업이익(손실)", -40, 20),
        ("ifrs-full_ProfitLoss", "당기순이익(손실)", net_income, 10),
        ("ifrs-full_CashFlowsFromUsedInOperatingActivities", "영업활동현금흐름", ocf, 30),
    ]
    return pd.DataFrame(
        [{"account_id": a, "account_nm": n, "thstrm_amount": f"{c:,.0f}", "frmtrm_amount": f"{p:,.0f}"}
         for a, n, c, p in rows]
    )


def fake_doc(audit: str, notes: str, year: int) -> list[str]:
    """본문 + 첨부 감사보고서 두 파일. 목차에 '감사의견'·'주석' 단어를 일부러 넣어 둔다."""
    body = (f"<DOCUMENT><TITLE ATOC='Y'>I. 회사의 개요</TITLE><P>감사의견 및 주석사항을 참고하시기 바랍니다. "
            f"{year}년 매출 {year * 7:,}백만원</P>"
            f"<TITLE ATOC='Y'>3. 연결재무제표 주석</TITLE><P>{notes} 단위 {year * 3:,}백만원</P></DOCUMENT>")
    attach = (f"<DOCUMENT><TITLE ATOC='Y'>독립된 감사인의 감사보고서</TITLE><P>{audit}</P>"
              f"<TITLE ATOC='Y'>(첨부)연결재무제표</TITLE><P>재무상태표 {year * 11:,}</P></DOCUMENT>")
    return [body, attach]


class FakeDart:
    """listed_from: 기업별 첫 사업연도 (그 전 연도는 빈 응답). quota: 이 횟수를 넘으면 모든 응답이 빈 값."""

    def __init__(self, listed_from=None, quota=None):
        self.listed_from = listed_from or {}
        self.quota = quota
        self.calls = 0
        self.corp_codes = pd.DataFrame({"corp_code": ["00000101", "00000102", "00000103"],
                                        "corp_name": ["정상가", "정상나", "정상다"],
                                        "stock_code": ["000101", "000102", "000103"]})

    def _over(self):
        self.calls += 1
        return self.quota is not None and self.calls > self.quota

    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        if self._over() or year < self.listed_from.get(corp, 0):
            return pd.DataFrame()
        return fake_finstate().assign(rcept_no=f"{year + 1}0311000001")

    def report(self, corp, key_word, year, reprt_code="11011"):
        return pd.DataFrame([
            {"bsns_year": "제10기\n(당기)", "adt_opinion": "한정의견",
             "emphs_matter": "계속기업 가정에 관한 중요한 불확실성", "adt_reprt_spcmnt_matter": "-"},
            {"bsns_year": "제9기\n(전기)", "adt_opinion": "적정의견", "emphs_matter": "-",
             "adt_reprt_spcmnt_matter": "-"},
        ])

    def list(self, corp=None, start=None, end=None, kind="", final=True):
        y = int(start[:4])
        if kind == "I":  # 거래소공시 (기업 지정 없이 월 단위)
            if start[5:7] != "04":
                return pd.DataFrame()
            return pd.DataFrame({
                "corp_code": ["00000201", "00000202", "00000101"],
                "corp_name": ["부실가", "케이스팩1호", "정상가"],
                "stock_code": ["000201", "000202", "000101"],
                "report_nm": ["관리종목지정(감사의견 비적정설)", "상장폐지결정(존속기간 만료)", "관리종목지정해제"],
                "rcept_dt": [f"{y}0402", f"{y}0403", f"{y}0404"],
                "rcept_no": [f"{y}0402000001", f"{y}0403000001", f"{y}0404000001"],
            })
        if kind == "A":
            return pd.DataFrame([{"report_nm": f"사업보고서 ({y - 1}.12)", "rcept_no": f"{y}0315000001"}])
        return pd.DataFrame({"report_nm": ["최대주주변경", "전환사채권발행결정", "[기재정정]사업보고서", "주주총회소집공고"]})

    def document_all(self, rcept_no):
        year = int(rcept_no[:4]) - 1
        if year < 2025:  # 전년
            return fake_doc("감사의견 적정의견입니다.", "일반사항 회사는 정상 영업 중입니다.", year)
        return fake_doc("감사의견 한정의견입니다. 계속기업 가정에 중요한 불확실성이 존재합니다.",
                        "일반사항 특수관계자 거래와 우발부채 소송이 있습니다.", year)
