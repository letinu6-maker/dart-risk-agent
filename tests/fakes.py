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
    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        return fake_finstate()

    def report(self, corp, key_word, year, reprt_code="11011"):
        return pd.DataFrame([
            {"bsns_year": "제10기\n(당기)", "adt_opinion": "한정의견",
             "emphs_matter": "계속기업 가정에 관한 중요한 불확실성", "adt_reprt_spcmnt_matter": "-"},
            {"bsns_year": "제9기\n(전기)", "adt_opinion": "적정의견", "emphs_matter": "-",
             "adt_reprt_spcmnt_matter": "-"},
        ])

    def list(self, corp, start=None, end=None, kind="", final=True):
        y = int(start[:4])
        if kind == "A":
            return pd.DataFrame([{"report_nm": f"사업보고서 ({y - 1}.12)", "rcept_no": f"{y}0315000001"}])
        return pd.DataFrame({"report_nm": ["최대주주변경", "전환사채권발행결정", "[기재정정]사업보고서", "주주총회소집공고"]})

    def document_all(self, rcept_no):
        year = int(rcept_no[:4]) - 1
        if year < 2025:  # 전년
            return fake_doc("감사의견 적정의견입니다.", "일반사항 회사는 정상 영업 중입니다.", year)
        return fake_doc("감사의견 한정의견입니다. 계속기업 가정에 중요한 불확실성이 존재합니다.",
                        "일반사항 특수관계자 거래와 우발부채 소송이 있습니다.", year)
