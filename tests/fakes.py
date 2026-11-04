"""OpenDartReader를 흉내 내는 가짜 클라이언트. API 키 없이 Agent 전체를 테스트한다."""

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


class FakeDart:
    def finstate_all(self, corp, year, reprt_code="11011", fs_div="CFS"):
        return fake_finstate()

    def report(self, corp, key_word, year, reprt_code="11011"):
        return pd.DataFrame([{"bsns_year": f"제10기({year})", "adt_opinion": "한정의견"}])

    def list(self, corp, start=None, end=None, kind="", final=True):
        y = int(start[:4])
        if kind == "A":
            return pd.DataFrame([{"report_nm": f"사업보고서 ({y - 1}.12)", "rcept_no": f"{y}0315000001"}])
        return pd.DataFrame({"report_nm": ["최대주주변경", "전환사채권발행결정", "[기재정정]사업보고서", "주주총회소집공고"]})

    def document(self, rcept_no):
        if rcept_no.startswith("2025"):  # 2024 사업보고서 (전년)
            return "<p>감사의견 적정의견입니다.</p><p>주석 1. 일반사항 회사는 정상 영업 중입니다.</p>"
        return ("<p>감사의견 한정의견입니다. 계속기업 가정에 중요한 불확실성이 존재합니다.</p>"
                "<p>주석 1. 일반사항 특수관계자 거래와 우발부채 소송이 있습니다.</p>")
