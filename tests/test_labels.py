import math

import pandas as pd

from dart_agent import labels


def filings():
    rows = [
        ("00000001", "가나다", "관리종목지정(감사의견 비적정설)", "20240402"),
        ("00000002", "라마바", "상장폐지사유발생", "20230510"),
        ("00000003", "사아자", "관리종목지정해제", "20230601"),          # 해제 → 제외
        ("00000004", "에이스팩3호", "상장폐지결정(존속기간 만료)", "20230701"),  # 스팩 → 제외
        ("00000005", "차카타", "상장폐지결정(합병)", "20230801"),          # 합병 → 제외
        ("00000006", "파하", "주권매매거래정지", "20230901"),              # 사건 아님
        ("00000007", "거너더", "상장적격성 실질심사 대상 결정", "20231001"),
    ]
    return pd.DataFrame(rows, columns=["corp_code", "corp_name", "report_nm", "rcept_dt"]).assign(
        stock_code="000000", rcept_no=lambda d: d.rcept_dt + "000001")


def test_classify_keeps_only_distress_events():
    ev = labels.classify_events(filings())
    assert sorted(ev.corp_code) == ["00000001", "00000002", "00000007"]
    assert dict(zip(ev.corp_code, ev.event_type)) == {
        "00000001": "admin_issue", "00000002": "delisting", "00000007": "delisting"}
    assert ev.loc[ev.corp_code == "00000001", "event_date"].item() == "2024-04-02"


def test_month_ranges_cover_year():
    r = list(labels.month_ranges(2024, 2024))
    assert len(r) == 12 and r[1] == ("2024-02-01", "2024-02-29")


def test_attach_labels_window_and_gap():
    events = pd.DataFrame({"corp_code": ["A", "B", "C"],
                           "event_date": ["2024-09-01", "2024-03-25", "2026-01-10"]})
    panel = pd.DataFrame({"corp_code": ["A", "B", "C", "D", "A"],
                          "report_date": ["20240315", "20240315", "20240315", "20240315", "20250315"]})
    out = labels.attach_labels(panel, events, gap_days=30, horizon_days=365)
    lab = out["label"].tolist()
    assert lab[0] == 1.0                 # 제출 후 30일~395일 안 사건
    assert math.isnan(lab[1])            # 제출 10일 뒤 사건 = 이미 공개된 정보로 생긴 사건 → 제외
    assert lab[2] == 0.0                 # 창 밖(1년 넘게 뒤)
    assert lab[3] == 0.0                 # 사건 없는 기업
    assert lab[4] == 0.0                 # 사건 이후 연도의 행


def test_universe_adds_event_firms_and_excludes_spac():
    corp_codes = pd.DataFrame({"corp_code": ["N1", "N2", "N3", "S1"],
                               "corp_name": ["정상1", "정상2", "정상3", "하나스팩"],
                               "stock_code": ["1", "2", "3", "4"]})
    events = pd.DataFrame({"corp_code": ["E1"], "corp_name": ["폐지기업"]})
    u = labels.sample_universe(corp_codes, events, n_normal=2, seed=0)
    assert "E1" in set(u.corp_code)                   # 상장 목록에 없어도 사건 기업은 포함
    assert "S1" not in set(u.corp_code)
    assert (u.source == "normal").sum() == 2
