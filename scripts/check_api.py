"""OpenDART 실제 응답이 collect.py·features.py 가정과 맞는지 한 번에 점검.

    python scripts/check_api.py                 # 삼성전자 2024
    python scripts/check_api.py --corp 035720 --year 2024

각 단계를 [OK]/[확인 필요]/[실패]로 출력한다. 결과 전체를 복사해서 공유하면 된다.
API 키는 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent import collect, features, text_diff  # noqa: E402


def step(name):
    print(f"\n=== {name} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", default="005930")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--skip-text", action="store_true", help="원문 다운로드 생략 (느림)")
    a = ap.parse_args()
    dart = collect.get_client()
    fs = opinion = filings = None

    step("1. 전체 재무제표 finstate_all")
    try:
        fs = collect.fetch_financials(dart, a.corp, a.year)
        print(f"{len(fs)}행, fs_div={fs['fs_div'].iloc[0] if len(fs) else '-'}")
        print("컬럼:", list(fs.columns))
        accts = features.extract_accounts(fs)
        for k, v in accts.items():
            mark = "OK" if not math.isnan(v) else "확인 필요"
            print(f"  [{mark}] {k:18s} {v:,.0f}" if mark == "OK" else f"  [{mark}] {k:18s} 못 찾음")
        missing = [k for k, v in accts.items() if math.isnan(v)]
        if missing:
            print("  ↳ 못 찾은 계정 후보 (account_id / account_nm):")
            kw = {"revenue": "매출|수익", "operating_income": "영업", "net_income": "순이익|순손익",
                  "operating_cf": "영업활동", "issued_capital": "자본금", "equity": "자본총계"}
            for k in missing:
                hit = fs[fs["account_nm"].astype(str).str.contains(kw.get(k, k))]
                for _, r in hit.head(5).iterrows():
                    print(f"     {k}: {r.get('account_id')} / {r.get('account_nm')} / {r.get('sj_nm', '')}")
    except Exception:
        print("[실패]"); traceback.print_exc()

    step("2. 감사의견 report('회계감사')")
    try:
        raw = dart.report(a.corp, "회계감사", a.year)
        print("컬럼:", list(raw.columns) if raw is not None else None)
        if raw is not None and len(raw):
            cols = [c for c in ("bsns_year", "adtor", "adt_opinion") if c in raw.columns]
            print(raw[cols].head(4).to_string())
        audit = collect.fetch_audit_info(dart, a.corp, a.year)
        opinion, emphasis = audit["opinion"], audit["emphasis"]
        print(f"[{'OK' if opinion else '확인 필요'}] 선택된 당기 감사의견: {opinion!r} → 비적정={features.audit_non_clean(opinion)}")
        print(f"  강조사항·특기사항: {(emphasis or '')[:150]!r} → 계속기업 언급={features.going_concern_flag(emphasis)}")
    except Exception:
        print("[실패]"); traceback.print_exc()

    step("3. 공시 목록 list")
    try:
        filings = collect.fetch_filings(dart, a.corp, f"{a.year}-01-01", f"{a.year}-12-31")
        print(f"{len(filings)}건, 컬럼:", list(filings.columns))
        print(features.count_events(filings))
    except Exception:
        print("[실패]"); traceback.print_exc()

    step("4. 변수 한 줄 build_feature_row")
    if fs is not None:
        row = features.build_feature_row(fs, opinion, filings if filings is not None else fs.iloc[0:0], emphasis)
        for k, v in row.items():
            print(f"  {k:24s} {v}")

    if a.skip_text:
        return
    step("5. 사업보고서 원문 + 문구 변화")
    try:
        cur_no = collect.find_annual_report_rcept_no(dart, a.corp, a.year)
        prev_no = collect.find_annual_report_rcept_no(dart, a.corp, a.year - 1)
        print(f"접수번호: {a.year - 1}={prev_no}, {a.year}={cur_no}")
        if cur_no and prev_no:
            cur = collect.fetch_document_sections(dart, cur_no)
            prev = collect.fetch_document_sections(dart, prev_no)
            print(f"섹션 수: 전년 {len(prev)}, 올해 {len(cur)}")
            print("  올해 섹션 제목 앞 40개:", [t for t, _ in cur[:40]])
            for sec in text_diff.SECTION_TITLES:
                titles = text_diff.section_titles(cur, sec)
                s = text_diff.select_section(cur, sec)
                print(f"  [{'OK' if s else '확인 필요'}] {sec}: 제목 {titles} → {len(s):,}자")
                print(f"       시작: {text_diff.normalize(s)[:100]!r}")
            tc = text_diff.text_change_features(prev, cur)
            print("문구 변화:", tc)
    except Exception:
        print("[실패]"); traceback.print_exc()


if __name__ == "__main__":
    main()
