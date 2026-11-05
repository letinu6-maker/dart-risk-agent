"""학습용 테이블 만들기: 기업 목록 × 연도 → 변수, 라벨 결합 → data/processed/panel.csv

사용 예:
    python scripts/build_dataset.py --corps data/labels/corp_list.csv --start 2016 --end 2024 --text

필요한 파일 (직접 준비):
    data/labels/corp_list.csv : corp_code(종목코드 6자리) 컬럼
    data/labels/events.csv    : corp_code, event_date(YYYY-MM-DD), event_type(delisting/admin_issue)
                                 KRX KIND 상장폐지·관리종목 목록을 정리한 것

라벨: t년 변수 → t+1년에 사건(상장폐지·관리종목 지정) 또는 t+1년 감사의견 비적정이면 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent import collect, features, text_diff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corps", required=True)
    ap.add_argument("--events", default="data/labels/events.csv")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--text", action="store_true", help="문구 변화 변수도 수집 (느림)")
    ap.add_argument("--out", default="data/processed/panel.csv")
    ap.add_argument("--sleep", type=float, default=0.3, help="API 호출 간격(초). 일일 호출 한도 주의")
    args = ap.parse_args()

    dart = collect.get_client()
    corps = pd.read_csv(args.corps, dtype=str)["corp_code"].str.zfill(6).tolist()
    rows = []
    for corp in corps:
        for year in range(args.start, args.end + 1):
            try:
                fs = collect.fetch_financials(dart, corp, year)
                if fs.empty:
                    continue
                audit = collect.fetch_audit_info(dart, corp, year)
                filings = collect.fetch_filings(dart, corp, f"{year}-01-01", f"{year}-12-31")
                row = features.build_feature_row(fs, audit["opinion"], filings, audit["emphasis"])
                if args.text:
                    p = collect.find_annual_report_rcept_no(dart, corp, year - 1)
                    c = collect.find_annual_report_rcept_no(dart, corp, year)
                    if p and c:
                        tc = text_diff.text_change_features(collect.fetch_document_sections(dart, p),
                                                            collect.fetch_document_sections(dart, c))
                        tc.pop("new_warning_terms")
                        row.update(tc)
                row.update(corp_code=corp, year=year)
                rows.append(row)
            except Exception as e:  # 한 기업 실패가 전체를 멈추지 않게
                print(f"[skip] {corp} {year}: {e}", file=sys.stderr)
            time.sleep(args.sleep)

    panel = pd.DataFrame(rows)
    panel = attach_labels(panel, args.events)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"saved {len(panel)} rows → {args.out} (위험=1 비율 {panel['label'].mean():.3%})")


def attach_labels(panel: pd.DataFrame, events_path: str) -> pd.DataFrame:
    """t년 행의 라벨 = t+1년 사건 발생 여부. 감사의견은 같은 기업의 t+1년 행에서 가져온다."""
    events = pd.read_csv(events_path, dtype={"corp_code": str})
    events["corp_code"] = events["corp_code"].str.zfill(6)
    events["event_year"] = pd.to_datetime(events["event_date"]).dt.year
    ev = set(zip(events["corp_code"], events["event_year"]))

    next_audit = panel[["corp_code", "year", "audit_non_clean"]].copy()
    next_audit["year"] -= 1
    panel = panel.merge(next_audit.rename(columns={"audit_non_clean": "next_audit_non_clean"}),
                        on=["corp_code", "year"], how="left")
    panel["label"] = [
        int((c, y + 1) in ev or a == 1)
        for c, y, a in zip(panel["corp_code"], panel["year"], panel["next_audit_non_clean"])
    ]
    # 다음 해 정보는 라벨에만 쓰고 변수에서는 뺀다 (누수 방지)
    return panel.drop(columns=["next_audit_non_clean"])


if __name__ == "__main__":
    main()
