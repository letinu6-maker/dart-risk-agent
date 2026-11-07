"""라벨과 학습 대상 기업 목록 만들기 (OpenDART만 사용).

    python scripts/build_labels.py --start 2016 --end 2025 --n-normal 600

출력
    data/labels/events.csv     위험 사건 (관리종목 지정, 상장폐지 사유 발생·결정)
    data/labels/corp_list.csv  학습 대상 기업 = 사건 기업 전부 + 정상 기업 표본

API 호출: 월 1회 × 기간 (10년이면 약 120회 + 페이지 수). 한 번 돌리면 된다.
events.csv의 report_nm을 팀에서 훑어보고, 부실과 무관한 폐지가 섞여 있으면
src/dart_agent/labels.py의 EXCLUDE_TITLE에 키워드를 추가한 뒤 다시 돌리세요.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent import collect, labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True, help="사건 수집 시작 연도")
    ap.add_argument("--end", type=int, required=True, help="사건 수집 끝 연도")
    ap.add_argument("--n-normal", type=int, default=600, help="정상 기업 표본 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/labels")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    dart = collect.get_client()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = []
    for s, e in labels.month_ranges(args.start, args.end):
        df = dart.list(start=s, end=e, kind="I", final=False)
        n = 0 if df is None else len(df)
        print(f"{s[:7]} 거래소공시 {n}건", flush=True)
        if n:
            raw.append(df)
        time.sleep(args.sleep)
    filings = pd.concat(raw, ignore_index=True) if raw else pd.DataFrame()
    events = labels.classify_events(filings)
    events.to_csv(out / "events.csv", index=False)
    print(f"\n사건 {len(events)}건, 기업 {events['corp_code'].nunique()}곳 → {out / 'events.csv'}")
    print(events["event_type"].value_counts().to_string())

    universe = labels.sample_universe(dart.corp_codes, events, args.n_normal, args.seed)
    universe.to_csv(out / "corp_list.csv", index=False)
    print(f"학습 대상 기업 {len(universe)}곳 (사건 {sum(universe.source == 'event')}, "
          f"정상 {sum(universe.source == 'normal')}) → {out / 'corp_list.csv'}")
    print("\n다음: events.csv의 report_nm을 훑어보고 이상한 사건이 없는지 확인하세요.")


if __name__ == "__main__":
    main()
