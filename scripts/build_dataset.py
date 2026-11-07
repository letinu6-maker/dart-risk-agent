"""학습용 테이블 만들기: 기업 목록 × 연도 → 변수 + 라벨 → data/processed/panel.csv

    python scripts/build_labels.py --start 2016 --end 2025          # 먼저 한 번
    python scripts/build_dataset.py --start 2016 --end 2024          # 재무 변수만 (빠름)
    python scripts/build_dataset.py --start 2016 --end 2024 --text   # 문구 변화까지 (느림)

- 기업×연도마다 결과를 data/raw/cache/에 저장한다. 중간에 끊겨도(한도 초과 등) 다시 실행하면
  이어서 수집한다. --text를 나중에 붙이면 문구 변수만 추가로 수집한다.
- OpenDART는 키당 하루 호출 한도가 있다. 한도를 넘기면 빈 응답이 연속으로 와서,
  연속 빈 응답이 --stop-after-empty번 나오면 멈춘다. 다음 날 같은 명령을 다시 실행하면 된다.
- 팀원끼리 나눠 돌리려면 --shard 0/3, 1/3, 2/3 처럼 기업 목록을 나누고 cache 폴더를 합치면 된다.

호출량 (기업당 연도당): 재무 변수 3회, --text면 원문 2회 + 접수번호 조회 2회 추가.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent import collect, features, labels, text_diff  # noqa: E402


def cache_path(cache_dir: Path, corp: str, year: int) -> Path:
    return cache_dir / f"{corp}_{year}.json"


def collect_row(dart, corp: str, year: int) -> dict | None:
    fs = collect.fetch_financials(dart, corp, year)
    if fs.empty:
        return None
    audit = collect.fetch_audit_info(dart, corp, year)
    filings = collect.fetch_filings(dart, corp, f"{year}-01-01", f"{year}-12-31")
    row = features.build_feature_row(fs, audit["opinion"], filings, audit["emphasis"])
    rcept = str(fs["rcept_no"].iloc[0]) if "rcept_no" in fs.columns else ""
    # 예측 시점 = 사업보고서 제출일. 접수번호 앞 8자리가 접수일이다.
    row["report_date"] = rcept[:8] if rcept[:8].isdigit() else f"{year + 1}0331"
    row["fs_div"] = str(fs["fs_div"].iloc[0])
    return row


def add_text(dart, corp: str, year: int, row: dict) -> dict:
    p = collect.find_annual_report_rcept_no(dart, corp, year - 1)
    c = collect.find_annual_report_rcept_no(dart, corp, year)
    if p and c:
        tc = text_diff.text_change_features(collect.fetch_document_sections(dart, p),
                                            collect.fetch_document_sections(dart, c))
        row["new_warning_terms"] = "|".join(tc.pop("new_warning_terms"))
        row.update(tc)
    row["text_done"] = True
    return row


QUOTA_EXIT = 3  # 일일 한도로 멈췄을 때 종료 코드 (team_collect.py가 구분용으로 씀)


def quota_exhausted(dart) -> bool:
    """확실히 있는 데이터(삼성전자 직전 2개 연도 중 하나)를 요청해 봐서 비면 한도 초과로 본다."""
    from datetime import date

    y = date.today().year - 2
    try:
        return collect.fetch_financials(dart, "005930", y).empty
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corps", default="data/labels/corp_list.csv")
    ap.add_argument("--events", default="data/labels/events.csv")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--text", action="store_true", help="문구 변화 변수도 수집 (느림)")
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--shard", default="0/1", help="i/n: 기업 목록을 n등분한 것 중 i번째만 수집")
    ap.add_argument("--cache-dir", default="data/raw/cache")
    ap.add_argument("--out", default="data/processed/panel.csv")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--stop-after-empty", type=int, default=30)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    corps = pd.read_csv(args.corps, dtype=str)["corp_code"].str.zfill(8).tolist()
    i, n = map(int, args.shard.split("/"))
    corps = corps[i::n]

    dart = None
    pending_missing: list[Path] = []  # 빈 응답. 한도 초과가 아님이 확인된 뒤에만 '없음'으로 기록
    calls = 0

    def flush_missing():
        for p in pending_missing:
            p.write_text(json.dumps({"missing": True}))
        pending_missing.clear()

    try:
        for k, corp in enumerate(corps, 1):
            for year in range(args.start, args.end + 1):
                cp = cache_path(cache_dir, corp, year)
                row = json.loads(cp.read_text()) if cp.exists() else None
                if row is not None and row.get("missing"):
                    continue  # 그 해 사업보고서가 없는 기업 (상장 전·폐지 후)
                need_fin, need_text = row is None, args.text and not (row or {}).get("text_done")
                if not (need_fin or need_text):
                    continue
                dart = dart or collect.get_client()
                try:
                    if need_fin:
                        row = collect_row(dart, corp, year)
                        calls += 3
                        if row is None:
                            pending_missing.append(cp)
                            if len(pending_missing) >= args.stop_after_empty:
                                if quota_exhausted(dart):
                                    print(f"\n일일 호출 한도를 넘은 것으로 보여 멈춥니다 (이번 실행 약 {calls}회)."
                                          " 내일 같은 명령으로 이어서 수집하세요.")
                                    pending_missing.clear()
                                    return QUOTA_EXIT
                                flush_missing()
                            continue
                        flush_missing()  # 정상 응답이 왔으니 앞선 빈 응답은 진짜로 없는 해
                    if need_text:
                        row = add_text(dart, corp, year, row)
                        calls += 4
                    cp.write_text(json.dumps(row, ensure_ascii=False))
                except Exception as e:  # 한 기업 실패가 전체를 멈추지 않게
                    print(f"[skip] {corp} {year}: {e}", file=sys.stderr)
                time.sleep(args.sleep)
            if k % 20 == 0:
                print(f"{k}/{len(corps)} 기업 완료 (이번 실행 API 약 {calls}회)", flush=True)
        if pending_missing and not quota_exhausted(dart):
            flush_missing()  # 정상 종료: 마지막 빈 응답들도 한도 초과가 아님을 확인한 뒤 기록
    finally:
        assemble(cache_dir, args)


def assemble(cache_dir: Path, args) -> None:
    rows = []
    for f in cache_dir.glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("missing"):
            continue
        corp, year = f.stem.rsplit("_", 1)
        if args.start <= int(year) <= args.end:
            rows.append({**r, "corp_code": corp, "year": int(year)})
    if not rows:
        print("수집된 행이 없습니다.")
        return
    panel = pd.DataFrame(rows)
    events = pd.read_csv(args.events, dtype={"corp_code": str})
    events["corp_code"] = events["corp_code"].str.zfill(8)
    panel = labels.attach_labels(panel, events, args.gap_days, args.horizon_days)
    dropped = panel["label"].isna().sum()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    lab = panel.dropna(subset=["label"])
    print(f"\n저장: {len(panel)}행 → {args.out}")
    print(f"  학습 가능 {len(lab)}행, 위험=1 {int(lab.label.sum())}행 ({lab.label.mean():.2%}), "
          f"gap 안 사건으로 제외 {dropped}행")


if __name__ == "__main__":
    sys.exit(main() or 0)
