"""한 기업에 대해 Agent 4단계를 실행하고 리포트를 출력.

    python scripts/run_agent.py --corp 005930 --year 2025 --name 삼성전자
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent.agent import RiskAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corp", required=True, help="종목코드 6자리")
    ap.add_argument("--year", type=int, required=True, help="사업연도")
    ap.add_argument("--name")
    ap.add_argument("--model", default="models/risk_model.joblib")
    ap.add_argument("--no-text", action="store_true")
    args = ap.parse_args()

    agent = RiskAgent(args.model, use_text=not args.no_text)
    res = agent.run(args.corp, args.year, args.name)
    for line in res.log:
        print("[agent]", line)
    print()
    print(res.report)


if __name__ == "__main__":
    main()
