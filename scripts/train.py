"""모델 3종 학습·비교 후 Agent가 쓸 모델을 저장.

    python scripts/train.py --data data/processed/panel.csv --test-start 2022 --use Logistic
    python scripts/train.py --synthetic   # 실데이터 없이 동작 확인
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dart_agent import model as M  # noqa: E402
from dart_agent.synthetic import make_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--test-start", type=int, default=2022)
    ap.add_argument("--use", default="Logistic", choices=["Logistic", "DecisionTree", "RandomForest"])
    ap.add_argument("--no-text", action="store_true", help="문구 변화 변수 제외")
    ap.add_argument("--out", default="models/risk_model.joblib")
    args = ap.parse_args()

    df = make_panel() if args.synthetic else pd.read_csv(args.data, dtype={"corp_code": str})
    feats = M.FINANCIAL_FEATURES + ([] if args.no_text else M.TEXT_FEATURES)
    feats = [f for f in feats if f in df.columns]
    train, test = M.time_split(df, args.test_start)
    models = M.build_models()
    print(M.evaluate(models, train[feats], train[M.LABEL], test[feats], test[M.LABEL]).round(3))

    best = models[args.use]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "features": feats, "trained_on": "synthetic" if args.synthetic else args.data}, args.out)
    print(f"saved {args.use} → {args.out}")


if __name__ == "__main__":
    main()
