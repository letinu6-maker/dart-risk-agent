"""Agent 오케스트레이터: 수집 → 변화 탐지 → 위험 예측 → 리포트."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import pandas as pd

from . import collect, features, report, text_diff


@dataclass
class AgentResult:
    corp: str
    year: int
    score: float
    feature_row: dict
    contributions: list[dict]
    report: str
    log: list[str] = field(default_factory=list)


class RiskAgent:
    def __init__(self, model_path: str = "models/risk_model.joblib", dart=None, use_text: bool = True):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.dart = dart or collect.get_client()
        self.use_text = use_text

    def run(self, corp: str, year: int, corp_name: str | None = None) -> AgentResult:
        log = []

        # ① 수집
        fs = collect.fetch_financials(self.dart, corp, year)
        opinion = collect.fetch_audit_opinion(self.dart, corp, year)
        filings = collect.fetch_filings(self.dart, corp, f"{year}-01-01", f"{year}-12-31")
        log.append(f"수집: 재무제표 {len(fs)}행, 감사의견 '{opinion}', 공시 {len(filings)}건")

        # ② 변화 탐지
        row = features.build_feature_row(fs, opinion, filings)
        new_terms: list[str] = []
        if self.use_text:
            prev_no = collect.find_annual_report_rcept_no(self.dart, corp, year - 1)
            curr_no = collect.find_annual_report_rcept_no(self.dart, corp, year)
            if prev_no and curr_no:
                tc = text_diff.text_change_features(
                    collect.fetch_document_text(self.dart, prev_no),
                    collect.fetch_document_text(self.dart, curr_no),
                )
                new_terms = tc.pop("new_warning_terms")
                row.update(tc)
                log.append(f"문구 변화: {tc}")
            else:
                log.append("문구 변화: 사업보고서 원문을 찾지 못해 건너뜀")

        # ③ 위험 예측
        X = pd.DataFrame([row]).reindex(columns=self.features)
        score = float(self.model.predict_proba(X)[0, 1])
        contrib = self._explain(X)
        log.append(f"위험 점수: {score:.3f}")

        # ④ 리포트
        payload = report.build_payload(corp_name or corp, year, score, contrib, new_terms)
        text = report.generate_report(payload)
        return AgentResult(corp, year, score, row, contrib, text, log)

    def _explain(self, X: pd.DataFrame) -> list[dict]:
        from .model import explain_row

        return explain_row(self.model, X, self.features)
