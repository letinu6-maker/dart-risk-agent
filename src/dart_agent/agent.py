"""Agent 오케스트레이터: 수집 → 변화 탐지 → 위험 예측 → 리포트."""

from __future__ import annotations

from collections.abc import Callable
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
    new_warning_terms: list[str] = field(default_factory=list)


class _Log(list):
    def __init__(self, on_step):
        super().__init__()
        self.on_step = on_step

    def append(self, line):
        super().append(line)
        if self.on_step:
            self.on_step(line)


class RiskAgent:
    def __init__(self, model_path: str = "models/risk_model.joblib", dart=None, use_text: bool = True):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.dart = dart or collect.get_client()
        self.use_text = use_text

    def run(self, corp: str, year: int, corp_name: str | None = None,
            on_step: Callable[[str], None] | None = None) -> AgentResult:
        """on_step: 단계가 끝날 때마다 로그 한 줄을 받는 콜백 (데모 화면의 진행 표시용)."""
        log = _Log(on_step)

        # ① 수집
        fs = collect.fetch_financials(self.dart, corp, year)
        if fs.empty:
            raise ValueError(f"{corp}의 {year} 사업연도 사업보고서 재무제표를 찾지 못했습니다.")
        audit = collect.fetch_audit_info(self.dart, corp, year)
        filings = collect.fetch_filings(self.dart, corp, f"{year}-01-01", f"{year}-12-31")
        log.append(f"수집: 재무제표 {len(fs)}행, 감사의견 '{audit['opinion']}', 공시 {len(filings)}건")

        # ② 변화 탐지
        row = features.build_feature_row(fs, audit["opinion"], filings, audit["emphasis"])
        new_terms: list[str] = []
        if self.use_text:
            prev_no = collect.find_annual_report_rcept_no(self.dart, corp, year - 1)
            curr_no = collect.find_annual_report_rcept_no(self.dart, corp, year)
            if prev_no and curr_no:
                tc = text_diff.text_change_features(
                    collect.fetch_document_sections(self.dart, prev_no),
                    collect.fetch_document_sections(self.dart, curr_no),
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
        log.append("리포트: " + report.status_line())
        return AgentResult(corp, year, score, row, contrib, text, list(log), new_terms)

    def _explain(self, X: pd.DataFrame) -> list[dict]:
        from .model import explain_row

        return explain_row(self.model, X, self.features)
