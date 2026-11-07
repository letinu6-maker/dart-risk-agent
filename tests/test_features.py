import math

import pandas as pd

from dart_agent import features
from fakes import fake_finstate


def test_ratios_from_finstate():
    row = features.build_feature_row(fake_finstate(), "한정의견", pd.DataFrame({"report_nm": ["최대주주변경"]}))
    assert row["capital_impairment"] == 0.5          # (100 - 50) / 100
    assert row["debt_to_assets"] == 0.95
    assert row["net_loss"] == 1.0 and row["ocf_negative"] == 1.0
    assert math.isclose(row["revenue_growth"], (500 - 800) / 800)
    assert row["audit_non_clean"] == 1.0
    assert row["n_major_holder_change"] == 1.0


def test_no_impairment_is_zero():
    row = features.build_feature_row(fake_finstate(equity=300), "적정", pd.DataFrame())
    assert row["capital_impairment"] == 0.0
    assert row["audit_non_clean"] == 0.0
    assert row["n_convertible_bond"] == 0.0


def test_fallback_to_account_name():
    fs = fake_finstate().assign(account_id="-")
    assert features.extract_accounts(fs)["equity"] == 50.0


def test_missing_opinion_is_nan():
    assert math.isnan(features.audit_non_clean(None))


def test_correction_counts_only_periodic_or_audit_reports():
    titles = ["[기재정정]사업보고서 (2024.12)", "[첨부정정]감사보고서제출", "[기재정정]주주총회소집공고",
              "[기재정정]임원ㆍ주요주주특정증권등소유상황보고서", "분기보고서 (2024.09)", "[기재정정]최대주주변경"]
    ev = features.count_events(pd.DataFrame({"report_nm": titles}))
    assert ev["n_correction"] == 2.0
    assert ev["n_major_holder_change"] == 1.0
