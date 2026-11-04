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
