from dart_agent import text_diff


def test_identical_text_has_no_change():
    t = "감사의견 적정의견입니다. 주석 1. 일반사항"
    out = text_diff.text_change_features(t, t)
    assert out["audit_text_change"] < 1e-9
    assert out["warning_terms_added"] == 0


def test_new_warning_terms_detected():
    prev = "감사의견 적정의견입니다. 주석 정상 영업"
    curr = "감사의견 한정의견. 계속기업 가정에 중요한 불확실성. 주석 우발부채"
    out = text_diff.text_change_features(prev, curr)
    assert out["audit_text_change"] > 0.2
    assert {"계속기업", "중요한 불확실성", "우발부채"} <= set(out["new_warning_terms"])
    assert out["warning_terms_added"] >= 3


def test_missing_section_gives_nan():
    out = text_diff.text_change_features("내용 없음", "내용 없음")
    assert out["notes_text_change"] != out["notes_text_change"]  # NaN
