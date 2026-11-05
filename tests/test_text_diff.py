import math

from dart_agent import collect, text_diff
from fakes import fake_doc


def sections(audit, notes, year):
    out = []
    for raw in fake_doc(audit, notes, year):
        out += collect.split_sections(raw)
    return out


def test_split_by_title_and_pick_right_section():
    secs = sections("감사의견 적정", "일반사항 정상", 2024)
    assert text_diff.section_titles(secs, "audit") == ["독립된 감사인의 감사보고서"]
    assert text_diff.section_titles(secs, "notes") == ["3. 연결재무제표 주석"]
    # 목차 문장('감사의견 및 주석사항을 참고')은 잡히지 않아야 함
    assert "참고" not in text_diff.select_section(secs, "audit")


def test_numbers_only_change_is_near_zero():
    """같은 문구에 숫자만 바뀐 해(안정적인 기업)는 변화량이 거의 0."""
    out = text_diff.text_change_features(sections("감사의견 적정의견입니다.", "일반사항 정상 영업", 2023),
                                         sections("감사의견 적정의견입니다.", "일반사항 정상 영업", 2024))
    assert out["audit_text_change"] < 1e-6
    assert out["notes_text_change"] < 1e-6
    assert out["warning_terms_added"] == 0


def test_new_warning_terms_detected():
    out = text_diff.text_change_features(
        sections("감사의견 적정의견입니다.", "일반사항 정상 영업", 2023),
        sections("감사의견 한정. 계속기업 가정에 중요한 불확실성.", "일반사항 우발부채 소송", 2024))
    assert out["audit_text_change"] > 0.2
    assert out["new_warning_terms"] == ["계속기업", "중요한 불확실성", "우발부채", "소송"]
    assert out["warning_terms_added"] == 4


def test_term_present_both_years_is_not_new():
    out = text_diff.text_change_features(sections("감사의견", "특수관계자 거래", 2023),
                                         sections("감사의견", "특수관계자 거래 증가", 2024))
    assert "특수관계자" not in out["new_warning_terms"]


def test_missing_section_gives_nan():
    out = text_diff.text_change_features([("I. 개요", "내용")], [("I. 개요", "내용")])
    assert math.isnan(out["notes_text_change"])
