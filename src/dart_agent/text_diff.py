"""② 변화 탐지 (텍스트): 전년 대비 주석·감사보고서 문구 변화량.

한국어는 띄어쓰기·조사 때문에 단어 단위보다 문자 n-gram TF-IDF가 안정적이다.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 경고성 문구. 팀에서 사례를 보며 계속 늘린다.
WARNING_TERMS = [
    "계속기업",
    "중요한 불확실성",
    "우발부채",
    "특수관계자",
    "소송",
    "횡령",
    "배임",
    "회계정책의 변경",
    "전기오류수정",
    "강조사항",
    "지급보증",
    "담보제공",
]

# 원문에서 자를 섹션 (시작 키워드)
SECTIONS = {
    "audit": ["감사의견", "감사인의 의견"],
    "notes": ["주석", "재무제표에 대한 주석"],
}


def extract_section(text: str, section: str, max_chars: int = 30000) -> str:
    """섹션 시작 키워드부터 max_chars만큼. 못 찾으면 빈 문자열.

    대략적인 방법이다. 실제 원문 구조를 보고 정교하게 바꾸는 것이 3단계 작업.
    """
    for kw in SECTIONS[section]:
        i = text.find(kw)
        if i >= 0:
            return text[i : i + max_chars]
    return ""


def tfidf_similarity(prev_text: str, curr_text: str) -> float:
    """전년·올해 문서의 TF-IDF 코사인 유사도 (0~1). 둘 중 하나가 비면 NaN."""
    if not prev_text.strip() or not curr_text.strip():
        return float("nan")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    m = vec.fit_transform([prev_text, curr_text])
    return float(cosine_similarity(m[0], m[1])[0, 0])


def warning_term_counts(text: str, terms: list[str] = WARNING_TERMS) -> dict[str, int]:
    norm = re.sub(r"\s+", " ", text)
    return {t: norm.count(t) for t in terms}


def text_change_features(prev_doc: str, curr_doc: str) -> dict[str, float]:
    """원문 두 개 → 문구 변화 변수.

    - {section}_text_change: 1 - 유사도 (클수록 많이 바뀜)
    - warning_terms_added: 올해 새로 늘어난 경고 문구 건수 합
    - new_warning_terms: 작년엔 없고 올해 처음 나온 경고 문구 목록 (리포트 근거용, 모델 변수 아님)
    """
    out: dict[str, float] = {}
    for sec in SECTIONS:
        sim = tfidf_similarity(extract_section(prev_doc, sec), extract_section(curr_doc, sec))
        out[f"{sec}_text_change"] = 1 - sim if sim == sim else float("nan")
    prev_c, curr_c = warning_term_counts(prev_doc), warning_term_counts(curr_doc)
    out["warning_terms_added"] = float(sum(max(curr_c[t] - prev_c[t], 0) for t in WARNING_TERMS))
    out["new_warning_terms"] = [t for t in WARNING_TERMS if curr_c[t] > 0 and prev_c[t] == 0]
    return out
