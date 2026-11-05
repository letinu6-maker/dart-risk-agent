"""② 변화 탐지 (텍스트): 전년 대비 감사보고서·주석 문구 변화량.

원문은 collect.fetch_document_sections()가 <TITLE> 기준으로 나눈 (제목, 본문) 목록이다.
- 섹션은 제목 정규식으로 고른다 (단어가 처음 나오는 위치로 자르면 목차가 잡힘).
- 숫자는 매년 바뀌므로 지우고 비교한다 (안 지우면 안정적인 기업도 변화량이 크게 나옴).
- 한국어는 띄어쓰기·조사 때문에 단어보다 문자 n-gram TF-IDF가 안정적이다.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Sections = list[tuple[str, str]]

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

# 섹션 제목 패턴. 예: '독립된 감사인의 감사보고서', 'V. 회계감사인의 감사의견 등',
# '3. 연결재무제표 주석', '5. 재무제표 주석', '주석'
SECTION_TITLES = {
    "audit": re.compile(r"감사인의\s*감사보고서|감사의견"),
    "notes": re.compile(r"주석"),
}


def select_section(sections: Sections, name: str) -> str:
    """제목이 패턴에 맞는 섹션 본문을 모두 이어 붙인다. 없으면 빈 문자열."""
    pat = SECTION_TITLES[name]
    return " ".join(body for title, body in sections if pat.search(title))


def section_titles(sections: Sections, name: str) -> list[str]:
    pat = SECTION_TITLES[name]
    return [t for t, _ in sections if pat.search(t)]


def normalize(text: str) -> str:
    """숫자·단위·기호를 지운다. 문구 자체의 변화만 남기기 위함."""
    text = re.sub(r"[0-9０-９][0-9０-９,.\s]*", " ", text)
    text = re.sub(r"[()\[\]{}%△▲\-~:;·,.'\"/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tfidf_similarity(prev_text: str, curr_text: str) -> float:
    """전년·올해 문서의 TF-IDF 코사인 유사도 (0~1). 둘 중 하나가 비면 NaN."""
    a, b = normalize(prev_text), normalize(curr_text)
    if not a or not b:
        return float("nan")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0, 0])


def present_terms(text: str, terms: list[str] = WARNING_TERMS) -> set[str]:
    norm = re.sub(r"\s+", " ", text)
    return {t for t in terms if t in norm}


def text_change_features(prev: Sections, curr: Sections) -> dict:
    """전년·올해 섹션 목록 → 문구 변화 변수.

    - {audit,notes}_text_change: 1 - 유사도 (클수록 많이 바뀜). 섹션이 없으면 NaN
    - warning_terms_added: 감사보고서·주석에 올해 처음 등장한 경고 문구의 종류 수
      (건수 차이는 문서 길이에 비례해 대기업일수록 커지므로 쓰지 않는다)
    - new_warning_terms: 그 목록 (리포트 근거용, 모델 변수 아님)
    """
    out: dict = {}
    scope_prev, scope_curr = [], []
    for sec in SECTION_TITLES:
        p, c = select_section(prev, sec), select_section(curr, sec)
        sim = tfidf_similarity(p, c)
        out[f"{sec}_text_change"] = max(0.0, 1 - sim) if sim == sim else float("nan")
        scope_prev.append(p)
        scope_curr.append(c)
    new = sorted(present_terms(" ".join(scope_curr)) - present_terms(" ".join(scope_prev)),
                 key=WARNING_TERMS.index)
    out["warning_terms_added"] = float(len(new))
    out["new_warning_terms"] = new
    return out
