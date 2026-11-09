"""④ 리포트: 위험 점수와 근거를 사람이 읽을 문장으로.

ANTHROPIC_API_KEY가 있으면 LLM이 쓰고, 없으면 템플릿 문장으로 만든다 (데모·테스트용).
LLM에는 계산된 숫자만 넘기고, 숫자에 없는 내용을 지어내지 말라고 지시한다.
"""

from __future__ import annotations

import json
import os

FEATURE_KO = {
    "capital_impairment": "자본잠식률",
    "debt_to_assets": "부채/자산 비율",
    "roa": "총자산이익률(ROA)",
    "operating_margin": "영업이익률",
    "ocf_to_assets": "영업현금흐름/자산",
    "net_loss": "당기순손실 여부",
    "ocf_negative": "영업현금흐름 적자 여부",
    "revenue_growth": "매출 증가율",
    "roa_change": "ROA 전년 대비 변화",
    "debt_to_assets_change": "부채비율 전년 대비 변화",
    "audit_non_clean": "감사의견 비적정",
    "going_concern_flag": "감사보고서 계속기업 불확실성 언급",
    "n_major_holder_change": "최대주주 변경 공시 건수",
    "n_convertible_bond": "CB·BW·EB 발행 공시 건수",
    "n_rights_offering": "유상증자 공시 건수",
    "n_correction": "정기·감사보고서 정정 건수",
    "audit_text_change": "감사보고서 문구 변화량",
    "notes_text_change": "주석 문구 변화량",
    "warning_terms_added": "올해 처음 등장한 경고 문구 종류 수",
}

SYSTEM_PROMPT = """너는 상장기업 공시를 분석하는 리스크 애널리스트다.
주어진 JSON의 숫자와 문구만 근거로 한국어 리포트를 쓴다.
규칙:
- JSON에 없는 사실, 뉴스, 주가 전망을 지어내지 않는다.
- 매수·매도 같은 투자 권유를 하지 않는다.
- 구성: (1) 한 줄 결론 (2) 위험 점수와 등급 (3) 점수를 올린 근거 3~5개 (4) 문구 변화에서 눈여겨볼 점 (5) 한계와 확인할 공시
- 마지막에 '본 리포트는 공시 데이터 기반 참고 자료이며 투자 권유가 아닙니다.'를 붙인다."""


def risk_grade(score: float) -> str:
    if score >= 0.7:
        return "높음"
    if score >= 0.4:
        return "주의"
    return "낮음"


def build_payload(corp_name: str, year: int, score: float, contributions: list[dict],
                  new_warning_terms: list[str] | None = None) -> dict:
    return {
        "기업": corp_name,
        "기준 사업연도": year,
        "위험 점수(0~1)": round(score, 3),
        "등급": risk_grade(score),
        "기여 변수": [
            {"변수": FEATURE_KO.get(c["feature"], c["feature"]), "값": c["value"],
             "방향": "위험 증가" if c["contribution"] > 0 else "위험 감소"}
            for c in contributions
        ],
        "올해 처음 등장한 경고 문구": new_warning_terms or [],
    }


def template_report(payload: dict) -> str:
    up = [c for c in payload["기여 변수"] if c["방향"] == "위험 증가"]
    lines = [
        f"## {payload['기업']} ({payload['기준 사업연도']} 사업연도) 재무위험 리포트",
        f"**위험 점수 {payload['위험 점수(0~1)']:.2f} · 등급 {payload['등급']}**",
        "",
        "### 점수를 올린 근거",
    ]
    lines += [f"- {c['변수']}: {c['값']}" for c in up] or ["- 위험을 높인 주요 변수가 없습니다."]
    if payload["올해 처음 등장한 경고 문구"]:
        lines += ["", "### 문구 변화", "- 올해 처음 등장: " + ", ".join(payload["올해 처음 등장한 경고 문구"])]
    lines += ["", "본 리포트는 공시 데이터 기반 참고 자료이며 투자 권유가 아닙니다."]
    return "\n".join(lines)


GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"  # 무료 등급으로 쓰기 좋은 가벼운 모델. GEMINI_MODEL로 바꿀 수 있음
CLAUDE_DEFAULT_MODEL = "claude-sonnet-5-5"


def llm_provider() -> str | None:
    """키가 있는 LLM. Gemini(무료 등급)를 우선 쓰고, 없으면 Claude, 둘 다 없으면 None(템플릿)."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def llm_enabled() -> bool:
    return llm_provider() is not None


def _user_message(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _gemini(payload: dict) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL),
        contents=_user_message(payload),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=1500,
                                           temperature=0.3),
    )
    if not resp.text:
        raise RuntimeError("Gemini가 빈 응답을 돌려줬습니다.")
    return resp.text


def _claude(payload: dict) -> str:
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("LLM_MODEL", CLAUDE_DEFAULT_MODEL),
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(payload)}],
    )
    return msg.content[0].text


LAST_STATUS = {"provider": None, "error": None}  # 직전 리포트를 무엇으로 썼는지 (로그·화면 표시용)


def generate_report(payload: dict, use_llm: bool | None = None) -> str:
    """LLM으로 리포트를 쓴다. 키가 없거나 호출이 실패하면(무료 한도 초과 등) 템플릿 문장으로 대신한다."""
    provider = llm_provider() if use_llm is not False else None
    LAST_STATUS.update(provider=None, error=None)
    if provider is None:
        return template_report(payload)
    try:
        text = _gemini(payload) if provider == "gemini" else _claude(payload)
        LAST_STATUS["provider"] = provider
        return text
    except Exception as e:  # 데모 중 LLM 오류로 화면이 깨지지 않게
        LAST_STATUS["error"] = f"{provider}: {str(e)[:200]}"
        return template_report(payload)


def status_line() -> str:
    if LAST_STATUS["provider"]:
        return f"{LAST_STATUS['provider']} 작성"
    if LAST_STATUS["error"]:
        return f"템플릿 작성 (LLM 호출 실패 → {LAST_STATUS['error']})"
    return "템플릿 작성 (LLM 키 없음)"
