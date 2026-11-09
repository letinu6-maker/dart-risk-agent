import sys
import types

from dart_agent import report

PAYLOAD = report.build_payload("테스트기업", 2024, 0.82,
                               [{"feature": "capital_impairment", "value": 0.6, "contribution": 1.2}], ["계속기업"])


def fake_genai(monkeypatch, text=None, error=None):
    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen.update(model=model, contents=contents, system=config.system_instruction)
            if error:
                raise error
            return types.SimpleNamespace(text=text)

    class Client:
        def __init__(self, api_key):
            seen["api_key"] = api_key
            self.models = Models()

    genai = types.ModuleType("google.genai")
    genai.Client = Client
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = lambda **kw: types.SimpleNamespace(**kw)
    genai.types = genai_types
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setitem(sys.modules, "google", google)
    return seen


def test_no_key_uses_template(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = report.generate_report(PAYLOAD)
    assert "투자 권유가 아닙니다" in out
    assert report.status_line() == "템플릿 작성 (LLM 키 없음)"


def test_gemini_used_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "c-key")  # 둘 다 있으면 무료인 Gemini 우선
    seen = fake_genai(monkeypatch, text="Gemini가 쓴 리포트")
    assert report.generate_report(PAYLOAD) == "Gemini가 쓴 리포트"
    assert seen["api_key"] == "g-key" and seen["model"] == report.GEMINI_DEFAULT_MODEL
    assert "투자 권유" in seen["system"] and "테스트기업" in seen["contents"]
    assert report.status_line() == "gemini 작성"


def test_gemini_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    fake_genai(monkeypatch, error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    out = report.generate_report(PAYLOAD)
    assert "투자 권유가 아닙니다" in out
    assert "429" in report.status_line()
