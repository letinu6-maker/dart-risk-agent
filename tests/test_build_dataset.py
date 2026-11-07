"""scripts/build_dataset.py: 캐시 재개, 한도 초과 감지, 라벨 결합."""

import json
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from dart_agent import collect
from fakes import FakeDart

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "build_dataset.py")


@pytest.fixture
def workspace(tmp_path):
    pd.DataFrame({"corp_code": ["00000001", "00000002"]}).to_csv(tmp_path / "corps.csv", index=False)
    pd.DataFrame({"corp_code": ["00000001"], "event_date": ["2023-09-01"]}).to_csv(tmp_path / "events.csv", index=False)
    return tmp_path


def run(ws, dart, monkeypatch, *extra):
    monkeypatch.setattr(collect, "get_client", lambda *a, **k: dart)
    argv = ["build_dataset.py", "--corps", str(ws / "corps.csv"), "--events", str(ws / "events.csv"),
            "--start", "2020", "--end", "2023", "--cache-dir", str(ws / "cache"), "--out", str(ws / "panel.csv"),
            "--sleep", "0", "--stop-after-empty", "3", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(SCRIPT, run_name="__main__")
    run.last_exit = exit_info.value.code
    return pd.read_csv(ws / "panel.csv", dtype={"corp_code": str})


def test_builds_panel_with_labels_and_missing_years(workspace, monkeypatch):
    panel = run(workspace, FakeDart(listed_from={"00000002": 2022}), monkeypatch)
    assert len(panel) == 4 + 2  # 00000002는 2022년부터 상장
    assert set(panel.columns) >= {"capital_impairment", "going_concern_flag", "report_date", "label"}
    a = panel[panel.corp_code == "00000001"].set_index("year")
    assert a.loc[2022, "label"] == 1.0   # 2023-03-11 제출 → 2023-09-01 사건
    assert a.loc[2021, "label"] == 0.0
    assert json.loads((workspace / "cache" / "00000002_2020.json").read_text()) == {"missing": True}


def test_quota_stop_does_not_poison_cache_and_resumes(workspace, monkeypatch, capsys):
    # 가짜 한도는 finstate_all 호출만 센다. 3행 뒤 한도 초과 → 빈 응답 3번 연속 → 확인 호출도 빔 → 멈춤
    run(workspace, FakeDart(quota=3), monkeypatch)
    assert "한도" in capsys.readouterr().out
    assert run.last_exit == 3
    cached = {p.name: json.loads(p.read_text()) for p in (workspace / "cache").glob("*.json")}
    assert not any(v.get("missing") for v in cached.values())  # 한도 초과 빈 응답을 '없음'으로 기록하지 않음
    n_before = len(cached)

    panel = run(workspace, FakeDart(), monkeypatch)  # 다음 날 다시 실행
    assert len(panel) == 8 and n_before < 8
    assert run.last_exit == 0


def test_build_labels_script(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "get_client", lambda *a, **k: FakeDart())
    monkeypatch.setattr(sys, "argv", ["build_labels.py", "--start", "2023", "--end", "2024", "--n-normal", "2",
                                      "--out-dir", str(tmp_path), "--sleep", "0"])
    runpy.run_path(str(Path(SCRIPT).with_name("build_labels.py")), run_name="__main__")
    ev = pd.read_csv(tmp_path / "events.csv", dtype=str)
    assert list(ev.corp_code) == ["00000201", "00000201"]  # 2023·2024년 4월, 스팩·해제 제외
    corps = pd.read_csv(tmp_path / "corp_list.csv", dtype=str)
    assert "00000201" in set(corps.corp_code) and (corps.source == "normal").sum() == 2


def test_text_added_later_only_fetches_text(workspace, monkeypatch):
    run(workspace, FakeDart(), monkeypatch)
    dart = FakeDart()
    panel = run(workspace, dart, monkeypatch, "--text")
    assert panel["audit_text_change"].notna().all()
    assert dart.calls == 0  # 재무제표는 다시 부르지 않음 (finstate_all 호출 0회)


class MissingDocDart(FakeDart):
    """원문 파일이 DART에 없는 기업 (status 014)."""

    def document_all(self, rcept_no):
        raise ValueError({"status": "014", "message": "파일이 존재하지 않습니다."})


class FlakyDocDart(FakeDart):
    """원문 받는 중 네트워크가 끊기는 경우."""

    def document_all(self, rcept_no):
        raise ConnectionError("Read timed out")


def test_missing_document_keeps_financial_row(workspace, monkeypatch):
    panel = run(workspace, MissingDocDart(), monkeypatch, "--text")
    assert len(panel) == 8                                # 014여도 재무 변수 행은 남음
    assert panel["capital_impairment"].notna().all()
    assert panel["text_error"].str.contains("014").all()
    dart = MissingDocDart()
    run(workspace, dart, monkeypatch, "--text")
    assert dart.calls == 0                                # 다시 실행해도 재시도 안 함


def test_network_drop_in_text_step_retries_text_only(workspace, monkeypatch):
    run(workspace, FlakyDocDart(), monkeypatch, "--text")
    cached = [json.loads(p.read_text()) for p in (workspace / "cache").glob("*.json")]
    assert len(cached) == 8 and not any(c.get("text_done") for c in cached)  # 재무는 저장, 문구는 미완
    dart = FakeDart()
    panel = run(workspace, dart, monkeypatch, "--text")   # 연결 회복 후
    assert dart.calls == 0                                # 재무제표는 다시 안 부름
    assert panel["audit_text_change"].notna().all()


def test_default_timeout_is_installed():
    import requests

    collect.install_default_timeout(5)
    seen = {}

    class Stop(Exception):
        pass

    def fake_send(self, request, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise Stop

    orig = requests.sessions.Session.send
    requests.sessions.Session.send = fake_send
    try:
        with pytest.raises(Stop):
            requests.get("http://example.invalid")
    finally:
        requests.sessions.Session.send = orig
    assert seen["timeout"] is not None


def test_build_labels_events_only_keeps_corp_list(tmp_path, monkeypatch):
    (tmp_path / "corp_list.csv").write_text("corp_code,corp_name,source\n00000999,원래목록,normal\n")
    monkeypatch.setattr(collect, "get_client", lambda *a, **k: FakeDart())
    monkeypatch.setattr(sys, "argv", ["build_labels.py", "--start", "2023", "--end", "2024", "--events-only",
                                      "--out-dir", str(tmp_path), "--sleep", "0"])
    runpy.run_path(str(Path(SCRIPT).with_name("build_labels.py")), run_name="__main__")
    assert "원래목록" in (tmp_path / "corp_list.csv").read_text()
    assert (tmp_path / "events.csv").exists()
