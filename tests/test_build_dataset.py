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
    runpy.run_path(SCRIPT, run_name="__main__")
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
    cached = {p.name: json.loads(p.read_text()) for p in (workspace / "cache").glob("*.json")}
    assert not any(v.get("missing") for v in cached.values())  # 한도 초과 빈 응답을 '없음'으로 기록하지 않음
    n_before = len(cached)

    panel = run(workspace, FakeDart(), monkeypatch)  # 다음 날 다시 실행
    assert len(panel) == 8 and n_before < 8


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
