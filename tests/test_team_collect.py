import zipfile

import team_collect as tc


def test_parse_shard():
    assert tc.parse_shard("1/4") == (1, 4)
    assert tc.parse_shard(" 0 / 3 ") == (0, 3)
    assert tc.parse_shard("4/4") is None
    assert tc.parse_shard("1-4") is None


def test_key_validation_and_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    key = "a" * 40
    answers = iter(["짧은키", f'"{key}"'])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    tc.ensure_key(env)
    assert tc.read_env_key(env) == key
    tc.ensure_key(env)  # 두 번째엔 묻지 않음 (input을 또 부르면 StopIteration)


def test_shard_is_remembered(tmp_path, monkeypatch):
    p = tmp_path / "shard.txt"
    answers = iter(["5/4", "2/4"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    assert tc.ensure_shard(p) == (2, 4)
    assert tc.ensure_shard(p) == (2, 4)


def test_zip_cache(tmp_path, monkeypatch):
    (tmp_path / "data" / "raw" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "cache" / "00000001_2020.json").write_text("{}")
    monkeypatch.setattr(tc, "ROOT", tmp_path)
    monkeypatch.setattr(tc, "SEND_DIR", tmp_path / "보낼파일")
    out = tc.zip_cache((1, 4))
    assert out.name.startswith("cache_1of4_")
    assert zipfile.ZipFile(out).namelist() == ["cache/", "cache/00000001_2020.json"]
