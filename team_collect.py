"""팀원용 학습 데이터 수집기. 이 파일을 더블클릭하면 된다 (Windows, macOS 공통).

하는 일
1. 필요한 패키지 설치
2. 처음 한 번: OpenDART 인증키와 담당 번호(예: 1/4)를 물어보고 저장
3. 담당 기업의 재무·문구 변수를 수집 (중간에 끊겨도 다시 실행하면 이어서)
4. 끝나면 '보낼파일' 폴더에 압축 파일을 만들고 그 폴더를 열어준다

하루 호출 한도에 걸리면 멈춘다. 다음 날 다시 더블클릭하면 이어서 수집한다.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START, END = 2018, 2024
QUOTA_EXIT = 3
SEND_DIR = ROOT / "보낼파일"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def finish(code: int = 0) -> None:
    input("\n창을 닫으려면 Enter를 누르세요...")
    sys.exit(code)


def check_python() -> None:
    if sys.version_info < (3, 10):
        say(f"파이썬 버전이 낮습니다 ({sys.version.split()[0]}). 3.10 이상을 설치해 주세요.")
        finish(1)


def install_packages() -> None:
    say("[1/4] 필요한 패키지를 확인하는 중... (처음엔 2~5분)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "-r", str(ROOT / "requirements-collect.txt")])
    if r.returncode != 0:
        say("패키지 설치에 실패했습니다. 위 오류 메시지를 캡처해서 Claude나 박준영에게 보내주세요.")
        finish(1)


def read_env_key(env_path: Path) -> str:
    if not env_path.exists():
        return ""
    m = re.search(r"^DART_API_KEY=(\S*)", env_path.read_text(encoding="utf-8", errors="ignore"), re.M)
    return m.group(1) if m else ""


def valid_key(key: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}", key))


def ensure_key(env_path: Path = ROOT / ".env") -> None:
    if valid_key(read_env_key(env_path)):
        say("[2/4] OpenDART 인증키: 저장된 키 사용")
        return
    say("[2/4] OpenDART 인증키를 붙여넣고 Enter를 누르세요 (40자리, 화면에 그대로 보여도 괜찮아요)")
    while True:
        key = input("인증키: ").strip().strip('"').strip("'")
        if valid_key(key):
            break
        say(f"  40자리 영문·숫자가 아닙니다 (지금 {len(key)}자). 다시 붙여넣어 주세요.")
    env_path.write_text(f"DART_API_KEY={key}\n", encoding="ascii")
    say("  저장했습니다.")


def parse_shard(text: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not m:
        return None
    i, n = int(m.group(1)), int(m.group(2))
    return (i, n) if 0 <= i < n else None


def ensure_shard(path: Path = ROOT / "shard.txt") -> tuple[int, int]:
    saved = parse_shard(path.read_text()) if path.exists() else None
    if saved:
        say(f"[3/4] 담당 번호: {saved[0]}/{saved[1]}")
        return saved
    say("[3/4] 박준영님이 알려준 담당 번호를 입력하세요 (예: 1/4)")
    while True:
        shard = parse_shard(input("담당 번호: "))
        if shard:
            break
        say("  '1/4'처럼 '내 번호/전체 인원' 형식으로 입력해 주세요. 내 번호는 전체 인원보다 작아야 해요.")
    path.write_text(f"{shard[0]}/{shard[1]}")
    return shard


def zip_cache(shard: tuple[int, int]) -> Path:
    SEND_DIR.mkdir(exist_ok=True)
    name = f"cache_{shard[0]}of{shard[1]}_{datetime.now():%m%d_%H%M}"
    out = shutil.make_archive(str(SEND_DIR / name), "zip", root_dir=ROOT / "data" / "raw", base_dir="cache")
    return Path(out)


def open_folder(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
    except Exception:
        pass


def main() -> None:
    os.chdir(ROOT)
    say("=" * 60)
    say(" 공시 위험 Agent - 팀원 데이터 수집")
    say("=" * 60)
    check_python()
    if not (ROOT / "data" / "labels" / "corp_list.csv").exists():
        say("data/labels/corp_list.csv가 없습니다. 박준영님에게 받은 압축 파일을 다시 풀어주세요.")
        finish(1)
    install_packages()
    ensure_key()
    shard = ensure_shard()

    say(f"\n[4/4] 수집 시작 ({START}~{END}년, 문구 변화 포함). 몇 시간 걸릴 수 있어요.")
    say("      컴퓨터가 잠자기 모드로 들어가지 않게 전원을 연결해 두세요. 창을 닫으면 멈춥니다.\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py"),
                        "--start", str(START), "--end", str(END), "--text",
                        "--shard", f"{shard[0]}/{shard[1]}"])

    if not (ROOT / "data" / "raw" / "cache").exists():
        say("\n수집된 파일이 없습니다. 위 메시지를 캡처해서 박준영님에게 보내주세요.")
        finish(1)
    out = zip_cache(shard)
    say("\n" + "=" * 60)
    if r.returncode == QUOTA_EXIT:
        say(" 오늘 호출 한도를 다 썼습니다. 내일 이 파일을 다시 더블클릭하면 이어서 수집해요.")
        say(f" (지금까지 수집한 것도 '{out.name}'로 묶어 뒀어요. 아직 보내지 않아도 됩니다.)")
    elif r.returncode == 0:
        say(" ✅ 수집이 끝났습니다!")
        say(f" '보낼파일' 폴더의 '{out.name}' 파일을 박준영님 메일로 보내주세요.")
    else:
        say(" 수집 중 오류로 멈췄습니다. 위 메시지를 캡처해서 보내주세요.")
        say(" 다시 더블클릭하면 이어서 수집을 시도합니다.")
    say("=" * 60)
    open_folder(SEND_DIR)
    finish(0 if r.returncode in (0, QUOTA_EXIT) else 1)


if __name__ == "__main__":
    main()
