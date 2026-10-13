"""Download the Pikafish engine and its NNUE weights into ``engines/pikafish/``.

    python tools/fetch_pikafish.py              # latest release, this platform only
    python tools/fetch_pikafish.py --all        # binaries for Windows, macOS and Linux
    python tools/fetch_pikafish.py --tag Pikafish-2026-09-06

Releases are published as a single .7z archive, so ``py7zr`` (see
requirements.txt) or a ``7z`` command line tool is required.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_player import ENGINE_DIR, PIKAFISH_BINARIES  # noqa: E402

API = "https://api.github.com/repos/official-pikafish/Pikafish/releases"
EXTRAS = ("pikafish.nnue", "Copying.txt", "NNUE-License.md")


def release_asset(tag: str) -> tuple[str, str]:
    url = f"{API}/tags/{tag}" if tag else f"{API}/latest"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "xiangqi-rl"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    for asset in release.get("assets", []):
        if asset["name"].endswith(".7z"):
            return release["tag_name"], asset["browser_download_url"]
    raise SystemExit(f"release {release.get('tag_name')} has no .7z asset")


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "xiangqi-rl"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}  {done >> 20} / {total >> 20} MiB", end="", flush=True)
    print()


def extract(archive: Path, names: list[str], into: Path) -> None:
    try:
        import py7zr
    except ImportError:
        tool = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
        if tool is None:
            raise SystemExit("需要 py7zr（pip install py7zr）或 7z 命令行工具来解压 Pikafish 发布包")
        subprocess.run([tool, "e", "-y", f"-o{into}", str(archive), *names], check=True, stdout=subprocess.DEVNULL)
        return
    with py7zr.SevenZipFile(archive) as packed:
        available = set(packed.getnames())
        packed.extract(path=into, targets=[name for name in names if name in available])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="", help="release tag (default: latest)")
    parser.add_argument("--all", action="store_true", help="install the binaries for every desktop platform")
    parser.add_argument("--dest", default=str(ENGINE_DIR))
    args = parser.parse_args()

    if args.all:
        binaries = list(PIKAFISH_BINARIES.values())
    elif platform.system() in PIKAFISH_BINARIES:
        binaries = [PIKAFISH_BINARIES[platform.system()]]
    else:
        raise SystemExit(f"Pikafish 没有为 {platform.system()} 提供预编译版本，请自行编译并设置 PIKAFISH_PATH")

    tag, url = release_asset(args.tag)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Pikafish {tag}\n  {url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "pikafish.7z"
        download(url, archive)
        extract(archive, binaries + list(EXTRAS), Path(tmp))
        for name in binaries + list(EXTRAS):
            source = Path(tmp) / name
            if source.is_file():
                shutil.move(str(source), dest / name)
    for name in binaries:
        path = dest / name
        if not path.is_file():
            raise SystemExit(f"发布包里缺少 {name}")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  已安装 {path}")
    (dest / "VERSION").write_text(tag + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
