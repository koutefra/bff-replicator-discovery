#!/usr/bin/env python3
"""Build the dependency-free static demo into web-demo/dist/."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
DIST = DEMO_ROOT / "dist"
FILES = ("index.html", "styles.css", "app.js", "favicon.svg")
DIRECTORIES = ("data", "assets")


def build_version() -> str:
    """A short token that changes on every deploy, used to cache-bust
    app.js/styles.css/the JSON data fetch so browsers don't keep serving
    a stale cached copy after a new commit is pushed."""
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:8]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=DEMO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return str(int(time.time()))


def stamp_version(dist: Path, version: str) -> None:
    index_path = dist / "index.html"
    html = index_path.read_text()
    html = html.replace('href="styles.css"', f'href="styles.css?v={version}"')
    html = html.replace('src="app.js" type="module"', f'src="app.js?v={version}" type="module"')
    index_path.write_text(html)

    app_js_path = dist / "app.js"
    app_js = app_js_path.read_text()
    app_js = app_js.replace(
        'const DATA_URL = "data/research-data.json";',
        f'const DATA_URL = "data/research-data.json?v={version}";',
    )
    app_js_path.write_text(app_js)


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for filename in FILES:
        shutil.copy2(DEMO_ROOT / filename, DIST / filename)
    for directory in DIRECTORIES:
        shutil.copytree(DEMO_ROOT / directory, DIST / directory)
    (DIST / ".nojekyll").touch()
    stamp_version(DIST, build_version())
    print(f"Built static site: {DIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
