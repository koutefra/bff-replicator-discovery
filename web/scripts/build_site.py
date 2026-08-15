#!/usr/bin/env python3
"""Build the dependency-free static demo into web-demo/dist/."""

from __future__ import annotations

import shutil
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
DIST = DEMO_ROOT / "dist"
FILES = ("index.html", "styles.css", "app.js", "favicon.svg")
DIRECTORIES = ("data", "assets")


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for filename in FILES:
        shutil.copy2(DEMO_ROOT / filename, DIST / filename)
    for directory in DIRECTORIES:
        shutil.copytree(DEMO_ROOT / directory, DIST / directory)
    (DIST / ".nojekyll").touch()
    print(f"Built static site: {DIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
