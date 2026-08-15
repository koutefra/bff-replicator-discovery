#!/usr/bin/env python3
"""Validate that the static demo is complete and safe for a Pages project path."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REQUIRED_FILES = (
    "index.html",
    "styles.css",
    "app.js",
    "data/research-data.json",
    "assets/bff-soup.mp4",
    "assets/bff-soup-poster.png",
    "assets/selfrep-test.mp4",
    "assets/selfrep-test-poster.png",
    ".nojekyll",
)
EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data"}


class AssetParser(HTMLParser):
    """Collect browser-loaded HTML resources, excluding ordinary page links."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"] or "")
        if tag == "link" and values.get("href"):
            self.assets.append(values["href"] or "")
        if tag in {"img", "source", "video", "audio"} and values.get("src"):
            self.assets.append(values["src"] or "")
        if tag == "video" and values.get("poster"):
            self.assets.append(values["poster"] or "")


def local_path(url: str) -> Path | None:
    parsed = urlsplit(url)
    if parsed.scheme in EXTERNAL_SCHEMES or parsed.netloc or not parsed.path:
        return None
    if parsed.path.startswith("/"):
        raise ValueError(f"root-relative asset URL is not project-site safe: {url}")
    return Path(unquote(parsed.path))


def validate(root: Path) -> None:
    root = root.resolve()
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing required build files: {', '.join(missing)}")

    json.loads((root / "data/research-data.json").read_text(encoding="utf-8"))

    index = (root / "index.html").read_text(encoding="utf-8")
    parser = AssetParser()
    parser.feed(index)
    for url in parser.assets:
        path = local_path(url)
        if path is not None and not (root / path).is_file():
            raise ValueError(f"HTML references a missing asset: {url}")

    css = (root / "styles.css").read_text(encoding="utf-8")
    for match in re.finditer(r"url\(\s*['\"]?([^)'\"]+)", css):
        path = local_path(match.group(1).strip())
        if path is not None and not (root / path).is_file():
            raise ValueError(f"CSS references a missing asset: {match.group(1)}")

    app = (root / "app.js").read_text(encoding="utf-8")
    match = re.search(r'\bDATA_URL\s*=\s*["\']([^"\']+)', app)
    if not match:
        raise ValueError("app.js does not define the expected DATA_URL")
    data_path = local_path(match.group(1))
    if data_path is None or not (root / data_path).is_file():
        raise ValueError(f"app.js references a missing data file: {match.group(1)}")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "dist"
    validate(root)
    print(f"Validated Pages artifact: {root.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Build validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
