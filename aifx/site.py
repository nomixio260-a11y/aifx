"""Render the dashboard: a single self-contained HTML file with the forecast
data embedded, so it can be opened locally, served, or hosted anywhere."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

HEAD_MARKER = "<!-- /head -->"
DATA_MARKER = "__FX_DATA__"


def _template() -> str:
    return resources.files("aifx").joinpath("templates/dashboard.html").read_text(encoding="utf-8")


def _embed(bundle: dict) -> str:
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Keep the JSON from closing the <script> element it lives in.
    return payload.replace("</", "<\\/").replace("<!--", "<\\!--")


def render_fragment(bundle: dict) -> str:
    """Page body with <title>/<style> at the top (no <html>/<head>/<body> wrapper)."""
    return _template().replace(DATA_MARKER, _embed(bundle), 1)


def render_document(bundle: dict) -> str:
    fragment = render_fragment(bundle)
    head, _, body = fragment.partition(HEAD_MARKER)
    return (
        "<!doctype html>\n<html lang=\"ja\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n"
    )


def write_site(bundle: dict, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_document(bundle), encoding="utf-8")
    (out / "forecast.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"
