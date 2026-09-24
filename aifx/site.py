"""Write the web client (a single HTML page) next to the API documents."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

HEAD_MARKER = "<!-- /head -->"


def _template() -> str:
    return resources.files("aifx").joinpath("templates/index.html").read_text(encoding="utf-8")


def render_fragment() -> str:
    """Page content with <title>/<style> first and no <html>/<head>/<body> wrapper."""
    return _template()


def render_document() -> str:
    head, _, body = _template().partition(HEAD_MARKER)
    return (
        "<!doctype html>\n<html lang=\"ja\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n"
    )


def write_site(site_dir: Path | str) -> Path:
    out = Path(site_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render_document(), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out / "index.html"
