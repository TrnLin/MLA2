"""Accessible notebook figure embedding and exported-HTML checks."""

from __future__ import annotations

import base64
import html
import mimetypes
from html.parser import HTMLParser
from pathlib import Path

from IPython.display import HTML


def figure_html(path: str | Path, alt: str) -> HTML:
    """Embed a figure as HTML whose useful alternative text survives nbconvert."""
    figure_path = Path(path)
    description = alt.strip()
    if not description:
        raise ValueError("figure alternative text must not be empty")
    mime_type = mimetypes.guess_type(figure_path.name)[0] or "image/png"
    encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
    safe_alt = html.escape(description, quote=True)
    return HTML(
        f'<img src="data:{mime_type};base64,{encoded}" alt="{safe_alt}" '
        'style="max-width: 100%; height: auto;" />'
    )


class _ImageDescriptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.missing_sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        attributes = dict(attrs)
        description = (attributes.get("alt") or "").strip()
        if not description or "no description has been provided" in description.lower():
            self.missing_sources.append(attributes.get("src") or "<unknown image>")


def validate_exported_image_descriptions(document: str) -> None:
    """Raise when an exported HTML image lacks useful alternative text."""
    parser = _ImageDescriptionParser()
    parser.feed(document)
    if parser.missing_sources:
        raise ValueError(
            f"exported HTML contains {len(parser.missing_sources)} image(s) without descriptions"
        )
