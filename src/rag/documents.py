"""Load policy markdown files and split them on ## section headings."""
from __future__ import annotations

from pathlib import Path

import yaml

from rag.types import PolicyDocument, Section

FRONT_MATTER_END = "---"
SECTION_PREFIX = "## "


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        raise ValueError("document is missing YAML front matter")
    end = raw.find("\n---\n", 4)
    if end < 0:
        raise ValueError("document front matter is not closed")
    payload = yaml.safe_load(raw[4:end])
    if not isinstance(payload, dict):
        raise ValueError("front matter must be a mapping")
    meta = {str(key): "" if value is None else str(value) for key, value in payload.items()}
    body = raw[end + len("\n---\n"):]
    return meta, body


def _require(meta: dict[str, str], key: str, path: Path) -> str:
    value = meta.get(key, "").strip()
    if not value:
        raise ValueError(f"{path.name} is missing required front-matter field {key!r}")
    return value


def parse_sections(raw: str) -> tuple[Section, ...]:
    """Split the full raw file on `## POL-...` headings.

    Offsets are into `raw`, so a later span check can slice the original file.
    """
    sections: list[Section] = []
    cursor = 0
    while True:
        start = raw.find(SECTION_PREFIX, cursor)
        if start < 0:
            break
        line_end = raw.find("\n", start)
        if line_end < 0:
            line_end = len(raw)
        heading = raw[start + len(SECTION_PREFIX):line_end].strip()
        if " - " not in heading:
            cursor = line_end + 1
            continue
        code, title = heading.split(" - ", 1)
        if not code.startswith("POL-"):
            cursor = line_end + 1
            continue
        next_start = raw.find("\n" + SECTION_PREFIX, line_end)
        body_end = next_start if next_start >= 0 else len(raw)
        # Body begins after the heading newline. Strip only the trailing
        # whitespace so the last word's offset still lands inside the file.
        body_start = min(line_end + 1, body_end)
        body = raw[body_start:body_end]
        stripped = body.strip()
        if stripped:
            lead = body.find(stripped[0])
            trail = body.rfind(stripped[-1]) + 1
            char_start = body_start + lead
            char_end = body_start + trail
            text = raw[char_start:char_end]
        else:
            char_start = body_start
            char_end = body_start
            text = ""
        sections.append(
            Section(
                code=code.strip(),
                title=title.strip(),
                text=text,
                char_start=char_start,
                char_end=char_end,
            )
        )
        cursor = body_end if next_start < 0 else next_start + 1
    if not sections:
        raise ValueError("document contains no POL- section headings")
    return tuple(sections)


def load_document(path: Path) -> PolicyDocument:
    raw = path.read_text(encoding="utf-8")
    meta, _body = _parse_front_matter(raw)
    return PolicyDocument(
        path=path.name,
        title=_require(meta, "title", path),
        doc_id=_require(meta, "doc_id", path),
        version=_require(meta, "version", path),
        status=_require(meta, "status", path),
        effective_date=_require(meta, "effective_date", path),
        owner=meta.get("owner", ""),
        raw=raw,
        sections=parse_sections(raw),
    )


def load_documents(raw_dir: Path) -> list[PolicyDocument]:
    paths = sorted(raw_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no markdown documents in {raw_dir}")
    return [load_document(path) for path in paths]
