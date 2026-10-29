"""Prompt templates, one Markdown file per prompt. Placeholders use {name}."""

from pathlib import Path

_DIR = Path(__file__).parent


def load(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


def fill(template: str, **values: str) -> str:
    """Replace {name} placeholders without interpreting braces inside SAS/Python code."""
    filled = template
    for name, value in values.items():
        filled = filled.replace("{" + name + "}", value)
    return filled


SYSTEM = load("system")
DOCUMENT = load("document")
