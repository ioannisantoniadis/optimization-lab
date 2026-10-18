"""Markdown tables for the docs book: a code cell that returns `markdown_table(...)`
renders as a real table on the page (via IPython's `Markdown` display), so computed
numbers reach the reader as a table rather than as raw printed output.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, align: str = ""):
    """A GitHub-flavored Markdown table. `align` is one character per column (`l`, `r`,
    `c`; default left for the first column and right for the rest, which suits a label
    column followed by numbers). Returns an `IPython.display.Markdown` object when
    IPython is installed (so a notebook or Quarto cell renders it), else the plain string.
    """
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every row needs exactly one cell per header")
    align = align or "l" + "r" * (len(headers) - 1)
    if len(align) != len(headers):
        raise ValueError("align needs one character per column")
    rule = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(rule[a] for a in align) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ]
    text = "\n".join(lines)
    try:
        from IPython.display import Markdown
    except ImportError:  # pragma: no cover - IPython ships with the docs/viz tooling
        return text
    return Markdown(text)
