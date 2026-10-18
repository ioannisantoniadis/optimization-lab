"""Quarto post-render step: remove the MathJax 2 <script> tags that Plotly's notebook
renderer injects next to every figure.

Quarto loads its own MathJax (v4) for the book's math. Plotly's renderer hard-codes an
extra MathJax 2.7.5 tag per figure, and the two versions race: whichever initializes
first wins, and when MathJax 2 wins, every equation on the page stays raw LaTeX. No
figure in the book uses LaTeX in its labels, so the Plotly copy is never needed.

Runs automatically from docs/_quarto.yml (`post-render`); standard library only.
"""

import os
import re
from pathlib import Path

PLOTLY_MATHJAX = re.compile(
    r'<script[^>]*src="https://cdnjs\.cloudflare\.com/ajax/libs/mathjax/2\.[^"]*"[^>]*>\s*</script>'
)


def main() -> None:
    output_dir = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
    stripped = 0
    for page in output_dir.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        cleaned, n = PLOTLY_MATHJAX.subn("", html)
        if n:
            page.write_text(cleaned, encoding="utf-8")
            stripped += n
    print(f"strip_plotly_mathjax: removed {stripped} duplicate MathJax tags")


if __name__ == "__main__":
    main()
