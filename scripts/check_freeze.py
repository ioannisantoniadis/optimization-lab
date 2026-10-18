"""Compare the computed text of a fresh, uncached render against the committed
docs/_freeze/ results.

`freeze: auto` re-executes a page only when its own .qmd changes, so a change to
src/optimlab/ can leave the committed outputs stale without any render failing. CI runs
this after deleting docs/_freeze/ and rendering from scratch.

Usage: python scripts/check_freeze.py <committed_freeze_dir> <fresh_freeze_dir>

What is compared is every line of the executed page's Markdown that a computation can
change: printed output, tables emitted by code cells, and numbers produced by inline
`{python}` expressions, which the freeze stores already substituted into the prose. Two
things are removed first because they differ on every render even when nothing changed:
Plotly figure payloads (random element ids) and cell ids.

Numbers are compared with a tolerance, because the committed freeze is rendered on a
developer machine and CI renders on Linux, with different floating-point libraries.
Two numbers match if they agree to within `REL_TOL`, or if both are below `TINY` in
magnitude: in this book such values are round-off-scale errors and residuals (for
example 1e-17 vs. 6e-17), whose exact digits depend on the platform, not on the code.
All non-numeric text must match exactly.
"""

import difflib
import json
import re
import sys
from pathlib import Path

_RAW_HTML = re.compile(r"```\{=html\}.*?```", re.DOTALL)
_CELL_ID = re.compile(r"\{#[0-9a-f]{8} ")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
REL_TOL = 0.05
TINY = 1e-2


def _numbers_match(a: str, b: str) -> bool:
    x, y = float(a), float(b)
    if abs(x) < TINY and abs(y) < TINY:
        return True
    return abs(x - y) <= REL_TOL * max(abs(x), abs(y))


def lines_match(old: str, new: str) -> bool:
    """Equal text outside numbers; numbers equal within tolerance."""
    old, new = old.replace("\\", ""), new.replace("\\", "")  # Markdown escapes: 3\.14
    if _NUMBER.sub("#", old) != _NUMBER.sub("#", new):
        return False
    return all(_numbers_match(a, b) for a, b in zip(_NUMBER.findall(old), _NUMBER.findall(new)))


def computed_text(html_json: Path) -> list[str]:
    markdown = json.loads(html_json.read_text())["result"]["markdown"]
    markdown = _CELL_ID.sub("{", _RAW_HTML.sub("[figure]", markdown))
    return [line for line in markdown.splitlines() if "plotly" not in line]


def main(committed: Path, fresh: Path) -> int:
    failures = 0
    for fresh_file in sorted(fresh.glob("**/execute-results/html.json")):
        rel = fresh_file.relative_to(fresh)
        old_file = committed / rel
        if not old_file.exists():
            print(f"NEW   {rel} (no committed freeze to compare)")
            continue
        new, old = computed_text(fresh_file), computed_text(old_file)
        if len(new) != len(old) or not all(lines_match(o, n) for o, n in zip(old, new)):
            failures += 1
            diff = list(difflib.unified_diff(old, new, "committed", "fresh", n=0, lineterm=""))
            print(f"STALE {rel}: {sum(1 for d in diff if d.startswith(('+', '-')))} changed lines")
            for line in diff[2:14]:
                print(f"  {line[:200]}")
        else:
            print(f"OK    {rel} ({len(new)} lines)")
    if failures:
        print(f"\n{failures} page(s) have stale committed outputs: re-render them and commit docs/_freeze/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
