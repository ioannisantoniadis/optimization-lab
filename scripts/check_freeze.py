"""Compare the printed outputs of a fresh, uncached render against the committed
docs/_freeze/ results.

`freeze: auto` re-executes a chapter only when its own .qmd changes, so a change to
src/optimlab/ can leave the committed outputs stale without any render failing. CI runs
this after deleting docs/_freeze/ and rendering from scratch.

Usage: python scripts/check_freeze.py <committed_freeze_dir> <fresh_freeze_dir>

Only printed (stdout) blocks are compared: Plotly figure payloads embed random element ids
that differ on every render even when the data is identical.
"""

import json
import re
import sys
from pathlib import Path


def printed_blocks(html_json: Path) -> list[str]:
    markdown = json.loads(html_json.read_text())["result"]["markdown"]
    blocks = re.findall(r"```\n(.*?)\n```", markdown, re.DOTALL)
    return [b for b in blocks if "plotly" not in b and len(b) < 4000]


def main(committed: Path, fresh: Path) -> int:
    failures = 0
    for fresh_file in sorted(fresh.glob("chapters/*/execute-results/html.json")):
        rel = fresh_file.relative_to(fresh)
        old_file = committed / rel
        if not old_file.exists():
            print(f"NEW   {rel} (no committed freeze to compare)")
            continue
        new, old = printed_blocks(fresh_file), printed_blocks(old_file)
        diffs = [(i, a, b) for i, (a, b) in enumerate(zip(new, old)) if a != b]
        if len(new) != len(old) or diffs:
            failures += 1
            print(f"STALE {rel}: {len(old)} committed vs {len(new)} fresh printed blocks, {len(diffs)} differ")
            for i, a, b in diffs[:3]:
                print(f"  block {i}:\n    committed: {b[:200]!r}\n    fresh:     {a[:200]!r}")
        else:
            print(f"OK    {rel} ({len(new)} printed blocks)")
    if failures:
        print(f"\n{failures} chapter(s) have stale committed outputs: re-render them and commit docs/_freeze/.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
