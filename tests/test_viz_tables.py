import pytest

from optimlab.viz import markdown_table


def _text(table):
    return getattr(table, "data", table)


def test_markdown_table_has_header_rule_and_one_line_per_row():
    text = _text(markdown_table(["name", "value"], [("a", 1), ("b", 2.5)]))
    lines = text.splitlines()
    assert lines[0] == "| name | value |"
    assert lines[1] == "|:---|---:|"
    assert lines[2:] == ["| a | 1 |", "| b | 2.5 |"]


def test_markdown_table_honors_explicit_alignment():
    text = _text(markdown_table(["a", "b", "c"], [(1, 2, 3)], align="crl"))
    assert text.splitlines()[1] == "|:---:|---:|:---|"


def test_markdown_table_rejects_ragged_rows():
    with pytest.raises(ValueError, match="one cell per header"):
        markdown_table(["a", "b"], [(1,)])
