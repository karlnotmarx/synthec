import json
from pathlib import Path

from synthec.utils.io import load_jsonl, load_prompt


def test_load_jsonl_accepts_str_path(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"id": 1, "label": "positive"}, {"id": 2, "label": "negative"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    assert load_jsonl(str(path)) == rows


def test_load_jsonl_accepts_path(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"id": 1, "label": "neutral"}]
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert load_jsonl(path) == rows


def test_load_jsonl_skips_blank_lines_including_trailing(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"id": 1}, {"id": 2}]
    path.write_text(
        "\n"
        + json.dumps(rows[0])
        + "\n"
        + "   \n"
        + "\n"
        + json.dumps(rows[1])
        + "\n\n",
        encoding="utf-8",
    )

    assert load_jsonl(path) == rows


def test_load_prompt_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    text = load_prompt("sentiment_prompt.md")

    assert "financial text generator" in text
