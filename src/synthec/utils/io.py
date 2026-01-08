"""
Data loading and saving utilities.
"""
import json
from pathlib import Path


def load_prompt(filename: str) -> str:
    """
    Load a prompt from a markdown/text file.
    """
    project_root = Path.cwd()
    prompt_path = project_root / "src/synthec/prompts" / filename
    return prompt_path.read_text(encoding="utf-8")


def load_jsonl(path:Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            rows.append(json.loads(line))
    
    return rows


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

