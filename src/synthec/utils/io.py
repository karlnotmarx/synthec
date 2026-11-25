"""
Data loading and saving utilities.
"""
import csv
import json
from pathlib import Path


def load_prompt(filename: str) -> str:
    """
    Load a prompt from a markdown/text file.
    """
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
    return prompt_path.read_text(encoding="utf-8")


def load_jsonl(path: str | Path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    
    return rows


def load_csv(path):
    """
    Expected csv columns:
    id,
    label (positive, negative  or neutral)
    """

    if not path:
        raise FileNotFoundError(f"Missing file at {path}.")

    out = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} must contain columns: id,label. Found: {reader.fieldnames}")
        for r in reader:
            out[r['id']] = r['label'].lower()

    return out




def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

