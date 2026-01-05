from pathlib import Path

def load_prompt(filename: str) -> str:
    """
    Load a prompt from a markdown/text file.
    """
    prompt_path = Path(__file__).parent / filename
    return prompt_path.read_text(encoding="utf-8")
