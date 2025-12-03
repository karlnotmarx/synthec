# utils/json_cleaner.py
import re, json
from typing import Any

def extract_json_array(text: str) -> str:
    """Return the first JSON array found in text."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
 
    if not match:
        raise ValueError("No JSON array found in text. LLM ouput could not be parsed.")
    return match.group(0)




def safe_load_json(text: str) -> Any:
    """Convert output text from LLM into proper python data."""
    if not text or text.strip() == "":
        raise ValueError("Empty response")
        
    # Remove markdown code fences
    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL)
    text = text.strip().strip("`").strip()
    
    # Remove python string quotes
    if (text.startswith("'") and text.endswith("'")) or \
        (text.startswith('"') and text.endswith('"')):
        try:
            text = text[1:-1].encode('utf-8').decode('unicode_escape')
        except UnicodeDecodeError:
            pass
    
    # Extract JSON array
    if "[" in text and "]" in text:
        text = extract_json_array(text)
        
    return json.loads(text)