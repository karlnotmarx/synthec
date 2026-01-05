import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path.cwd()   
sys.path.append(str(PROJECT_ROOT / "src"))

# --- Load API key from .env ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env.")

from synthec.prompts.load_prompt import load_prompt
from synthec.utils.json_cleaner import safe_load_json
from synthec.utils.validate_format import validate_json

PROMPT_FILE = os.getenv("SYNTHEC_PROMPT_FILE", "sentiment_prompt.md")
TARGET = int(os.getenv("SYNTHEC_TARGET", "50"))
BATCH_SIZE = int(os.getenv("SYNTHEC_BATCH_SIZE", "5"))
MODEL_NAME = os.getenv("SYNTHEC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("SYNTHEC_TEMPERATURE", "0.8"))


client = OpenAI()

def call_model(system_prompt, user_prompt, model_name=MODEL_NAME, temperature=TEMPERATURE):

    """
    Call the LLM and return raw text output.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    return response.choices[0].message.content



def generate_ec(target=TARGET, batch_size=BATCH_SIZE, prompt_file=PROMPT_FILE):
    # Generate Dataset
  
    dataset = []
    failures = []
    i = 0

    USER_PROMPT=f"""
    Generate {batch_size} earnings call excerpt paragraphs with a mix of positive, negative and neutral labels.
    Return ONLY JSON array (no markdown).
    """

    SYSTEM_PROMPT=load_prompt(prompt_file)

    while i < TARGET:

        raw_output = call_model(SYSTEM_PROMPT, USER_PROMPT)

        try:
            data = safe_load_json(raw_output)
            ok, err = validate_json(data)

            if not ok:

                failures.append({"raw":raw_output,
                                "error":err})
                continue

            for item in data:
                if i >= TARGET:
                    break

                dataset.append({'paragraph':item['paragraph'],
                                    'label':item['label']})
                i+= 1

        except Exception as e:
            failures.append({"raw":raw_output,
                            "error": repr(e)})

    return dataset, failures

if __name__ == "__main__":
    dataset, failures = generate_ec()
    print(f"Generated {len(dataset)} samples")
    print(f"Logged {len(failures)} failures")


