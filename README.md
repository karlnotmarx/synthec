# Synthec — Synthetic Earnings Call Excerpt Generator

Synthec is a lightweight tool for generating realistic earnings call excerpts with sentiment labels using large language models. It's designed to give you clean, validated synthetic financial text data for experiments, prototypes, and model evaluation.

## What it does

Synthec generates short earnings call-style paragraphs and labels them as positive, neutral, or negative. It uses templated prompts (kept separate from the code), validates the output to ensure quality, and gives you reproducible results through a simple script.

## Project structure
```text
synthec/
├── scripts/
│   └── generate_dataset.py
├── src/
│   └── synthec/
│       ├── prompts/
│       │   ├── sentiment_prompt.md
│       │   └── load_prompt.py
│       └── utils/
│           ├── json_cleaner.py
│           └── validate_format.py
```

## Running the generator

From the repository root:
```bash
python scripts/generate_dataset.py
```

This generates 50 samples in batches of 5 using `gpt-4o-mini` at temperature 0.8.

## Customizing generation

You can tweak settings using environment variables—no need to touch the code.

**Use your own prompt:**
```bash
SYNTHEC_PROMPT_FILE=path/to/your_prompt.md python scripts/generate_dataset.py
```

**Change dataset size and batch size:**
```bash
SYNTHEC_TARGET=200 SYNTHEC_BATCH_SIZE=10 python scripts/generate_dataset.py
```

**Switch models or adjust temperature:**
```bash
SYNTHEC_MODEL=gpt-4o-mini SYNTHEC_TEMPERATURE=0.5 python scripts/generate_dataset.py
```

## Configuration options

| Variable | Description | Default |
|----------|-------------|---------|
| `SYNTHEC_PROMPT_FILE` | Path to prompt file | `sentiment_prompt.md` |
| `SYNTHEC_TARGET` | Total samples to generate | 50 |
| `SYNTHEC_BATCH_SIZE` | Samples per API call | 5 |
| `SYNTHEC_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `SYNTHEC_TEMPERATURE` | Sampling temperature | 0.8 |


## Notes

All companies and financial figures are fictional. This data is meant for experimentation and evaluation, not real-world financial analysis.