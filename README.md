# Synthec — Synthetic Earnings Call Excerpt Generator

Synthec is a small tool for generating **high-quality synthetic earnings call excerpts with sentiment labels** using large language models.

The goal of the project is to create **clean, validated synthetic financial text data** for experimentation, prototyping, and evaluation.

## What it does

- Generates short earnings call–style paragraphs
- Assigns sentiment labels: `Positive`, `Neutral`, `Negative`
- Uses prompt engineering techniques with versioned templates
- Cleans and validates model output 

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
