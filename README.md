# SYNTHEC —  Financial Sentiment Analysis

Production-grade toolkit for generating and evaluating synthetic financial text data. Designed for ML research, model benchmarking, and annotation quality assessment.

## Overview

SYNTHEC provides two core capabilities:

1. **Synthetic Data Generation**: Create realistic earnings call excerpts with ground-truth sentiment labels using LLM prompting
2. **Comprehensive Evaluation Suite**: Measure model performance and inter-annotator agreement with statistical metrics


## Architecture
```text
synthec/
├── src/synthec/              
│   ├── prompts/              
│   │   └── sentiment_prompt.md
│   └── utils/                
│       ├── io.py             
│       ├── json_cleaner.py   # JSON validation
│       └── validate_format.py # Output validation
├── scripts/                  
│   ├── generate_dataset.py   # Synthetic data generation
│   ├── evaluate_finbert.py   # Model evaluation
│   └── evaluate_human_vs_finbert.py # Inter-rater reliability
├── data/                     # Generated datasets
│   ├── synthec_v0.jsonl      # Synthetic samples
│   └── failures_v0.jsonl     # Generation failures 
│   ├── annotations/          # Human labels
│   │   ├── financial_analyst_a.csv
│   │   └── financial_analyst_b.csv
│   └── reports/              # Metrics & results
│       ├── finbert_eval_summary.json
│       ├── finbert_eval.jsonl
│       └── agreement_eval.jsonl
├── notebooks/                
│   └── experiments.ipynb
├── requirements.txt          # Dependencies
└── README.md
```

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/karlnotmarx/synthec.git
cd synthec

# Install package in editable mode
pip install -e .

# Set OpenAI API key
export OPENAI_API_KEY='your-key'
```

### Generate Synthetic Data
```bash
python scripts/generate_dataset.py
```

**Outputs**: 
- `data/synthec_v0.jsonl` (50 labeled excerpts by default)
- `data/failures_v0.jsonl` (failed generations for debugging)

### Evaluate Model Performance
```bash
python scripts/evaluate_finbert.py
```

**Outputs**: 
- `evaluation/reports/finbert_eval_summary.json` (aggregate metrics)
- `evaluation/reports/finbert_eval.jsonl` (per-item predictions)

### Measure Annotation Agreement
```bash
python scripts/evaluate_human_vs_finbert.py
```

**Requires**:
- `evaluation/annotations/financial_analyst_a.csv`
- `evaluation/annotations/financial_analyst_b.csv`

**Outputs**: `evaluation/reports/agreement_eval.jsonl`

**Metrics**:
- Cohen's Kappa (human-human, human-AI)
- Consensus performance
- Error analysis with confidence scores

## Configuration

Environment variables control generation without code changes:
```bash
# Dataset size & batching
SYNTHEC_TARGET=200 SYNTHEC_BATCH_SIZE=10 \
# Model selection
SYNTHEC_MODEL=gpt-4o \
SYNTHEC_TEMPERATURE=0.7 \
# Custom prompts (relative to project root)
SYNTHEC_PROMPT_FILE=src/synthec/prompts/custom_prompt.md \
python scripts/generate_dataset.py
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNTHEC_TARGET` | 50 | Total samples to generate |
| `SYNTHEC_BATCH_SIZE` | 5 | Samples per API call |
| `SYNTHEC_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `SYNTHEC_TEMPERATURE` | 0.8 | Sampling temperature (0.0-2.0) |
| `SYNTHEC_PROMPT_FILE` | `sentiment_prompt.md` | Prompt template path |




### Evaluation (`evaluation/`)

**Purpose**: Evaluation inputs and outputs

- **`annotations/`**: Human-labeled data
  - CSV format: `id,label`
  - Two annotators for inter-rater reliability
- **`reports/`**: Evaluation metrics and predictions
  - Summary JSON: Aggregate metrics
  - Predictions JSONL: Per-item results

## Metrics

### Model Performance
- **Accuracy**: Overall correctness
- **Per-class metrics**: Precision, recall, F1 for {negative, neutral, positive}
- **Confusion matrix**: Systematic error patterns
- **Confidence calibration**: Prediction uncertainty vs correctness

### Annotation Quality
- **Cohen's Kappa (κ)**: Agreement beyond chance
  - κ > 0.80: Almost perfect agreement
  - κ 0.60-0.80: Substantial agreement
  - κ < 0.60: Task may need refinement
- **Consensus rate**: Percentage of items where annotators agree
- **Consensus-only accuracy**: Model performance on unambiguous cases

### Error Analysis
- **Disagreement slices**: Human-human, human-AI mismatches
- **Low confidence cases**: Model uncertainty indicators (< 60% confidence)
- **Item-level diagnostics**: Text, labels, predictions, confidence scores

## Output Formats

### Generated Data (`data/synthec_v0.jsonl`)
```jsonl
{"id": 1, "paragraph": "Revenue grew 15% year-over-year...", "label": "positive"}
{"id": 2, "paragraph": "Margins compressed due to...", "label": "negative"}
{"id": 3, "paragraph": "Guidance remains unchanged...", "label": "neutral"}
```

### Evaluation Summary (`evaluation/reports/finbert_eval_summary.json`)
```json
{
  "dataset_path": "data/synthec_v0.jsonl",
  "num_samples": 50,
  "accuracy": 0.84,
  "avg_confidence": 0.91,
  "labels": ["negative", "neutral", "positive"],
  "confusion_matrix": [[45, 2, 3], [1, 38, 1], [2, 1, 47]],
  "classification_report": "..."
}
```

### Predictions (`evaluation/reports/finbert_eval.jsonl`)
```jsonl
{"id": 1, "text": "...", "true_label": "positive", "predicted_label": "positive", "confidence": 0.95, "correct": true}
{"id": 2, "text": "...", "true_label": "negative", "predicted_label": "neutral", "confidence": 0.62, "correct": false}
```

### Agreement Analysis (`evaluation/reports/agreement_eval.jsonl`)
```json
{
  "n_annotated_overlap": 50,
  "n_consensus": 42,
  "kappa": {
    "human_a_vs_human_b": 0.82,
    "human_a_vs_finbert": 0.71,
    "human_b_vs_finbert": 0.73
  },
  "consensus_only": {
    "accuracy_finbert_vs_human_consensus": 0.92,
    "classification_report": "..."
  }
}
```

### Adding Custom Prompts
1. Create new prompt file in `src/synthec/prompts/`
2. Run with: `SYNTHEC_PROMPT_FILE=src/synthec/prompts/your_prompt.md python scripts/generate_dataset.py`

## Notes

- All financial entities and figures are fictional
- Generated data is for research and prototyping only
- Not suitable for production financial analysis
- Prompt templates use GPT-4 family models (OpenAI API required)

## Citation

If you use SYNTHEC in your research, please cite:
```bibtex
@software{synthec2025,
  author = {Charles Drame},
  title = {SYNTHEC: Synthetic Financial Sentiment Analysis Suite},
  year = {2025},
  url = {https://github.com/karlnotmarx/synthec}
}
```

