# SYNTHEC

[![Tests](https://github.com/karlnotmarx/synthec/actions/workflows/tests.yml/badge.svg)](https://github.com/karlnotmarx/synthec/actions/workflows/tests.yml)

How far can you trust a financial sentiment model's labels and confidence? SYNTHEC is a small research harness for that question: it pairs a synthetic gold set of earnings-call excerpts with two independent human annotations, evaluates FinBERT's labels and confidence, and reports where accuracy and calibration disagree. The current focus is neutral text, where confident errors are easy to miss in aggregate accuracy.

## Layout

```
.
├── .github/workflows/tests.yml
├── evaluation
│   ├── annotations
│   │   ├── README.md
│   │   ├── financial_analyst_a.csv
│   │   └── financial_analyst_b.csv
│   └── reports
│       ├── agreement_eval.json
│       ├── finbert_eval.jsonl
│       └── finbert_eval_summary.json
├── scripts
│   ├── evaluate_finbert.py
│   ├── evaluate_human_vs_finbert.py
│   └── generate_dataset.py
├── src/synthec
│   ├── __init__.py
│   ├── agreement.py
│   ├── metrics.py
│   ├── prompts
│   │   ├── __init__.py
│   │   └── sentiment_prompt.md
│   └── utils
│       ├── __init__.py
│       ├── io.py
│       ├── json_cleaner.py
│       └── validate_format.py
├── tests
│   ├── test_agreement.py
│   ├── test_agreement_report.py
│   ├── test_finbert_summary.py
│   ├── test_io.py
│   └── test_metrics.py
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Quickstart

```bash
uv sync
uv run pytest
```

Rebuild the committed reports from saved predictions and annotations (no model download):

```bash
uv run python scripts/evaluate_finbert.py --resummarise
uv run python scripts/evaluate_human_vs_finbert.py
```

For a full FinBERT re-run, install the model extra:

```bash
uv sync --extra models
uv run python scripts/evaluate_finbert.py
```

Requires `data/synthec_v0.jsonl`, produced by `scripts/generate_dataset.py` (not committed).

Dataset generation needs the generate extra and an OpenAI API key:

```bash
uv sync --extra generate
export OPENAI_API_KEY=...
uv run python scripts/generate_dataset.py
```

## Results

| Metric | Value |
| --- | --- |
| FinBERT accuracy | 0.780 |
| FinBERT macro F1 | 0.680 |
| FinBERT ECE (10 bins) | 0.174 |
| FinBERT mean confidence | 0.935 |
| Recall: positive | 1.000 (20/20) |
| Recall: negative | 0.944 (17/18) |
| Recall: neutral | 0.167 (2/12) |
| Analyst-analyst kappa | 0.966 |
| Analyst consensus rate | 0.980 (49/50) |
| Gold vs analyst consensus agreement | 0.898 (44/49) |
| FinBERT vs consensus accuracy | 0.898 (n=49) |
| FinBERT vs consensus kappa | 0.816 |
| FinBERT vs consensus, neutral recall | 0.333 (2/6) |

**Key finding:** Scored against the synthetic gold labels, FinBERT looks confidently wrong on neutral text. It labels 10 of 12 neutral excerpts positive, with a mean confidence of 0.935 against 0.78 accuracy. But both analysts also call 5 of those excerpts positive. Every disagreement between the gold set and the analysts is of this kind. Scored against the analysts' consensus, FinBERT reaches 0.898, so about half the apparent neutral error is noise in the synthetic labels. The bias is still real: FinBERT labels 4 of the 6 neutral excerpts positive, so neutral recall is 2/6, and all 5 of its disagreements with the analysts are false positives. (The two 0.898 rows are a coincidence. Each is 44/49, but they compare different things.) The practical lesson is that synthetic gold labels need human checking before they're trusted, especially for the ambiguous class.

## Report schemas

`evaluation/reports/finbert_eval_summary.json` contains:

- `num_samples`, `accuracy`, `labels`, `avg_confidence`, `ece`
- `confusion_matrix`: rows are true labels and columns are predicted labels, both in `labels` order
- `classification_report`: sklearn `output_dict` with per-class metrics plus macro and weighted averages

`evaluation/reports/agreement_eval.json` contains:

- `num_items`
- `analyst_agreement`: `agreed`, `rate`, `cohen_kappa` (`null` when undefined)
- `analyst_disagreements`: `id`, `paragraph`, `label_a`, `label_b`
- `gold_vs_consensus`: `n`, `agreed`, `rate`, `confusion_matrix` (rows = gold, columns = consensus, both in `labels` order), and `disagreements` records with `id`, `gold_label`, `consensus_label`, `finbert_label`
- `finbert_vs_consensus`: `n`, `accuracy`, `cohen_kappa` (`null` when undefined), `confusion_matrix`, `classification_report`, and `disagreements` records with `id`, `paragraph`, `consensus_label`, `finbert_label`, `finbert_confidence`
- `low_confidence`: `id`, `finbert_label`, `confidence`, `consensus_label`

## Status

This is a research prototype on a 50-paragraph synthetic gold set. Generation is unseeded and not reproducible; the sample is small, so uncertainty is wide; there is no drift monitoring; and it is not production-ready. A production system would need a larger human-labelled set, seeded and versioned data, confidence intervals, per-class calibration, and ongoing monitoring.

## Next steps

- Compare against a second model (for example FinGPT).
- Test perturbation robustness, including flip rate under context changes.
- Add per-class calibration curves.

## Changelog

Recent commits fix issues found on review: label order in the classification report, path handling, and the agreement script.
