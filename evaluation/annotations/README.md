# Human Annotation Protocol

Two independent human annotators working in the financial industry
labeled 50 synthetic earnings call excerpts.

## CSV format

Each annotation file has three columns:

| Column | Description |
| --- | --- |
| `id` | Integer item id, matching the FinBERT prediction ids |
| `paragraph` | The earnings call excerpt text |
| `label` | Sentiment label: `positive`, `neutral`, or `negative` |

## Each annotator assigned one sentiment label per paragraph:
- positive
- neutral
- negative

## Purpose
1. Measure inter-annotator agreement (Cohen's κ)
2. Compare human consensus vs FinBERT predictions
3. Evaluate the evaluator (model-as-judge reliability)

Annotations were performed independently without discussion.
