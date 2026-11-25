"""
Evaluate Human vs FinBERT agreement.
"""
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthec.agreement import build_agreement_report
from synthec.metrics import LABELS
from synthec.utils.io import load_jsonl

ANALYST_A_PATH = PROJECT_ROOT / "evaluation" / "annotations" / "financial_analyst_a.csv"
ANALYST_B_PATH = PROJECT_ROOT / "evaluation" / "annotations" / "financial_analyst_b.csv"
FINBERT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "finbert_eval.jsonl"
OUT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "agreement_eval.json"


def parse_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_annotations(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "id": parse_id(row["id"]),
                "paragraph": row["paragraph"],
                "label": row["label"],
            }
            for row in reader
        ]


def main():
    for path in (ANALYST_A_PATH, ANALYST_B_PATH, FINBERT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    low_confidence_threshold = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))

    report = build_agreement_report(
        load_annotations(ANALYST_A_PATH),
        load_annotations(ANALYST_B_PATH),
        load_jsonl(FINBERT_PATH),
        low_confidence_threshold=low_confidence_threshold,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    analyst_agreement = report["analyst_agreement"]
    gold_vs_consensus = report["gold_vs_consensus"]
    finbert_vs_consensus = report["finbert_vs_consensus"]

    print("Evaluation summary\n")
    print(f"Items: {report['num_items']}")
    print(
        f"Analysts agreed: {analyst_agreement['agreed']} "
        f"({analyst_agreement['rate']:.2%})"
    )
    print(f"Cohen's kappa (analyst-analyst): {analyst_agreement['cohen_kappa']:.3f}")
    print(
        "Gold vs consensus "
        f"(n={gold_vs_consensus['n']}): "
        f"agreed={gold_vs_consensus['agreed']}, "
        f"rate={gold_vs_consensus['rate']:.3f}"
    )
    print(
        "FinBERT vs consensus "
        f"(n={finbert_vs_consensus['n']}): "
        f"accuracy={finbert_vs_consensus['accuracy']:.3f}, "
        f"kappa={finbert_vs_consensus['cohen_kappa']:.3f}"
    )

    labels = list(LABELS)
    print("FinBERT vs consensus confusion matrix (rows=consensus, cols=FinBERT):")
    print(" " * 10 + " ".join(f"{label:>9}" for label in labels))
    for label, row in zip(labels, finbert_vs_consensus["confusion_matrix"]):
        print(f"{label:>10} " + " ".join(f"{value:>9}" for value in row))

    print(f"Analyst disagreements: {len(report['analyst_disagreements'])}")
    print(
        f"FinBERT vs consensus disagreements: "
        f"{len(finbert_vs_consensus['disagreements'])}"
    )
    print(f"Low-confidence items: {len(report['low_confidence'])}")
    print(f"\nSaved report: {OUT_PATH}")


if __name__ == "__main__":
    main()
