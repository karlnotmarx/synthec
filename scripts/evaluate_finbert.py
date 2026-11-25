"""
Evaluate the FinBERT LLM classification model using synthetic data.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthec.metrics import summarise_predictions
from synthec.utils.io import load_jsonl, save_jsonl

DATA_PATH = PROJECT_ROOT / "data" / "synthec_v0.jsonl"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"
SUMMARY_PATH = REPORT_DIR / "finbert_eval_summary.json"
PREDICTION_PATH = REPORT_DIR / "finbert_eval.jsonl"


class FinBert:
    def __init__(self, model_name="ProsusAI/finbert"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, use_safetensors=True
        )
        self.model.eval()
        self.id2label = self.model.config.id2label

    def prediction_one(self, text):
        torch = self._torch
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        pred_label = self.id2label[pred_id]
        confidence = float(probs[pred_id].item())
        return pred_label, confidence


def gold_label(row):
    if "label" in row:
        return row["label"]
    if "Label" in row:
        return row["Label"]
    raise KeyError("Row is missing a 'label' or 'Label' key")


def write_summary(summary):
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resummarise():
    predictions = load_jsonl(PREDICTION_PATH)
    if not predictions:
        raise FileNotFoundError(f"No predictions found in {PREDICTION_PATH}")

    summary = summarise_predictions(
        [row["true_label"] for row in predictions],
        [row["Finbert_label"] for row in predictions],
        [row["confidence"] for row in predictions],
    )
    write_summary(summary)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {SUMMARY_PATH}")


def run_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data in {DATA_PATH}")
    rows = load_jsonl(DATA_PATH)
    print(f"Loaded {len(rows)} rows")

    finbert = FinBert()

    y_true = []
    y_pred = []
    confidences = []
    fin_pred = []

    for i, row in enumerate(rows):
        text = row["paragraph"]
        true = gold_label(row)
        pred, confidence = finbert.prediction_one(text)

        y_true.append(true)
        y_pred.append(pred)
        confidences.append(confidence)

        fin_pred.append(
            {
                "id": i,
                "paragraph": text,
                "Finbert_label": pred,
                "true_label": true,
                "confidence": confidence,
            }
        )

    summary = summarise_predictions(y_true, y_pred, confidences)

    save_jsonl(fin_pred, PREDICTION_PATH)
    write_summary(summary)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {SUMMARY_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resummarise",
        action="store_true",
        help="Rebuild the summary from saved predictions without loading the model.",
    )
    args = parser.parse_args()

    if args.resummarise:
        resummarise()
    else:
        run_model()


if __name__ == "__main__":
    main()
