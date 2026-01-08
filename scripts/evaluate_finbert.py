"""
Evaluate the FinBERT LLM classification model using synthetic data.
"""
import pandas as pd
import encodings
import sys
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path.cwd()   
sys.path.append(str(PROJECT_ROOT / "src"))

from synthec.utils.io import load_jsonl

DATA_PATH = Path("data/synthec_v0.jsonl")
OUT_PATH = Path("evaluation/reports/finbert_eval.jsonl")



class FinBert:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
        self.model.eval()

        self.id2label = self.model.config.id2label


    @torch.no_grad()
    def prediction_one(self, text):
        inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
        )

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        pred_label = self.id2label[pred_id]
        confidence = float(probs[pred_id].item())
        # return pred_id, pred_label, logits, probs, confidence
        return pred_label, confidence



def main():
    if not DATA_PATH:
        raise FileNotFoundError(f"Missing data in {DATA_PATH}")
    rows  = load_jsonl(DATA_PATH)

    print(f"Loaded {len(rows)} rows")

    finbert = FinBert()


    y_true = []
    y_pred = []
    confidences = []

    for row in rows:
        text = row['paragraph']
        true = row['label']
        pred, confidence = finbert.prediction_one(text)

        y_true.append(true)
        y_pred.append(pred)
        confidences.append(confidence)

    
    sentiments = ['positive', 'negative', 'neutral']
    acc = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=sentiments, zero_division=0)
    print(y_true)
    print(y_pred)

    # cm = clean_confusion_matrix(y_true, y_pred, sentiments).to_dict()
    cm = confusion_matrix(y_true, y_pred).tolist()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out ={
        "dataset_path": str(DATA_PATH),
        "num_samples": len(rows),
        "accuracy": acc,
        "labels": sentiments,
        "avg_confidence": sum(confidences)/len(confidences) if confidences else None,
        "confusion_matrix": cm,
        "classification_report": class_report
    }

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n")
    print(f"Accuracy: {acc: .2%}")
    print(f"Confusion matrix: {cm}")
    print(f"Classification report: {class_report}")
    print(f"\nSaved: {OUT_PATH}")



if __name__ == "__main__":
    main()