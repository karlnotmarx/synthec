"""
Evaluate Human vs FinBert agreement
"""

import os
import sys
import csv
import json
from pathlib import Path
from tkinter import Y
from typing import Dict, List, Tuple

from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path.cwd()   
sys.path.append(str(PROJECT_ROOT / "src"))
from synthec.utils.io import load_csv, load_jsonl

DATA_PATH = Path("data/synthec_v0.jsonl")
ANALYST_A_PATH = Path("evaluation/annotations/financial_analyst_a.csv")
ANALYST_B_PATH = Path("evaluation/annotations/financial_analyst_b.csv")
FINBERT_PATH = Path("evaluation/reports/finbert_eval.jsonl")
OUT_PATH = Path("evaluation/reports/agreement_eval.jsonl")

LABELS = {"positive": 0, "negative": 1, "neutral": 2}

TARGET = os.getenv("CONFIDENCE_TRESHOLD", 0.60)


def label_to_id(labels: list):
    bad_labels = [l for l in labels if l not in LABELS.keys()]
    if bad_labels:
        raise ValueError(f"Unexpected labels: {set(bad_labels)}. Expected one of {LABELS.keys()}")
    
    return [LABELS[l] for l in labels]


def consensus(a , b):
    """
    Consensus between the two annotators.
    If they disagree return neutral
    """

    if a != b:
        return "neutral", False
    return a, True
    


def classification_metrics(y_true: list, y_pred: list):

    yt = label_to_id(y_true)
    yp = label_to_id(y_pred)

    return{
        "sample_size": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, target_names=LABELS.keys(), digits=4)
    }

def kappa_score(y_true: list, y_pred: list):
    try :
        return cohen_kappa_score(label_to_id(y_true), label_to_id(y_pred))
    except Exception:
        return None


def main():
    for p in [DATA_PATH, ANALYST_A_PATH, ANALYST_B_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    
    # collect all the results synthetic, finbert and humans
    data = load_jsonl(DATA_PATH)

    finbert_eval = load_jsonl(FINBERT_PATH)
    f = [r["Finbert_label"] for r in finbert_eval]
    f_conf = [r["confidence"] for r in finbert_eval]
    text = [r["paragraph"] for r in finbert_eval]


    analyst_a = load_csv(ANALYST_A_PATH)
    a = [v for k,v in analyst_a.items()]

    analyst_b = load_csv(ANALYST_B_PATH)
    b = [v for k,v in analyst_b.items()]



    # human consensus check
    consensus_flags = []
    resp = []
    for a_i, b_i in zip(a, b):
        r, ok = consensus(a_i, b_i)
        resp.append(r)
        consensus_flags.append(ok)


    assert len(data) == len(f) == len(a) == len(b), "All dataset must have the same lenght"
    num_total = len(data)
    num_consensus = sum(consensus_flags)
    consensus_rate = num_consensus/num_total
    print(f"{consensus_rate}")

    #cohen kappa score
    K_a_b = kappa_score(a, b)
    K_a_f = kappa_score(a, f)
    K_b_f = kappa_score(b, f)

    #concensus only subset
    a_cons = [x for x, ok in zip(a, consensus_flags) if ok]
    b_cons = [x for x, ok in zip(b, consensus_flags) if ok]
    c_cons = [x for x, ok in zip(resp, consensus_flags) if ok]  
    f_cons = [x for x, ok in zip(f, consensus_flags) if ok]

    f_conf_cons = [x for x, ok in zip(f_conf, consensus_flags) if ok]
    text_cons = [x for x, ok in zip(text, consensus_flags) if ok]



    #finbert vs consensus
    K_c_cons_f = kappa_score(c_cons, f_cons)
    fin_eval_sum = classification_metrics(c_cons, f_cons)


    #Disagreements analysts
    disagreements = []
    hard_cases = []

    for a_i, b_i, f_i, conf, txt in zip(a, b, f, f_conf, text):
        if a_i != b_i:
            disagreements.append(
                {   
                    "paragraph": txt,
                    "type": "Human disagreement",
                    "analyst_a" : a_i,
                    "analyst_b": b_i,
                    "finbert": f_i,
                    "finbert confidence": conf
                }
            )


        if conf < TARGET:
            hard_cases.append(
                {
                    "paragraph": txt,
                    "type": "low finbert confidence",
                    "analyst_a" : a_i,
                    "analyst_b": b_i,
                    "finbert": f_i,
                    "finbert confidence": conf
                }
            )

    #Disagreements finbert vs analysts consensus
    for c_i, f_i, conf, txt in zip(c_cons, f_cons, f_conf_cons, text_cons):
        if c_i != f_i:
            disagreements.append(
                {   
                    "paragraph": text_cons,
                    "type": "Finbert vs Humans - disagreement",
                    "analyst_a&b" : c_i,
                    "finbert": f_i,
                    "finbert confidence": f_conf_cons
                }
             )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "inputs": {
            "synthetic_data": str(DATA_PATH),
            "analyst_a_csv": str(ANALYST_A_PATH),
            "analyst_b_csv": str(ANALYST_B_PATH),
            "finbert_predictions": str(FINBERT_PATH),
        },
        "num_items": num_total,
        "labels": LABELS,
        "analysts_consensus": {
            "num_consensus": num_consensus,
            "consensus_rate": consensus_rate,
        },
        "cohen_kappa_score": {
            "analyst_analyst": K_a_b,
            "analyst_a_finbert": K_a_f,
            "analyst_b_finbert": K_b_f,
            "analysts_a&b_finbert": K_c_cons_f,
        },
        "slices": {
            "disagreements": disagreements,
            "hard_cases_low_confidence": hard_cases,
        },
    }

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")


    # Result summary

    print("Evaluation summary\n")
    print(f"Earning call excerpts: {num_total}")
    print(f"Analysts consensus: {consensus_rate:.2%}")

    print(f"Cohen’s κ (analyst↔analyst): {K_a_b:.3f}")
    print(f"Cohen’s κ (analysts consensus↔FinBERT): {K_c_cons_f:.3f}")

    print(f"Accuray (FinBERT vs consensus): {fin_eval_sum['accuracy']:.3f}")


    print(f"Saved report: {OUT_PATH}")

if __name__ == "__main__":
    main()