import json
from pathlib import Path

import pytest

from synthec.metrics import summarise_predictions
from synthec.utils.io import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"
PREDICTION_PATH = REPORTS_DIR / "finbert_eval.jsonl"
SUMMARY_PATH = REPORTS_DIR / "finbert_eval_summary.json"


def assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_close(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_close(actual_item, expected_item)
    elif isinstance(actual, float) or isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-9, rel=0)
    else:
        assert actual == expected


def test_finbert_saved_predictions_summary():
    predictions = load_jsonl(PREDICTION_PATH)
    summary = summarise_predictions(
        [row["true_label"] for row in predictions],
        [row["Finbert_label"] for row in predictions],
        [row["confidence"] for row in predictions],
    )

    assert summary["accuracy"] == 0.78
    assert summary["confusion_matrix"] == [
        [20, 0, 0],
        [1, 17, 0],
        [10, 0, 2],
    ]

    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert_close(summary, committed)
