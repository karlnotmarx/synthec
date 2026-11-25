import csv
import json
from pathlib import Path

import pytest

from synthec.agreement import build_agreement_report
from synthec.utils.io import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = REPO_ROOT / "evaluation" / "annotations"
REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"
ANALYST_A_PATH = ANNOTATIONS_DIR / "financial_analyst_a.csv"
ANALYST_B_PATH = ANNOTATIONS_DIR / "financial_analyst_b.csv"
FINBERT_PATH = REPORTS_DIR / "finbert_eval.jsonl"
REPORT_PATH = REPORTS_DIR / "agreement_eval.json"


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


def test_rebuilt_agreement_report_matches_committed_file():
    report = build_agreement_report(
        load_annotations(ANALYST_A_PATH),
        load_annotations(ANALYST_B_PATH),
        load_jsonl(FINBERT_PATH),
    )

    assert report["num_items"] == 50
    assert report["analyst_agreement"]["agreed"] == 49

    committed = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert_close(report, committed)
