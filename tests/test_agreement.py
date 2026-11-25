import json

import pytest

from synthec.agreement import build_agreement_report, consensus


def make_inputs(labels_a, labels_b, finbert_labels, confidences=None, true_labels=None):
    if confidences is None:
        confidences = [0.9] * len(labels_a)
    if true_labels is None:
        true_labels = labels_a
    analyst_a = [
        {"id": index, "paragraph": f"Item {index}", "label": label}
        for index, label in enumerate(labels_a)
    ]
    analyst_b = [
        {"id": index, "paragraph": f"Item {index}", "label": label}
        for index, label in enumerate(labels_b)
    ]
    finbert = [
        {
            "id": index,
            "paragraph": f"Item {index}",
            "Finbert_label": label,
            "confidence": confidence,
            "true_label": true_label,
        }
        for index, (label, confidence, true_label) in enumerate(
            zip(finbert_labels, confidences, true_labels)
        )
    ]
    return analyst_a, analyst_b, finbert


def test_consensus_rule():
    assert consensus("positive", "positive") == ("positive", True)
    assert consensus("neutral", "neutral") == ("neutral", True)
    assert consensus("positive", "negative") == ("neutral", False)


def test_kappa_is_one_for_identical_analysts():
    labels = ["positive", "negative", "neutral"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    assert report["analyst_agreement"]["agreed"] == 3
    assert report["analyst_agreement"]["rate"] == 1.0
    assert report["analyst_agreement"]["cohen_kappa"] == 1.0
    assert report["finbert_vs_consensus"]["cohen_kappa"] == 1.0


def test_mismatched_ids_raise_value_error():
    labels = ["positive", "negative"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)
    finbert[1]["id"] = 99

    with pytest.raises(ValueError):
        build_agreement_report(analyst_a, analyst_b, finbert)


def test_mismatched_paragraphs_raise_value_error():
    labels = ["positive", "negative"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)
    finbert[1]["paragraph"] = "A different paragraph"

    with pytest.raises(ValueError):
        build_agreement_report(analyst_a, analyst_b, finbert)


def test_paragraphs_are_compared_stripped():
    labels = ["positive", "negative"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)
    analyst_a[0]["paragraph"] = "  Item 0  "
    analyst_b[0]["paragraph"] = "Item 0\n"

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    assert report["num_items"] == 2


def test_unknown_label_raises_value_error():
    labels = ["positive", "negative"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)
    analyst_a[0]["label"] = "mixed"

    with pytest.raises(ValueError):
        build_agreement_report(analyst_a, analyst_b, finbert)


def test_labels_are_lowercased_before_validation():
    labels = ["positive", "negative"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)
    analyst_a[0]["label"] = "POSITIVE"

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    assert report["num_items"] == 2
    assert report["analyst_agreement"]["agreed"] == 2


def test_disagreement_records_hold_single_item_paragraph_and_confidence():
    analyst_a = [
        {"id": 1, "paragraph": "One", "label": "positive"},
        {"id": 2, "paragraph": "Two", "label": "positive"},
        {"id": 3, "paragraph": "Three", "label": "neutral"},
    ]
    analyst_b = [
        {"id": 1, "paragraph": "One", "label": "positive"},
        {"id": 2, "paragraph": "Two", "label": "negative"},
        {"id": 3, "paragraph": "Three", "label": "neutral"},
    ]
    finbert = [
        {
            "id": 1,
            "paragraph": "One",
            "Finbert_label": "negative",
            "confidence": 0.7,
            "true_label": "positive",
        },
        {
            "id": 2,
            "paragraph": "Two",
            "Finbert_label": "positive",
            "confidence": 0.5,
            "true_label": "neutral",
        },
        {
            "id": 3,
            "paragraph": "Three",
            "Finbert_label": "neutral",
            "confidence": 0.9,
            "true_label": "neutral",
        },
    ]

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    assert report["analyst_disagreements"] == [
        {"id": 2, "paragraph": "Two", "label_a": "positive", "label_b": "negative"}
    ]

    disagreements = report["finbert_vs_consensus"]["disagreements"]
    assert disagreements == [
        {
            "id": 1,
            "paragraph": "One",
            "consensus_label": "positive",
            "finbert_label": "negative",
            "finbert_confidence": 0.7,
        }
    ]
    assert isinstance(disagreements[0]["paragraph"], str)
    assert isinstance(disagreements[0]["finbert_confidence"], float)

    assert report["low_confidence"] == [
        {
            "id": 2,
            "finbert_label": "positive",
            "confidence": 0.5,
            "consensus_label": "neutral",
        }
    ]


def test_confusion_matrix_uses_explicit_label_order():
    y_true = ["positive", "negative", "neutral", "neutral"]
    y_pred = ["positive", "negative", "positive", "neutral"]

    analyst_a, analyst_b, finbert = make_inputs(
        y_true, y_true, y_pred, [0.9, 0.8, 0.7, 0.6]
    )

    report = build_agreement_report(analyst_a, analyst_b, finbert)
    finbert_vs_consensus = report["finbert_vs_consensus"]

    assert finbert_vs_consensus["n"] == 4
    assert finbert_vs_consensus["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]
    assert finbert_vs_consensus["classification_report"]["neutral"]["recall"] == 0.5
    assert finbert_vs_consensus["classification_report"]["positive"]["precision"] == 0.5


def test_report_is_json_serialisable():
    labels = ["positive", "negative", "neutral"]
    analyst_a, analyst_b, finbert = make_inputs(labels, labels, labels)

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    json.dumps(report)


def test_kappa_nan_on_single_class_subset_is_reported_as_null():
    analyst_a = [
        {"id": 1, "paragraph": "Item 1", "label": "positive"},
        {"id": 2, "paragraph": "Item 2", "label": "positive"},
        {"id": 3, "paragraph": "Item 3", "label": "negative"},
    ]
    analyst_b = [
        {"id": 1, "paragraph": "Item 1", "label": "positive"},
        {"id": 2, "paragraph": "Item 2", "label": "positive"},
        {"id": 3, "paragraph": "Item 3", "label": "neutral"},
    ]
    finbert = [
        {
            "id": 1,
            "paragraph": "Item 1",
            "Finbert_label": "positive",
            "confidence": 0.9,
            "true_label": "positive",
        },
        {
            "id": 2,
            "paragraph": "Item 2",
            "Finbert_label": "positive",
            "confidence": 0.9,
            "true_label": "positive",
        },
        {
            "id": 3,
            "paragraph": "Item 3",
            "Finbert_label": "neutral",
            "confidence": 0.5,
            "true_label": "neutral",
        },
    ]

    report = build_agreement_report(analyst_a, analyst_b, finbert)

    assert report["finbert_vs_consensus"]["n"] == 2
    assert report["finbert_vs_consensus"]["cohen_kappa"] is None
    assert (
        json.loads(json.dumps(report))["finbert_vs_consensus"]["cohen_kappa"] is None
    )


def test_gold_vs_consensus_section():
    analyst_a = [
        {"id": 1, "paragraph": "One", "label": "positive"},
        {"id": 2, "paragraph": "Two", "label": "negative"},
        {"id": 3, "paragraph": "Three", "label": "neutral"},
        {"id": 4, "paragraph": "Four", "label": "positive"},
    ]
    analyst_b = [
        {"id": 1, "paragraph": "One", "label": "positive"},
        {"id": 2, "paragraph": "Two", "label": "negative"},
        {"id": 3, "paragraph": "Three", "label": "neutral"},
        {"id": 4, "paragraph": "Four", "label": "neutral"},
    ]
    finbert = [
        {
            "id": 1,
            "paragraph": "One",
            "Finbert_label": "positive",
            "confidence": 0.9,
            "true_label": "neutral",
        },
        {
            "id": 2,
            "paragraph": "Two",
            "Finbert_label": "negative",
            "confidence": 0.9,
            "true_label": "negative",
        },
        {
            "id": 3,
            "paragraph": "Three",
            "Finbert_label": "neutral",
            "confidence": 0.9,
            "true_label": "neutral",
        },
        {
            "id": 4,
            "paragraph": "Four",
            "Finbert_label": "positive",
            "confidence": 0.9,
            "true_label": "positive",
        },
    ]

    report = build_agreement_report(analyst_a, analyst_b, finbert)
    gold_vs_consensus = report["gold_vs_consensus"]

    assert gold_vs_consensus["n"] == 3
    assert gold_vs_consensus["agreed"] == 2
    assert gold_vs_consensus["rate"] == pytest.approx(2 / 3)
    assert gold_vs_consensus["confusion_matrix"] == [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]
    assert gold_vs_consensus["disagreements"] == [
        {
            "id": 1,
            "gold_label": "neutral",
            "consensus_label": "positive",
            "finbert_label": "positive",
        }
    ]
