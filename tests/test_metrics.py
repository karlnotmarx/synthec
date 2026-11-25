import json

from synthec.metrics import calculate_ece, summarise_predictions


def test_ece_is_zero_when_all_confident_and_correct():
    confidences = [1.0, 1.0, 1.0]
    predictions = ["positive", "negative", "neutral"]
    true_labels = ["positive", "negative", "neutral"]

    assert calculate_ece(confidences, predictions, true_labels) == 0.0


def test_ece_hand_computed():
    confidences = [0.9, 0.9, 0.6, 0.6]
    predictions = ["positive", "negative", "positive", "positive"]
    true_labels = ["positive", "positive", "positive", "positive"]

    assert calculate_ece(confidences, predictions, true_labels) == 0.4


def test_ece_empty_input_is_zero():
    assert calculate_ece([], [], []) == 0.0


def test_label_order_regression():
    y_true = ["positive", "negative", "neutral", "neutral"]
    y_pred = ["positive", "negative", "positive", "neutral"]
    confidences = [0.9, 0.8, 0.7, 0.6]

    summary = summarise_predictions(y_true, y_pred, confidences)

    assert summary["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert summary["classification_report"]["neutral"]["recall"] == 0.5
    assert summary["classification_report"]["positive"]["precision"] == 0.5


def test_summary_is_json_serialisable():
    summary = summarise_predictions(
        ["positive", "negative", "neutral"],
        ["positive", "negative", "positive"],
        [0.9, 0.8, 0.7],
    )

    json.dumps(summary)
