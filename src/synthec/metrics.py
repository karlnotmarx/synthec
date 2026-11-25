"""
Evaluation metrics for sentiment predictions.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ("positive", "negative", "neutral")


def _to_python(value):
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def calculate_ece(confidences, predictions, true_labels, num_bins=10):
    """
    Top-label expected calibration error with equal-width bins over [0, 1].

    A confidence of exactly 1.0 belongs to the last bin.
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_counts = [0] * num_bins
    bin_confidences = [0.0] * num_bins
    bin_correct = [0] * num_bins

    for confidence, prediction, true_label in zip(confidences, predictions, true_labels):
        index = int(confidence * num_bins)
        index = min(max(index, 0), num_bins - 1)
        bin_counts[index] += 1
        bin_confidences[index] += float(confidence)
        bin_correct[index] += int(prediction == true_label)

    ece = 0.0
    for index in range(num_bins):
        count = bin_counts[index]
        if count == 0:
            continue
        accuracy = bin_correct[index] / count
        mean_confidence = bin_confidences[index] / count
        ece += (count / n) * abs(accuracy - mean_confidence)

    return float(ece)


def summarise_predictions(y_true, y_pred, confidences, labels=LABELS):
    """
    Build a JSON-serialisable summary of predictions in the given label order.
    """
    labels = list(labels)
    num_samples = len(y_true)

    accuracy = accuracy_score(y_true, y_pred) if num_samples else 0.0
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "num_samples": num_samples,
        "accuracy": accuracy,
        "labels": labels,
        "avg_confidence": sum(confidences) / num_samples if num_samples else 0.0,
        "ece": calculate_ece(confidences, y_pred, y_true),
        "confusion_matrix": matrix,
        "classification_report": report,
    }

    return _to_python(summary)
