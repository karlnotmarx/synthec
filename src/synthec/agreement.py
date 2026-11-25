"""
Agreement analysis between human annotators and FinBERT predictions.
"""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

from .metrics import LABELS, _to_python


def consensus(a, b):
    """
    Consensus rule between two annotators.

    Agreeing annotators keep their shared label. Disagreeing annotators fall
    back to "neutral", and the returned flag is False.
    """
    if a != b:
        return "neutral", False
    return a, True


def _id_key(value):
    return str(value)


def _index_by_id(items, name):
    indexed = {}
    for item in items:
        key = _id_key(item["id"])
        if key in indexed:
            raise ValueError(f"Duplicate id {item['id']!r} in {name}")
        indexed[key] = item
    return indexed


def _normalise_label(value, allowed):
    label = str(value).lower()
    if label not in allowed:
        raise ValueError(
            f"Unknown label {value!r}; expected one of {sorted(allowed)}"
        )
    return label


def _paragraph(value):
    return str(value).strip()


def build_agreement_report(
    analyst_a,
    analyst_b,
    finbert,
    labels=LABELS,
    low_confidence_threshold=0.60,
):
    """
    Join annotator and FinBERT records by id and build an agreement report.
    """
    label_order = [str(label).lower() for label in labels]
    allowed = set(label_order)

    a_by_id = _index_by_id(analyst_a, "analyst_a")
    b_by_id = _index_by_id(analyst_b, "analyst_b")
    f_by_id = _index_by_id(finbert, "finbert")

    if not (set(a_by_id) == set(b_by_id) == set(f_by_id)):
        raise ValueError("Analyst and FinBERT id sets differ")

    analyst_disagreements = []
    finbert_disagreements = []
    low_confidence = []
    consensus_items = []
    analyst_a_labels = []
    analyst_b_labels = []
    agreed = 0

    for key, a_item in a_by_id.items():
        b_item = b_by_id[key]
        f_item = f_by_id[key]

        paragraph = _paragraph(a_item["paragraph"])
        if (
            paragraph != _paragraph(b_item["paragraph"])
            or paragraph != _paragraph(f_item["paragraph"])
        ):
            raise ValueError(f"Paragraph text differs for id {a_item['id']!r}")

        a_label = _normalise_label(a_item["label"], allowed)
        b_label = _normalise_label(b_item["label"], allowed)
        f_label = _normalise_label(f_item["Finbert_label"], allowed)
        confidence = float(f_item["confidence"])

        analyst_a_labels.append(a_label)
        analyst_b_labels.append(b_label)

        if a_label == b_label:
            agreed += 1
            consensus_label = a_label
            consensus_items.append(
                (f_item["id"], f_item["paragraph"], consensus_label, f_label, confidence)
            )
        else:
            consensus_label = "neutral"
            analyst_disagreements.append(
                {
                    "id": a_item["id"],
                    "paragraph": a_item["paragraph"],
                    "label_a": a_label,
                    "label_b": b_label,
                }
            )

        if confidence < low_confidence_threshold:
            low_confidence.append(
                {
                    "id": f_item["id"],
                    "finbert_label": f_label,
                    "confidence": confidence,
                    "consensus_label": consensus_label,
                }
            )

    for item_id, paragraph, consensus_label, f_label, confidence in consensus_items:
        if consensus_label != f_label:
            finbert_disagreements.append(
                {
                    "id": item_id,
                    "paragraph": paragraph,
                    "consensus_label": consensus_label,
                    "finbert_label": f_label,
                    "finbert_confidence": confidence,
                }
            )

    consensus_labels = [item[2] for item in consensus_items]
    finbert_labels = [item[3] for item in consensus_items]
    num_consensus = len(consensus_items)

    if num_consensus:
        finbert_accuracy = accuracy_score(consensus_labels, finbert_labels)
        finbert_kappa = cohen_kappa_score(
            consensus_labels, finbert_labels, labels=label_order
        )
    else:
        finbert_accuracy = 0.0
        finbert_kappa = 0.0

    num_items = len(a_by_id)
    report = {
        "num_items": num_items,
        "analyst_agreement": {
            "agreed": agreed,
            "rate": agreed / num_items if num_items else 0.0,
            "cohen_kappa": (
                cohen_kappa_score(
                    analyst_a_labels, analyst_b_labels, labels=label_order
                )
                if num_items
                else 0.0
            ),
        },
        "analyst_disagreements": analyst_disagreements,
        "finbert_vs_consensus": {
            "n": num_consensus,
            "accuracy": finbert_accuracy,
            "cohen_kappa": finbert_kappa,
            "confusion_matrix": confusion_matrix(
                consensus_labels, finbert_labels, labels=label_order
            ).tolist(),
            "classification_report": classification_report(
                consensus_labels,
                finbert_labels,
                labels=label_order,
                target_names=label_order,
                output_dict=True,
                zero_division=0,
            ),
            "disagreements": finbert_disagreements,
        },
        "low_confidence": low_confidence,
    }

    return _to_python(report)
