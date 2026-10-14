from .config import LABELS
from .io import digest


def classification_metrics(targets, predictions, labels=LABELS):
    if not targets or len(targets) != len(predictions):
        raise ValueError("Nonempty, aligned targets/predictions required")
    label_count = len(labels)
    cm = [[0] * label_count for _ in range(label_count)]
    for truth, pred in zip(targets, predictions):
        if type(truth) is not int or type(pred) is not int or truth not in range(label_count) or pred not in range(label_count):
            raise ValueError("Labels must be valid integer IDs for this protocol")
        cm[truth][pred] += 1
    per_class = {}
    for name, i in labels.items():
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"metric_version": 1, "labels": labels, "sample_count": len(targets),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / label_count,
            "accuracy": sum(cm[i][i] for i in range(label_count)) / len(targets),
            "per_class": per_class, "confusion_matrix": cm,
            "coverage_complete": all(v["support"] > 0 for v in per_class.values()),
            "zero_division": 0}


def split_fingerprint(records, split):
    return digest(sorted((r["sample_id"], r["sha256"], r["canonical_label"])
                         for r in records if r["split"] == split))
