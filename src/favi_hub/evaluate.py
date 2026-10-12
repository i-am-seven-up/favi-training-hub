from .config import LABELS
from .io import digest


def classification_metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError("Nonempty, aligned targets/predictions required")
    cm = [[0] * 3 for _ in range(3)]
    for truth, pred in zip(targets, predictions):
        if type(truth) is not int or type(pred) is not int or truth not in range(3) or pred not in range(3):
            raise ValueError("Labels must be canonical integer IDs 0,1,2")
        cm[truth][pred] += 1
    per_class = {}
    for name, i in LABELS.items():
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"metric_version": 1, "labels": LABELS, "sample_count": len(targets),
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 3,
            "accuracy": sum(cm[i][i] for i in range(3)) / len(targets),
            "per_class": per_class, "confusion_matrix": cm,
            "coverage_complete": all(v["support"] > 0 for v in per_class.values()),
            "zero_division": 0}


def split_fingerprint(records, split):
    return digest(sorted((r["sample_id"], r["sha256"], r["canonical_label"])
                         for r in records if r["split"] == split))
