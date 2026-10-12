"""Offline, explicitly synthetic workflow check. No model or research claim."""

from pathlib import Path

from PIL import Image

from .config import LABELS, resolve, steps
from .data import audit
from .evaluate import classification_metrics
from .io import file_hash, write


def smoke(root, config_path):
    cfg = resolve(config_path, root)
    destination = Path(root) / "outputs/smoke"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (label, label_id) in enumerate(LABELS.items()):
        for j, split in enumerate(("train", "validation", "test")):
            name = f"synthetic-{label_id}-{split}.png"
            Image.new("RGB", (16, 16), (i * 70, j * 70, 42)).save(destination / name)
            rows.append({"sample_id": name, "image_uri": name, "sha256": file_hash(destination / name),
                         "canonical_label": label, "original_label": "synthetic", "source_id": "fixture",
                         "source_revision": "fixture-v1", "group_id": None, "subtype": "synthetic",
                         "review_status": "approved", "split": split,
                         "non_sexual": label == "Safe", "non_violent": label == "Safe"})
    sources = {"fixture": {"status": "pass", "revision": "fixture-v1", "license": "synthetic",
                           "license_evidence": "generated locally", "review_evidence": "fixture only",
                           "video_origin": False}}
    report = audit(rows, sources)
    metrics = classification_metrics([0, 1, 2], [0, 0, 2])
    result = {"mock": True, "research_evidence": False, "audit": report,
              "metric_fixture": metrics, "budget_example": steps(cfg, 1000),
              "note": "Schema/metric smoke only; no pretrained model or actual training"}
    write(destination / "manifest.json", {"mock": True, "records": rows})
    write(destination / "result.json", result)
    return {"ok": report["ok"], "mock": True, "result": str(destination / "result.json")}
