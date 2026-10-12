from pathlib import Path

import pytest

from favi_hub.config import LABELS, resolve
from favi_hub.data import freeze
from favi_hub.io import digest, write

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def research(tmp_path):
    """Synthetic isolated research state; it never enters the real experiment registry."""
    rows = []
    for split, count in (("train", 30), ("validation", 2), ("test", 2)):
        for label in LABELS:
            for index in range(count):
                sample_id = f"{split}-{label}-{index}"
                rows.append({
                    "sample_id": sample_id,
                    "image_uri": sample_id + ".png",
                    "sha256": digest(sample_id),
                    "canonical_label": label,
                    "original_label": label,
                    "source_id": "fixture",
                    "source_revision": "fixture-v1",
                    "group_id": None,
                    "subtype": "synthetic",
                    "review_status": "approved",
                    "split": split,
                    "non_sexual": label == "Safe",
                    "non_violent": label == "Safe",
                })
    source = {
        "status": "pass",
        "revision": "fixture-v1",
        "license": "synthetic",
        "license_evidence": "generated locally",
        "review_evidence": "review.json",
        "video_origin": False,
    }
    write(tmp_path / "review.json", {
        "reviewer": "test",
        "reviewed_at": "2026-10-11",
        "decision": "pass",
        "source_revision": "fixture-v1",
        "sample_ids": [row["sample_id"] for row in rows[:20]],
        "notes": "Synthetic test only",
    })
    write(tmp_path / "registry.yaml", {"sources": {"fixture": source}})
    write(tmp_path / "manifest.json", rows)
    frozen = freeze(
        tmp_path / "manifest.json",
        tmp_path / "registry.yaml",
        tmp_path / "data.json",
        tmp_path,
        "fixture-v1",
        42,
    )
    config = resolve(REPO / "configs/frozen.yaml", REPO)
    config["protocol"].update(seed=42, input_resolution=16)
    config["protocol"]["model"].update(id="fixture/model", revision="a" * 40)
    config["protocol"]["dataset"].update(
        version="fixture-v1",
        frozen_manifest="data.json",
        fingerprint=frozen["fingerprint"],
    )
    config["remote"].update(
        modal_gpu="T4",
        max_hours=1.0,
        max_cost_usd=1.0,
        estimated_hourly_usd=1.0,
        rate_checked_at="2026-10-11",
        kaggle_owner="fixture",
        kaggle_accelerator="NvidiaTeslaT4",
        dataset_hf_repo="fixture/data",
        dataset_hf_revision="b" * 40,
    )
    write(tmp_path / "config.yaml", config)
    return tmp_path, config, rows, {"fixture": source}
