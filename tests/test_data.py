import copy

import pytest

from favi_hub.data import audit, verify_frozen
from favi_hub.io import read, write


def test_valid_manifest(research):
    root, _, rows, sources = research
    assert audit(rows, sources, root)["ok"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("canonical_label", "NSFW"), ("review_status", "pending"),
     ("non_violent", False), ("image_uri", "../secret.png"),
     ("sha256", "invalid"), ("source_revision", "changed")],
)
def test_invalid_sample_rejected(research, field, value):
    _, _, rows, sources = research
    rows[0][field] = value
    assert not audit(rows, sources)["ok"]


def test_safe_needs_both_concept_reviews(research):
    _, _, rows, sources = research
    rows[0].update(original_label="neutral", non_violent=None)
    assert any("Safe requires" in error for error in audit(rows, sources)["errors"])


@pytest.mark.parametrize("field", ["sha256", "group_id", "duplicate_group"])
def test_cross_split_leakage(research, field):
    _, _, rows, sources = research
    first_test = next(row for row in rows if row["split"] == "test")
    value = rows[0]["sha256"] if field == "sha256" else "same-scene"
    rows[0][field] = first_test[field] = value
    assert any("cross-split" in error for error in audit(rows, sources)["errors"])


def test_video_source_requires_group(research):
    _, _, rows, sources = research
    sources["fixture"]["video_origin"] = True
    assert any("group_id" in error for error in audit(rows, sources)["errors"])


def test_near_duplicate_candidates_require_review(research):
    _, _, rows, sources = research
    rows = copy.deepcopy(rows)
    rows[0]["dhash"] = rows[1]["dhash"] = "0" * 16
    assert not audit(rows, sources)["ok"]


def test_frozen_manifest_tamper_detected(research):
    root, *_ = research
    assert verify_frozen(root / "data.json")
    payload = read(root / "data.json")
    payload["records"][0]["canonical_label"] = "Violence"
    write(root / "data.json", payload)
    with pytest.raises(ValueError, match="fingerprint"):
        verify_frozen(root / "data.json")
