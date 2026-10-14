import pytest

from favi_hub.config import steps
from favi_hub.evaluate import classification_metrics, split_fingerprint
from favi_hub.io import read, write
from favi_hub.runs import calibration_decision, local_full_gate, modal_gate, new_run, prepare, review, update


def completed_pilot(research):
    root, config, rows, _ = research
    experiment = prepare(root, root / "config.yaml")
    folder, run = new_run(root, experiment["experiment_id"], "pilot", "local")
    update(folder, status="completed")
    metrics = classification_metrics([0, 1, 2], [0, 0, 2])
    metrics.update(
        split="validation",
        mock=False,
        split_fingerprint=split_fingerprint(rows, "validation"),
        finite_gradients=True,
        gradient_flow_verified=True,
        completed_steps=steps(config, 90)["pilot"],
        train_seconds=5.0,
        peak_vram_gb=1.0,
    )
    write(folder / "metrics.json", metrics)
    return root, config, folder, run


def test_missing_pilot_blocks_modal(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    with pytest.raises(ValueError, match="No approved pilot"):
        modal_gate(root, experiment["experiment_id"])


def test_low_metric_can_pass_structural_gate(research):
    root, _, _, run = completed_pilot(research)
    review(root, run["run_id"], "ready_for_modal", "reviewer", "Pipeline and learning signal valid")
    assert modal_gate(root, run["experiment_id"])["ok"]


def test_stale_metrics_invalidate_review(research):
    root, _, folder, run = completed_pilot(research)
    review(root, run["run_id"], "ready_for_modal", "reviewer", "Pipeline valid")
    metrics = read(folder / "metrics.json")
    metrics["macro_f1"] = 0.99
    write(folder / "metrics.json", metrics)
    with pytest.raises(ValueError, match="Stale"):
        modal_gate(root, run["experiment_id"])


@pytest.mark.parametrize(("status", "mock"), [("failed", False), ("completed", True)])
def test_failed_or_mock_pilot_cannot_be_approved(research, status, mock):
    root, _, folder, run = completed_pilot(research)
    update(folder, status=status, mock=mock)
    with pytest.raises(ValueError):
        review(root, run["run_id"], "ready_for_modal", "reviewer", "Invalid")


def test_inconclusive_review_can_record_failure(research):
    root, _, folder, run = completed_pilot(research)
    update(folder, status="failed")
    result = review(root, run["run_id"], "inconclusive", "reviewer", "Pilot was interrupted")
    assert result["decision"] == "inconclusive"


def test_calibration_respects_time_and_cost_budget(research):
    _, config, *_ = research
    assert calibration_decision(config, 0.01, 100)["ok"]
    assert not calibration_decision(config, 100, 10000)["ok"]


def test_pilot_fraction_reduces_optimizer_steps_once(research):
    _, config, *_ = research
    assert steps(config, 90) == {"full": 36, "pilot": 4, "warmup": 2}


def test_duplicate_inflight_submit_blocked(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    new_run(root, experiment["experiment_id"], "pilot", "local")
    with pytest.raises(ValueError, match="Reconcile"):
        new_run(root, experiment["experiment_id"], "pilot", "local")


def test_kaggle_remains_an_accepted_pilot_fallback(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    _, run = new_run(root, experiment["experiment_id"], "pilot", "kaggle")
    assert run["backend"] == "kaggle"


def test_completed_comparable_pilot_can_gate_local_full(research):
    root, _, _, pilot = completed_pilot(research)
    target = prepare(root, root / "config.yaml")
    assert local_full_gate(root, target["experiment_id"], pilot["run_id"])["ok"]
    _, full = new_run(root, target["experiment_id"], "full", "local")
    assert full["stage"] == "full"
    assert full["backend"] == "local"
