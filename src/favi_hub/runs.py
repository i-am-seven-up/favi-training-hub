import math
import re
from pathlib import Path

from .config import comparison_key, readiness, resolve, steps
from .data import verify_frozen
from .evaluate import split_fingerprint
from .io import code_hash, digest, inside, now, read, write


def prepare(root, config_path):
    root = Path(root)
    cfg = resolve(config_path, root)
    folder = root / "experiments"
    folder.mkdir(exist_ok=True)
    for number in range(1, 100000):
        experiment = folder / f"EXP-{number:03d}"
        try:
            experiment.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("Experiment IDs exhausted")
    write(experiment / "config.yaml", cfg)
    metadata = {"experiment_id": experiment.name, "created_at": now(), "config_hash": digest(cfg),
                "code_hash": code_hash(root), "comparison_key": comparison_key(cfg), "method": cfg["method"]}
    write(experiment / "experiment.json", metadata)
    return metadata


def experiment_config(root, experiment_id):
    if not re.fullmatch(r"EXP-\d{3,}", experiment_id):
        raise ValueError("Invalid experiment ID")
    folder = Path(root) / "experiments" / experiment_id
    cfg, meta = read(folder / "config.yaml"), read(folder / "experiment.json")
    if digest(cfg) != meta["config_hash"] or code_hash(root) != meta["code_hash"]:
        raise ValueError("Experiment config/code changed; prepare a new experiment")
    return cfg, meta


def dataset_ready(root, cfg):
    issues = readiness(cfg)
    if issues:
        raise ValueError("Research not ready: " + "; ".join(issues))
    data = verify_frozen(inside(root, cfg["protocol"]["dataset"]["frozen_manifest"]))
    expected = cfg["protocol"]["dataset"]
    if data["fingerprint"] != expected["fingerprint"] or data["version"] != expected["version"]:
        raise ValueError("Dataset fingerprint/version differs from protocol")
    if data["seed"] != cfg["protocol"]["seed"]:
        raise ValueError("Dataset and protocol seeds differ")
    return data


def new_run(root, experiment_id, stage, backend):
    cfg, meta = experiment_config(root, experiment_id)
    if (stage, backend) not in {("pilot", "local"), ("pilot", "kaggle"), ("full", "modal")}:
        raise ValueError("Only local/Kaggle pilot and Modal full are research training runs")
    exp = Path(root) / "experiments" / experiment_id
    # Prevent duplicate submitted/running/uncertain jobs, not merely duplicate IDs.
    for existing in exp.glob("*/run.json"):
        state = read(existing)
        if state["status"] in {"prepared", "submitting", "submitted", "running", "submission_unknown"}:
            raise ValueError(f"Reconcile existing run before resubmission: {state['run_id']}")
    for i in range(1, 100000):
        folder = exp / f"{stage}-{i:02d}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            continue
    record = {**meta, "run_id": f"{experiment_id}/{folder.name}", "stage": stage, "backend": backend,
              "status": "prepared", "updated_at": now(), "mock": False, "remote_id": None,
              "dataset_fingerprint": cfg["protocol"]["dataset"]["fingerprint"],
              "protocol_hash": digest(cfg["protocol"]), "artifact_uri": None}
    write(folder / "run.json", record)
    return folder, record


def run_folder(root, run_id):
    if not re.fullmatch(r"EXP-\d{3,}/(?:pilot|full)-\d{2,}", run_id):
        raise ValueError("Invalid run ID; use EXP-001/pilot-01")
    return inside(Path(root) / "experiments", run_id)


def update(folder, **values):
    record = read(Path(folder) / "run.json")
    record.update(values, updated_at=now())
    write(Path(folder) / "run.json", record)
    return record


def validate_pilot(root, run_id):
    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    cfg, _ = experiment_config(root, run["experiment_id"])
    data = dataset_ready(root, cfg)
    if (run["mock"] or run["backend"] not in {"local", "kaggle"}
            or run["stage"] != "pilot" or run["status"] != "completed"):
        raise ValueError("Approval requires a completed real local or Kaggle pilot")
    if run["config_hash"] != digest(cfg) or run["dataset_fingerprint"] != data["fingerprint"]:
        raise ValueError("Pilot fingerprint mismatch")
    metrics = read(folder / "metrics.json")
    if metrics.get("mock") or metrics.get("split") != "validation" or not metrics.get("coverage_complete"):
        raise ValueError("Pilot needs real validation metrics covering all classes")
    if metrics.get("split_fingerprint") != split_fingerprint(data["records"], "validation"):
        raise ValueError("Validation fingerprint mismatch")
    if not metrics.get("finite_gradients") or not metrics.get("gradient_flow_verified"):
        raise ValueError("Pilot gradients not verified")
    budget = steps(cfg, sum(r["split"] == "train" for r in data["records"]))
    if metrics.get("completed_steps") != budget["pilot"]:
        raise ValueError("Pilot did not complete its configured budget")
    if metrics["completed_steps"] <= budget["warmup"]:
        raise ValueError("Pilot is still in warmup: review as inconclusive")
    for key in ("macro_f1", "train_seconds", "peak_vram_gb"):
        if metrics.get(key) is None or not math.isfinite(metrics[key]):
            raise ValueError(f"Pilot missing finite {key}")
    return run, cfg, metrics


def review(root, run_id, decision, reviewer, reason):
    folder = run_folder(root, run_id)
    if (folder / "review.json").exists():
        raise ValueError("Review already exists; preserve history and create a new pilot attempt")
    if decision not in {"revise", "inconclusive", "ready_for_modal", "stop"} or not reviewer or not reason:
        raise ValueError("Review requires decision, reviewer and evidence-based reason")
    run = read(folder / "run.json")
    if decision == "ready_for_modal":
        _, cfg, _ = validate_pilot(root, run_id)
        validate_budget(cfg)
    record = {"decision": decision, "reviewer": reviewer, "reason": reason, "reviewed_at": now(),
              "run_id": run_id, "config_hash": run["config_hash"], "code_hash": run["code_hash"],
              "dataset_fingerprint": run["dataset_fingerprint"],
              "metrics_hash": digest(read(folder / "metrics.json")) if (folder / "metrics.json").exists() else None}
    write(folder / "review.json", record)
    return record


def validate_budget(cfg):
    remote = cfg["remote"]
    for key in ("max_hours", "max_cost_usd", "estimated_hourly_usd"):
        v = remote.get(key)
        if not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"remote.{key} must be a finite positive budget")
    if not remote.get("modal_gpu") or not remote.get("rate_checked_at"):
        raise ValueError("Target GPU and dated total hourly-rate estimate required")
    if not isinstance(remote.get("calibration_steps"), int) or remote["calibration_steps"] < 2:
        raise ValueError("calibration_steps must be at least 2")


def modal_gate(root, experiment_id):
    cfg, _ = experiment_config(root, experiment_id)
    dataset_ready(root, cfg)
    validate_budget(cfg)
    pilots = sorted((Path(root) / "experiments" / experiment_id).glob("pilot-*/run.json"))
    if not pilots:
        raise ValueError("No approved pilot candidate exists")
    folder = pilots[-1].parent
    run_id = f"{experiment_id}/{folder.name}"
    run, _, metrics = validate_pilot(root, run_id)
    approval = read(folder / "review.json")
    if approval["decision"] != "ready_for_modal":
        raise ValueError("Latest pilot has no ready_for_modal decision")
    if (approval["metrics_hash"] != digest(metrics) or approval["config_hash"] != run["config_hash"]
            or approval["code_hash"] != run["code_hash"]
            or approval["dataset_fingerprint"] != run["dataset_fingerprint"]):
        raise ValueError("Stale review")
    return {"ok": True, "pilot_run_id": run_id, "review_hash": digest(approval)}


def calibration_decision(cfg, seconds_per_step, full_steps, elapsed_seconds=0):
    validate_budget(cfg)
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError("Invalid target-GPU calibration throughput")
    # Conservative range, including eval/checkpoint overhead and elapsed setup/calibration.
    low = (elapsed_seconds + full_steps * seconds_per_step * 1.2) / 3600
    high = (elapsed_seconds + full_steps * seconds_per_step * 1.8) / 3600
    cost = high * cfg["remote"]["estimated_hourly_usd"]
    return {"ok": high <= cfg["remote"]["max_hours"] and cost <= cfg["remote"]["max_cost_usd"],
            "estimated_hours_range": [low, high], "estimated_cost_upper_usd": cost,
            "assumptions": "Target GPU steady-state; 20–80% eval/checkpoint/I/O allowance; not a billing guarantee"}


def collect_result(root, run_id, incoming):
    folder = run_folder(root, run_id)
    expected = read(folder / "run.json")
    received = read(Path(incoming) / "run.json")
    for key in ("run_id", "experiment_id", "config_hash", "code_hash", "dataset_fingerprint", "stage", "backend"):
        if received.get(key) != expected.get(key):
            raise ValueError(f"Collected run mismatch: {key}")
    if received.get("mock"):
        raise ValueError("Cannot collect mock as research run")
    if received.get("status") not in {"completed", "failed", "blocked", "cancelled"}:
        raise ValueError("Remote result is not terminal")
    if received["status"] == "completed" and not (Path(incoming) / "metrics.json").exists():
        raise ValueError("Completed result has no metrics")
    for filename in ("metrics.json", "calibration.json"):
        source = Path(incoming) / filename
        target = folder / filename
        if source.exists():
            value = read(source)
            if target.exists() and read(target) != value:
                raise ValueError(f"Conflicting existing {filename}")
            write(target, value)
    history = Path(incoming) / "history.csv"
    if history.exists():
        target = folder / "history.csv"
        if target.exists() and target.read_bytes() != history.read_bytes():
            raise ValueError("Conflicting history.csv")
        target.write_bytes(history.read_bytes())
    write(folder / "run.json", received)
    return received
