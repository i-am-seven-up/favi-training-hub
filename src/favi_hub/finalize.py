from pathlib import Path

from .io import digest, now, read, write
from .runs import experiment_config, run_folder


def lock_selection(root, run_ids, reviewer, reason):
    path = Path(root) / "results/final-selection.json"
    if path.exists():
        raise ValueError("Final selection already locked; do not retune using held-out test")
    if not reviewer or not reason or len(set(run_ids)) < 2:
        raise ValueError("Select baseline + LoRA, reviewer and validation-based reason")
    records, keys, methods = {}, set(), set()
    for rid in run_ids:
        folder = run_folder(root, rid)
        run = read(folder / "run.json")
        if run["status"] != "completed" or run["stage"] != "full" or run["mock"]:
            raise ValueError("Final selection needs completed non-mock full runs")
        experiment_config(root, run["experiment_id"])
        metrics = read(folder / "metrics.json")
        if metrics["split"] != "validation" or not metrics["coverage_complete"]:
            raise ValueError("Complete validation evidence required")
        keys.add((run["comparison_key"], run["code_hash"], metrics["split_fingerprint"]))
        methods.add(run["method"])
        records[rid] = {"config_hash": run["config_hash"], "validation_hash": digest(metrics)}
    if len(keys) != 1 or methods != {"frozen", "lora"}:
        raise ValueError("Selection must contain comparable frozen and LoRA runs")
    value = {"runs": records, "reviewer": reviewer, "reason": reason, "locked_at": now()}
    write(path, value)
    return value


def evaluate_artifact(root, run_id, data_root, artifact, split):
    import torch

    from .data import verify_frozen
    from .evaluate import split_fingerprint
    from .io import file_hash, inside
    from .models import load_model
    from .train import evaluate_model

    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    cfg, _ = experiment_config(root, run["experiment_id"])
    if run["status"] != "completed" or run.get("mock"):
        raise ValueError("Only completed research artifacts can be evaluated")
    if split == "test":
        selection = read(Path(root) / "results/final-selection.json")
        selected = selection["runs"].get(run_id)
        if not selected or selected["config_hash"] != run["config_hash"]:
            raise ValueError("Run not in locked final selection")
        if selected["validation_hash"] != digest(read(folder / "metrics.json")):
            raise ValueError("Validation evidence changed after selection")
    destination = folder / f"metrics.{split}.json"
    if destination.exists():
        raise ValueError("Evaluation already exists; preserve the result")
    artifact = Path(artifact)
    identity = read(artifact / "identity.json")
    for key in ("run_id", "config_hash", "code_hash", "dataset_fingerprint"):
        if identity[key] != run[key]:
            raise ValueError("Artifact identity mismatch")
    model, processor = load_model(cfg)
    if cfg["method"] == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model.encoder, load_file(str(artifact / "adapter/adapter_model.safetensors")))
    model.head.load_state_dict(torch.load(artifact / "head.pt", weights_only=True, map_location="cpu"))
    data = verify_frozen(inside(root, cfg["protocol"]["dataset"]["frozen_manifest"]))
    rows = [r for r in data["records"] if r["split"] == split]
    for row in rows:
        if file_hash(inside(data_root, row["image_uri"])) != row["sha256"]:
            raise ValueError("Evaluation image checksum mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_model(model, processor, rows, data_root, cfg["training"]["batch_size"], device)
    metrics.update(split=split, split_fingerprint=split_fingerprint(data["records"], split), mock=False,
                   config_hash=run["config_hash"], run_id=run_id)
    write(destination, metrics)
    return metrics
