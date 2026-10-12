"""Shared image classification loop; deterministic sample order, bounded pilot, resumable state."""

import csv
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .config import LABELS, steps
from .evaluate import classification_metrics, split_fingerprint
from .io import digest, file_hash, inside, read, write
from .models import load_model
from .runs import calibration_decision, dataset_ready, experiment_config, run_folder, update


def batch_plan(count, cfg):
    t = cfg["training"]
    for epoch in range(t["epochs"]):
        order = list(range(count))
        random.Random(cfg["protocol"]["seed"] + epoch).shuffle(order)
        micro = [order[i:i + t["batch_size"]] for i in range(0, count, t["batch_size"])]
        for i in range(0, len(micro), t["accumulation_steps"]):
            yield micro[i:i + t["accumulation_steps"]]


def pixels_for(records, processor, data_root, device):
    images = []
    for row in records:
        with Image.open(inside(data_root, row["image_uri"])) as im:
            images.append(ImageOps.exif_transpose(im).convert("RGB"))
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    labels = torch.tensor([LABELS[r["canonical_label"]] for r in records], device=device)
    return pixels, labels


def evaluate_model(model, processor, records, data_root, batch_size, device):
    model.eval()
    targets, predictions = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            pixels, labels = pixels_for(records[i:i + batch_size], processor, data_root, device)
            logits = model(pixels)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
    return {**classification_metrics(targets, predictions), "loss": loss_sum / len(records)}


def save_checkpoint(path, model, optimizer, scheduler, completed_steps, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "completed_steps": completed_steps, "identity": identity, "rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer, scheduler, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Checkpoint identity mismatch; cross-stage resume is disallowed")
    parameters = dict(model.named_parameters())
    expected = {n for n, p in parameters.items() if p.requires_grad}
    if set(state["trainable"]) != expected:
        raise ValueError("Checkpoint trainable keys mismatch")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["completed_steps"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def execute(root, run_id, data_root, persist=None, resume=False):
    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    if run["status"] == "completed":
        raise ValueError("Run already completed")
    cfg, _ = experiment_config(root, run["experiment_id"])
    data = dataset_ready(root, cfg)
    from .runs import modal_gate

    if run["stage"] == "full":
        modal_gate(root, run["experiment_id"])
    started = time.perf_counter()
    try:
        update(folder, status="running")
        return _execute(root, folder, run, cfg, data, data_root, started, persist, resume)
    except torch.cuda.OutOfMemoryError:
        update(folder, status="failed", error_type="OutOfMemoryError")
        torch.cuda.empty_cache()
        if persist:
            persist()
        raise ValueError(
            "CUDA out of memory; create a new experiment with a smaller batch_size "
            "or use the Kaggle fallback"
        ) from None
    except Exception as exc:
        # Store error type only: provider exceptions may contain signed URLs or secrets.
        update(folder, status="failed", error_type=type(exc).__name__)
        if persist:
            persist()
        raise


def _execute(root, folder, run, cfg, data, data_root, started, persist, resume):
    train_rows = [r for r in data["records"] if r["split"] == "train"]
    val_rows = [r for r in data["records"] if r["split"] == "validation"]
    for row in train_rows + val_rows:
        if file_hash(inside(data_root, row["image_uri"])) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['sample_id']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("Research training expects CUDA; install the CUDA train extra or use smoke for core checks")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model, processor = load_model(cfg)
    model.to(device)
    t, budget = cfg["training"], steps(cfg, len(train_rows))
    max_steps = budget["pilot"] if run["stage"] == "pilot" else budget["full"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=t["learning_rate"], weight_decay=t["weight_decay"])

    def lr_factor(step):
        if step < budget["warmup"]:
            return (step + 1) / max(1, budget["warmup"])
        return max(0.0, (budget["full"] - step) / max(1, budget["full"] - budget["warmup"]))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    identity = {k: run[k] for k in ("run_id", "stage", "config_hash", "code_hash", "dataset_fingerprint")}
    checkpoint = folder / "checkpoint/latest.pt"
    initial = folder / "checkpoint/initial.pt"
    completed = 0
    if resume:
        completed = restore_checkpoint(checkpoint, model, optimizer, scheduler, identity)
    elif checkpoint.exists():
        raise ValueError("Checkpoint exists; use explicit same-run resume")
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.perf_counter() - started
    history_path = folder / "history.csv"
    if not resume:
        history_path.write_text("step,learning_rate,train_loss,validation_loss,macro_f1\n", encoding="utf-8")
    if run["stage"] == "full" and not resume:
        save_checkpoint(initial, model, optimizer, scheduler, 0, identity)
        timings = []
        for chunks in list(batch_plan(len(train_rows), cfg))[:cfg["remote"]["calibration_steps"]]:
            _sync(device)
            begin = time.perf_counter()
            _train_step(model, processor, train_rows, chunks, data_root, device, optimizer)
            _sync(device)
            timings.append(time.perf_counter() - begin)
        if len(timings) < 2:
            raise ValueError("Not enough steps for target-GPU calibration")
        seconds_per_step = sum(timings[1:]) / len(timings[1:])
        decision = calibration_decision(cfg, seconds_per_step, budget["full"], time.perf_counter() - started)
        decision["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        write(folder / "calibration.json", decision)
        if not decision["ok"]:
            update(folder, status="blocked", reason="Target-GPU resource estimate exceeds configured budget")
            if persist:
                persist()
            return decision
        restore_checkpoint(initial, model, optimizer, scheduler, identity)
        torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    loss_first, loss_last, observed = None, None, set()
    gradients = {"head": False, "adapter": cfg["method"] == "frozen"}
    for step_index, chunks in enumerate(batch_plan(len(train_rows), cfg)):
        if step_index < completed:
            continue
        if step_index >= max_steps:
            break
        if run["stage"] == "full":
            elapsed_hours = (time.perf_counter() - started) / 3600
            r = cfg["remote"]
            if elapsed_hours >= min(r["max_hours"], r["max_cost_usd"] / r["estimated_hourly_usd"]):
                raise TimeoutError("Runtime budget exhausted")
        lr = optimizer.param_groups[0]["lr"]
        loss_last, flow = _train_step(model, processor, train_rows, chunks, data_root, device, optimizer)
        for key, active in flow.items():
            gradients[key] = gradients[key] or active
        scheduler.step()
        completed = step_index + 1
        if loss_first is None:
            loss_first = loss_last
        observed.update(train_rows[i]["sample_id"] for chunk in chunks for i in chunk)
        measured = {}
        if completed % t["eval_every"] == 0 or completed == max_steps:
            measured = evaluate_model(model, processor, val_rows, data_root, t["batch_size"], device)
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([completed, lr, loss_last, measured.get("loss"), measured.get("macro_f1")])
        if completed % t["checkpoint_every"] == 0 or completed == max_steps:
            save_checkpoint(checkpoint, model, optimizer, scheduler, completed, identity)
            if persist:
                persist()
    _sync(device)
    seconds = time.perf_counter() - train_started
    metrics = evaluate_model(model, processor, val_rows, data_root, t["batch_size"], device)
    params = model.check_trainable()
    metrics.update({**params, "trainable_fraction": params["trainable_params"] / params["total_params"],
                    "split": "validation", "split_fingerprint": split_fingerprint(data["records"], "validation"),
                    "mock": False, "train_seconds": seconds, "setup_seconds": setup_seconds,
                    "timing_scope": "current invocation; includes periodic evaluation/checkpoint",
                    "completed_steps": completed, "planned_full_steps": budget["full"],
                    "unique_train_images_this_invocation": len(observed), "loss_first": loss_first,
                    "loss_last": loss_last, "finite_gradients": True,
                    "gradient_flow_verified": all(gradients.values()),
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
                    "peak_reserved_vram_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
                    "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
                    "resumed": resume})
    write(folder / "metrics.json", metrics)
    artifact = folder / "artifact"
    artifact.mkdir(exist_ok=True)
    if cfg["method"] == "lora":
        model.encoder.save_pretrained(artifact / "adapter")
    torch.save(model.head.state_dict(), artifact / "head.pt")
    processor.save_pretrained(artifact)
    write(artifact / "config.yaml", cfg)
    write(artifact / "identity.json", {**identity, "labels": LABELS, "config_hash": digest(cfg)})
    update(folder, status="completed", completed_steps=completed,
           artifact_uri=f"{run['backend']}://{run['run_id']}/artifact")
    if persist:
        persist()
    return metrics


def _train_step(model, processor, rows, chunks, data_root, device, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = sum(map(len, chunks))
    loss_value = 0.0
    for chunk in chunks:
        pixels, labels = pixels_for([rows[i] for i in chunk], processor, data_root, device)
        loss = torch.nn.functional.cross_entropy(model(pixels), labels) * (len(chunk) / total)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        loss_value += loss.item()
    flow = {"head": False, "adapter": False}
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise ValueError("Nonfinite gradients")
            active = bool(p.grad.abs().sum() > 0)
            if name.startswith("head."):
                flow["head"] |= active
            if "lora_" in name:
                flow["adapter"] |= active
    optimizer.step()
    return loss_value, flow
