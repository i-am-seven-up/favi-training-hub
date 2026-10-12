import math
import re
from pathlib import Path

from .io import digest, read

LABELS = {"Safe": 0, "Sexual": 1, "Violence": 2}


def resolve(path, root):
    cfg = read(path)
    if isinstance(cfg.get("protocol"), str):
        cfg["protocol"] = read(Path(root) / cfg["protocol"])
    validate(cfg)
    return cfg


def validate(cfg):
    p = cfg["protocol"]
    if p.get("labels") != LABELS or p.get("primary_metric") != "macro_f1":
        raise ValueError("Protocol must use canonical labels and macro_f1")
    if cfg.get("method") not in {"frozen", "lora"}:
        raise ValueError("method must be frozen or lora")
    pilot = cfg["pilot"]
    if pilot.get("basis") != "optimizer_steps":
        raise ValueError("v1 supports optimizer_steps only; data_fraction needs a separate protocol")
    if not 0.05 <= pilot["fraction"] <= 0.1:
        raise ValueError("pilot fraction must be 5–10%")
    t = cfg["training"]
    for key in ("epochs", "batch_size", "accumulation_steps", "eval_every", "checkpoint_every"):
        if type(t.get(key)) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not math.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= t["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio must be in [0,1)")
    if t["precision"] != "fp32":
        raise ValueError("v1 supports fp32 only for a controlled cross-backend comparison")
    if cfg["method"] == "lora":
        lora = cfg["lora"]
        if lora["r"] < 1 or lora["alpha"] <= 0 or not 0 <= lora["dropout"] < 1:
            raise ValueError("Invalid LoRA parameters")
    return cfg


def readiness(cfg):
    p = cfg["protocol"]
    issues = []
    if type(p.get("seed")) is not int:
        issues.append("protocol.seed is not locked")
    if p.get("model", {}).get("family") != "clip":
        issues.append("v1 model integration supports CLIP vision only")
    for key in ("id", "revision"):
        if not p.get("model", {}).get(key):
            issues.append(f"model.{key} is missing")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", p.get("model", {}).get("revision") or ""):
        issues.append("model.revision must be an immutable 40-character commit")
    if not isinstance(p.get("input_resolution"), int) or p["input_resolution"] <= 0:
        issues.append("input_resolution is missing")
    for key in ("version", "frozen_manifest", "fingerprint"):
        if not p.get("dataset", {}).get(key):
            issues.append(f"dataset.{key} is missing")
    if cfg["method"] == "lora" and not cfg["lora"]["target_modules"]:
        issues.append("lora.target_modules is not selected")
    return issues


def steps(cfg, count):
    t = cfg["training"]
    batches = math.ceil(count / t["batch_size"])
    full = math.ceil(batches / t["accumulation_steps"]) * t["epochs"]
    return {"full": full, "pilot": max(1, math.ceil(full * cfg["pilot"]["fraction"])),
            "warmup": math.ceil(full * t["warmup_ratio"])}


def comparison_key(cfg):
    # Method, optimizer LR and LoRA rank differ by design; protocol and training budget do not.
    t = cfg["training"]
    return digest({"protocol": cfg["protocol"], "epochs": t["epochs"],
                   "effective_batch": t["batch_size"] * t["accumulation_steps"],
                   "precision": t["precision"], "metric_version": 1})
