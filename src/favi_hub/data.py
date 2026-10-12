import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .config import LABELS
from .io import digest, file_hash, inside, now, read, write

SPLITS = ("train", "validation", "test")


def records_from(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = read(path)
    return value["records"] if isinstance(value, dict) else value


def audit(records, sources, evidence_root=None):
    errors, warnings = [], []
    ids, uris = set(), set()
    groups = defaultdict(set)
    class_counts = {s: Counter() for s in SPLITS}
    near_pairs = []
    by_dhash = defaultdict(list)
    for row in records:
        sid = row.get("sample_id", "<missing>")
        required = ("sample_id", "image_uri", "sha256", "canonical_label", "original_label",
                    "source_id", "source_revision", "subtype", "review_status", "split")
        for field in required:
            if not row.get(field):
                errors.append(f"{sid}: missing {field}")
        if sid in ids or row.get("image_uri") in uris:
            errors.append(f"{sid}: duplicate sample_id or image_uri")
        ids.add(sid)
        uris.add(row.get("image_uri"))
        uri = row.get("image_uri", "")
        if Path(uri).is_absolute() or ".." in Path(uri).parts or ":" in uri:
            errors.append(f"{sid}: image_uri must be relative to dataset root")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            errors.append(f"{sid}: invalid sha256")
        label, split = row.get("canonical_label"), row.get("split")
        if label not in LABELS or split not in SPLITS:
            errors.append(f"{sid}: invalid canonical_label or split")
        else:
            class_counts[split][label] += 1
        if row.get("review_status") != "approved":
            errors.append(f"{sid}: canonical review not approved")
        if label == "Safe" and (row.get("non_sexual") is not True or row.get("non_violent") is not True):
            errors.append(f"{sid}: Safe requires both non_sexual and non_violent review")
        source = sources.get(row.get("source_id"), {})
        if source.get("status") not in {"pass", "pass_with_limitation"}:
            errors.append(f"{sid}: source not verified/approved")
        for field in ("revision", "license", "license_evidence", "review_evidence"):
            if not source.get(field):
                errors.append(f"{sid}: source missing {field}")
        if source.get("revision") != row.get("source_revision"):
            errors.append(f"{sid}: source revision mismatch")
        if type(source.get("video_origin")) is not bool:
            errors.append(f"{sid}: unknown video provenance")
        if source.get("video_origin") and not row.get("group_id"):
            errors.append(f"{sid}: video-derived sample missing group_id")
        for field in ("sha256", "duplicate_group"):
            if row.get(field):
                groups[(field, row[field])].add(split)
        if row.get("group_id"):
            groups[("group", row["source_id"], row["group_id"])].add(split)
        if row.get("dhash"):
            if not re.fullmatch(r"[0-9a-f]{16}", row["dhash"]):
                errors.append(f"{sid}: invalid dhash")
            else:
                by_dhash[row["dhash"]].append(row)
    for key, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"cross-split leakage: {key}")
    # Exact perceptual-hash collisions are review candidates, not automatic duplicates.
    for rows in by_dhash.values():
        if len(rows) > 1:
            near_pairs.append([r["sample_id"] for r in rows])
            if any(not r.get("near_duplicate_reviewed") for r in rows):
                errors.append("Unresolved perceptual-hash duplicate candidates")
    for split in SPLITS:
        missing = set(LABELS) - set(class_counts[split])
        if missing:
            errors.append(f"{split}: missing classes {sorted(missing)}")
    if evidence_root is not None:
        for source_id in {r.get("source_id") for r in records}:
            src = sources.get(source_id, {})
            review_path = src.get("review_evidence")
            try:
                evidence = read(inside(evidence_root, review_path)) if review_path else {}
                if (not evidence.get("reviewer") or not evidence.get("reviewed_at")
                        or not evidence.get("sample_ids") or not evidence.get("notes")
                        or evidence.get("decision") not in {"pass", "pass_with_limitation"}
                        or evidence.get("source_revision") != src.get("revision")):
                    errors.append(f"{source_id}: incomplete source review evidence")
            except (OSError, ValueError, TypeError):
                errors.append(f"{source_id}: review evidence unreadable")
    warnings.append("dHash collisions only: cropped/edited near-duplicates require additional manual audit")
    return {"ok": not errors, "errors": sorted(set(errors)), "warnings": warnings,
            "split_counts": {s: dict(c) for s, c in class_counts.items()},
            "source_counts": dict(Counter(r.get("source_id") for r in records)),
            "near_duplicate_candidates": near_pairs, "count": len(records)}


def freeze(manifest, registry, destination, root, version, seed):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Frozen manifest already exists; choose a new version/path")
    records = sorted(records_from(manifest), key=lambda r: r["sample_id"])
    sources = read(registry)["sources"]
    used = {r["source_id"]: copy.deepcopy(sources[r["source_id"]]) for r in records}
    report = audit(records, used, root)
    if not report["ok"]:
        raise ValueError("Dataset not ready: " + "; ".join(report["errors"]))
    for source in used.values():
        evidence = inside(root, source["review_evidence"])
        source["review_evidence_sha256"] = file_hash(evidence)
    payload = {"schema_version": 1, "version": version, "seed": seed,
               "records": records, "sources": used, "mock": False}
    value = {**payload, "fingerprint": digest(payload), "frozen_at": now(), "audit": report}
    write(destination, value)
    return value


def verify_frozen(path):
    data = read(path)
    payload = {k: data[k] for k in ("schema_version", "version", "seed", "records", "sources", "mock")}
    if digest(payload) != data["fingerprint"]:
        raise ValueError("Frozen manifest fingerprint mismatch")
    result = audit(data["records"], data["sources"])
    if not result["ok"] or data["mock"]:
        raise ValueError("Frozen research dataset is invalid or synthetic")
    return data


def image_inventory(folder):
    folder = Path(folder)
    rows, errors = [], []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                small = list(im.convert("L").resize((9, 8)).getdata())
                bits = [small[y * 9 + x] > small[y * 9 + x + 1] for y in range(8) for x in range(8)]
                dhash = sum(int(v) << i for i, v in enumerate(bits))
                rows.append({"image_uri": path.relative_to(folder).as_posix(), "sha256": file_hash(path),
                             "width": im.width, "height": im.height, "dhash": f"{dhash:016x}",
                             "original_label": path.parent.name})
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": type(exc).__name__})
    return {"count": len(rows), "records": rows, "errors": errors,
            "label_counts": dict(Counter(r["original_label"] for r in rows))}


def sample_sheet(folder, count, seed, destination):
    inventory = image_inventory(folder)
    chosen = random.Random(seed).sample(inventory["records"], min(count, inventory["count"]))
    if not chosen:
        raise ValueError("No readable images in source")
    canvas = Image.new("RGB", (5 * 180, ((len(chosen) + 4) // 5) * 210), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(chosen):
        with Image.open(Path(folder) / row["image_uri"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((174, 174))
            x, y = i % 5 * 180, i // 5 * 210
            canvas.paste(im, (x, y))
            draw.text((x, y + 176), f"{i + 1}: {row['sha256'][:12]}", fill="black")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    canvas.save(destination / "contact-sheet.jpg")
    write(destination / "sample.json", {"seed": seed, "records": chosen, "review_status": "pending"})
    return {"count": len(chosen), "review_sheet": str(destination / "contact-sheet.jpg")}
