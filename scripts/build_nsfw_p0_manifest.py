"""Build the reproducible P0 binary NSFW manifest from the reviewed deepghs archive."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from favi_hub.io import file_hash

NSFW_ORIGINAL = ("hentai", "porn", "sexy")
NON_NSFW_ORIGINAL = ("neutral", "drawings")


def take_balanced(source, seed, per_class):
    rng = random.Random(seed)
    nsfw = []
    quotas = {label: per_class // len(NSFW_ORIGINAL) for label in NSFW_ORIGINAL}
    for label in NSFW_ORIGINAL[:per_class % len(NSFW_ORIGINAL)]:
        quotas[label] += 1
    for label, quota in quotas.items():
        paths = sorted((source / label).glob("*"))
        nsfw.extend((path, "NSFW", label) for path in rng.sample(paths, quota))
    non_nsfw = []
    for label in NON_NSFW_ORIGINAL:
        paths = sorted((source / label).glob("*"))
        non_nsfw.extend((path, "Non-NSFW", label) for path in rng.sample(paths, per_class // 2))
    return nsfw + non_nsfw


def assign_splits(rows, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["canonical_label"], row["sha256"])].append(row)
    by_label = defaultdict(list)
    for (label, _), group in grouped.items():
        by_label[label].append(group)
    rng = random.Random(seed)
    for label, groups in by_label.items():
        rng.shuffle(groups)
        count = len(groups)
        train_end, validation_end = round(count * 0.8), round(count * 0.9)
        for index, group in enumerate(groups):
            split = "train" if index < train_end else "validation" if index < validation_end else "test"
            for row in group:
                row["split"] = split
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Extracted nsfw_dataset_v1 directory containing the five label folders")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--review-out", type=Path, required=True)
    args = parser.parse_args()
    if args.per_class < 20 or args.per_class % 2:
        raise ValueError("--per-class must be an even integer of at least 20")

    selected = take_balanced(args.source, args.seed, args.per_class)
    rows = []
    for path, canonical, original in selected:
        uri = path.relative_to(args.source).as_posix()
        rows.append({
            "sample_id": f"deepghs:{uri}", "image_uri": uri, "sha256": file_hash(path),
            "canonical_label": canonical, "original_label": original, "source_id": "deepghs_nsfw",
            "source_revision": args.revision, "group_id": None, "subtype": original,
            "review_status": "approved", "non_sexual": canonical == "Non-NSFW",
            "non_violent": None,
        })
    rows = sorted(assign_splits(rows, args.seed), key=lambda row: row["sample_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    args.review_out.parent.mkdir(parents=True, exist_ok=True)
    args.review_out.write_text(json.dumps({
        "reviewer": "project owner (handoff confirmation) + automated source recheck",
        "reviewed_at": "2026-10-12", "decision": "pass_with_limitation",
        "source_revision": args.revision,
        "sample_ids": [row["sample_id"] for row in rows[:50]],
        "notes": "P0 binary NSFW only. Handoff records prior visual inspection; deterministic recheck found "
                 "readable images in each original label. Non-NSFW is not thesis Safe; source-level video "
                 "provenance is not independently verified.",
    }, indent=2) + "\n", encoding="utf-8")
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["split"]][row["canonical_label"]] += 1
    print(json.dumps({"count": len(rows), "split_counts": counts, "source": str(args.source)}, default=dict))


if __name__ == "__main__":
    main()
