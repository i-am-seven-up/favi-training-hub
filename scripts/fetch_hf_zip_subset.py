"""Fetch a deterministic, class-balanced subset from one ZIP in a Hugging Face dataset repo."""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import HfFileSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.per_label < 1:
        raise ValueError("--per-label must be positive")

    remote = f"datasets/{args.repo}@{args.revision}/{args.file}"
    selected = []
    with HfFileSystem().open(remote, "rb") as stream, zipfile.ZipFile(stream) as archive:
        by_label = defaultdict(list)
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            for label in args.labels:
                if label in path.parts:
                    by_label[label].append(member)
                    break
        rng = random.Random(args.seed)
        for label in args.labels:
            members = by_label[label]
            if len(members) < args.per_label:
                raise ValueError(f"{label} has only {len(members)} eligible files")
            for member in rng.sample(members, args.per_label):
                destination = args.out / label / PurePosixPath(member.filename).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
                selected.append({"label": label, "archive_path": member.filename,
                                 "local_path": destination.relative_to(args.out).as_posix()})

    metadata = {"repo": args.repo, "revision": args.revision, "archive": args.file,
                "seed": args.seed, "per_label": args.per_label, "count": len(selected),
                "purpose": "source inspection subset; not a canonical training dataset",
                "records": selected}
    (args.out / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("repo", "revision", "count", "purpose")}))


if __name__ == "__main__":
    main()
