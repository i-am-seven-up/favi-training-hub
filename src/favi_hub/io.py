"""Small deterministic file primitives shared by local and remote runners."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml


def now():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    return yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
            if path.suffix in {".yaml", ".yml"}
            else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must stay inside {root}: {relative}")
    return path


def code_files(root):
    root = Path(root)
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for folder in ("src", "runners"):
        paths.extend((root / folder).rglob("*.py"))
    return sorted(p for p in paths if p.is_file())


def code_hash(root):
    root = Path(root)
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in code_files(root)})
