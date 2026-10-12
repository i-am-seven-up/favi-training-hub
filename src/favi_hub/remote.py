"""Cloud adapters. No SDK import, network call or resource creation on dry-run."""

import base64
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from .io import code_files, inside, read, write
from .runs import collect_result, dataset_ready, experiment_config, modal_gate, new_run, run_folder, update


def bundle(root, experiment_id, destination):
    root = Path(root)
    cfg, _ = experiment_config(root, experiment_id)
    files = code_files(root)
    files += list((root / "experiments" / experiment_id).rglob("*.json"))
    files += [root / "experiments" / experiment_id / "config.yaml"]
    manifest = cfg["protocol"]["dataset"].get("frozen_manifest")
    if manifest:
        files.append(inside(root, manifest))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            # Explicit allowlist; no raw images, environment files, credentials or checkpoints.
            if any(part in {"artifact", "checkpoint"} for part in path.relative_to(root).parts):
                continue
            archive.write(path, path.relative_to(root).as_posix())
    return target


def unpack(archive, destination):
    with zipfile.ZipFile(io.BytesIO(archive)) as stream:
        for info in stream.infolist():
            target = inside(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.read(info))


def requirements(cfg, backend):
    remote = cfg["remote"]
    missing = []
    keys = ("dataset_hf_repo", "dataset_hf_revision")
    if backend == "kaggle":
        keys += ("kaggle_owner", "kaggle_accelerator")
    for key in keys:
        if not remote.get(key):
            missing.append(f"remote.{key}")
    if remote.get("dataset_hf_revision") and not re.fullmatch(r"[a-f0-9]{40}", remote["dataset_hf_revision"]):
        missing.append("dataset_hf_revision must be an immutable commit")
    return missing


def submit(root, experiment_id, backend, dry_run=False):
    cfg, _ = experiment_config(root, experiment_id)
    issues = requirements(cfg, backend)
    try:
        dataset_ready(root, cfg)
        if backend == "modal":
            modal_gate(root, experiment_id)
    except (OSError, ValueError, KeyError) as exc:
        issues.append(str(exc))
    if dry_run:
        return {"dry_run": True, "backend": backend, "experiment_id": experiment_id,
                "ready": not issues, "issues": issues, "network_calls": 0}
    if issues:
        raise ValueError("; ".join(issues))
    folder, run = new_run(root, experiment_id, "pilot" if backend == "kaggle" else "full", backend)
    package = Path(root) / "outputs" / "packages" / experiment_id / folder.name
    package.mkdir(parents=True, exist_ok=True)
    if backend == "kaggle":
        # Unique kernel per attempt makes interrupted submissions reconcilable.
        remote_id = f"{cfg['remote']['kaggle_owner']}/favi-{experiment_id.lower()}-{folder.name}"
        update(folder, status="submitting", remote_id=remote_id)
        archive = bundle(root, experiment_id, package / "bundle.zip")
        template = (Path(root) / "runners/kaggle/entry.py").read_text(encoding="utf-8")
        entry = template.replace('BUNDLE = ""', f'BUNDLE = {base64.b64encode(archive.read_bytes()).decode()!r}')
        entry = entry.replace('RUN_ID = ""', f'RUN_ID = {run["run_id"]!r}')
        (package / "entry.py").write_text(entry, encoding="utf-8")
        write(package / "kernel-metadata.json", {
            "id": remote_id, "title": f"FAVI {run['run_id']}", "code_file": "entry.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": cfg["remote"].get("kaggle_dataset_sources", []),
            "competition_sources": [], "kernel_sources": []})
        command = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(package),
                   "--accelerator", cfg["remote"]["kaggle_accelerator"]]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError):
            update(folder, status="submission_unknown")
            raise ValueError(f"Submission uncertain; reconcile Kaggle kernel {remote_id} before retry") from None
        return update(folder, status="submitted")
    update(folder, status="submitting")
    archive = bundle(root, experiment_id, package / "bundle.zip")
    unpack(archive.read_bytes(), package / "project")
    try:
        # Load the repo adapter without importing it for other backends/dry runs.
        import importlib.util

        spec = importlib.util.spec_from_file_location("favi_modal_runner", Path(root) / "runners/modal_app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.submit_job(root, run["run_id"], cfg, package / "project")
    except Exception:  # noqa: BLE001 - provider/SDK failures leave submission state uncertain
        update(folder, status="submission_unknown")
        raise ValueError("Modal submission uncertain; inspect workspace before another submission") from None


def status(root, run_id):
    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    if not run.get("remote_id"):
        return {"run_id": run_id, "status": run["status"], "action": "reconcile backend; remote ID missing"}
    if run["backend"] == "kaggle":
        p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", run["remote_id"]],
                           check=True, capture_output=True, text=True)
        # Provider status is observational; completed research status requires collected metrics.
        state = next((s for s in ("complete", "running", "queued", "error", "cancel")
                      if s in p.stdout.lower()), "unknown")
        update(folder, remote_status=state)
        return {"run_id": run_id, "local_status": run["status"], "remote_status": state}
    import modal

    try:
        result = modal.FunctionCall.from_id(run["remote_id"]).get(timeout=0)
        return {"run_id": run_id, "remote_status": result["run"]["status"], "action": "collect"}
    except TimeoutError:
        return {"run_id": run_id, "remote_status": "running_or_queued"}


def collect(root, run_id):
    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    incoming = Path(root) / "outputs/collected" / run_id
    incoming.mkdir(parents=True, exist_ok=True)
    if run["backend"] == "kaggle":
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", run["remote_id"],
                        "-p", str(incoming), "--file-pattern", "result-bundle.zip"],
                       check=True, capture_output=True, text=True)
        unpack((incoming / "result-bundle.zip").read_bytes(), incoming / "result")
        return collect_result(root, run_id, incoming / "result")
    import modal

    result = modal.FunctionCall.from_id(run["remote_id"]).get(timeout=0)
    write(incoming / "run.json", {**result["run"], "remote_id": run["remote_id"]})
    for name in ("metrics", "calibration"):
        if result.get(name) is not None:
            write(incoming / f"{name}.json", result[name])
    if result.get("history"):
        (incoming / "history.csv").write_text(result["history"], encoding="utf-8")
    return collect_result(root, run_id, incoming)


def remote_work(root, run_id, cache, persist=None):
    """Remote-only entry; dataset downloads are immutable and cached across runs."""
    from huggingface_hub import snapshot_download

    from .train import execute

    folder = run_folder(root, run_id)
    cfg, _ = experiment_config(root, run_id.split("/")[0])
    r = cfg["remote"]
    try:
        data_root = snapshot_download(repo_id=r["dataset_hf_repo"], repo_type="dataset",
                                      revision=r["dataset_hf_revision"], cache_dir=str(Path(cache) / "hf"))
        execute(root, run_id, data_root, persist=persist)
    except Exception as exc:  # noqa: BLE001 - remote worker must persist a terminal state
        update(folder, status="failed", error_type=type(exc).__name__)
    if persist:
        persist()
    result = {"run": read(folder / "run.json")}
    for name in ("metrics", "calibration"):
        result[name] = read(folder / f"{name}.json") if (folder / f"{name}.json").exists() else None
    result["history"] = (folder / "history.csv").read_text() if (folder / "history.csv").exists() else None
    return result


def publish_artifact(root, run_id):
    """Explicit upload only; refuses to create a repo or publish to a public repo."""
    from huggingface_hub import HfApi

    folder = run_folder(root, run_id)
    run = read(folder / "run.json")
    cfg, _ = experiment_config(root, run["experiment_id"])
    repo = cfg["remote"].get("artifact_hf_repo")
    if run["status"] != "completed" or not repo or not (folder / "artifact").is_dir():
        raise ValueError("Completed local artifact and configured private HF repo required")
    api = HfApi()
    if not api.repo_info(repo).private:
        raise ValueError("Artifact destination must be private")
    result = api.upload_folder(repo_id=repo, folder_path=str(folder / "artifact"), path_in_repo=run_id)
    return update(folder, artifact_uri=f"hf://{repo}@{result.oid}/{run_id}")
