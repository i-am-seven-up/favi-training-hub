"""Dynamically configure a detached Modal job; imported only on explicit live submit."""

import math
from pathlib import Path


def submit_job(root, run_id, cfg, project):
    import modal

    from favi_hub.runs import run_folder, update

    r = cfg["remote"]
    app = modal.App("favi-training-hub")
    volume = modal.Volume.from_name("favi-training-cache", create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("uv==0.9.9")
             .add_local_dir(str(project), "/opt/favi", copy=True)
             .run_commands("cd /opt/favi && uv sync --frozen --no-dev --extra train --extra remote")
             .env({"PYTHONPATH": "/opt/favi/src", "HF_HOME": "/cache/hf",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
    timeout = min(86400, math.ceil(min(r["max_hours"], r["max_cost_usd"] / r["estimated_hourly_usd"]) * 3600))

    @app.function(image=image, gpu=r["modal_gpu"], timeout=timeout, retries=0,
                  volumes={"/cache": volume}, secrets=[modal.Secret.from_name("favi-hf")])
    def worker(job_id):
        import json
        import shutil
        import subprocess

        job_root = Path("/cache/jobs") / job_id / "project"
        if job_root.exists():
            raise ValueError("Existing job workspace: inspect checkpoint before explicit recovery")
        shutil.copytree("/opt/favi", job_root, ignore=shutil.ignore_patterns(".venv"))
        # Run locked environment's Python, not the Modal container's base interpreter.
        subprocess.run(["/opt/favi/.venv/bin/python", "-m", "favi_hub", "--root", str(job_root),
                        "remote-work", "--run", job_id, "--cache", "/cache"], check=True)
        volume.commit()
        folder = job_root / "experiments" / job_id
        record = json.loads((folder / "run.json").read_text())
        record["artifact_uri"] = f"modal-volume://favi-training-cache/jobs/{job_id}/project/experiments/{job_id}/artifact"
        (folder / "run.json").write_text(json.dumps(record))
        volume.commit()
        result = {"run": record}
        for name in ("metrics", "calibration"):
            result[name] = json.loads((folder / f"{name}.json").read_text()) if (folder / f"{name}.json").exists() else None
        result["history"] = (folder / "history.csv").read_text() if (folder / "history.csv").exists() else None
        return result

    with app.run(detach=True):
        call = worker.spawn(run_id)
        return update(run_folder(root, run_id), status="submitted", remote_id=call.object_id)
