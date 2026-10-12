import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from .config import readiness, resolve
from .data import audit, freeze, image_inventory, records_from, sample_sheet
from .io import read, write
from .report import build_report
from .runs import new_run, prepare, review, run_folder
from .smoke import smoke


def parser():
    p = argparse.ArgumentParser(description="FAVI action-to-result research CLI")
    p.add_argument("--root", type=Path, default=Path.cwd())
    top = p.add_subparsers(dest="command", required=True)
    doctor = top.add_parser("doctor")
    doctor.add_argument("--config", default="configs/frozen.yaml")
    data = top.add_parser("data").add_subparsers(dest="action", required=True)
    for action in ("inspect", "sample"):
        s = data.add_parser(action)
        s.add_argument("--source", required=True)
        s.add_argument("--registry", default="data/sources.yaml")
        s.add_argument("--out", type=Path)
        if action == "sample":
            s.add_argument("--count", type=int, default=50)
            s.add_argument("--seed", type=int, required=True)
    for action in ("audit", "freeze"):
        s = data.add_parser(action)
        s.add_argument("--manifest", required=True)
        s.add_argument("--registry", default="data/sources.yaml")
        if action == "freeze":
            s.add_argument("--out", required=True)
            s.add_argument("--version", required=True)
            s.add_argument("--seed", required=True, type=int)
    run = top.add_parser("run").add_subparsers(dest="action", required=True)
    for action in ("prepare", "smoke"):
        s = run.add_parser(action)
        s.add_argument("--config", default="configs/frozen.yaml")
    s = run.add_parser("local", help="Run or resume the bounded pilot on this machine's CUDA GPU")
    target = s.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment")
    target.add_argument("--resume-run")
    s.add_argument("--data-root", required=True)
    s = run.add_parser("submit")
    s.add_argument("--backend", required=True, choices=["kaggle", "modal"])
    s.add_argument("--experiment", required=True)
    s.add_argument("--dry-run", action="store_true")
    for action in ("status", "collect", "publish", "review"):
        s = run.add_parser(action)
        s.add_argument("--run", required=True)
        if action == "review":
            s.add_argument("--decision", choices=["revise", "inconclusive", "ready_for_modal", "stop"], required=True)
            s.add_argument("--reviewer", required=True)
            s.add_argument("--reason", required=True)
    s = top.add_parser("finalize")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--reviewer", required=True)
    s.add_argument("--reason", required=True)
    s = top.add_parser("evaluate")
    s.add_argument("--run", required=True)
    s.add_argument("--split", choices=["validation", "test"], default="validation")
    s.add_argument("--data-root", required=True)
    s.add_argument("--artifact", required=True)
    top.add_parser("report")
    s = top.add_parser("remote-work", help="Internal backend worker; not a submit command")
    s.add_argument("--run", required=True)
    s.add_argument("--cache", required=True)
    return p


def dispatch(args):
    root = args.root.resolve()
    if args.command == "doctor":
        cfg = resolve(root / args.config, root)
        cuda = _cuda_status()
        return {"core_ok": True, "research_ready": not readiness(cfg), "issues": readiness(cfg),
                "local_pilot_ready": cuda.get("available", False), "cuda": cuda,
                "optional_packages": {n: importlib.util.find_spec(n) is not None
                                      for n in ("torch", "transformers", "peft", "modal", "kaggle")},
                "credential_environment_present": {n: bool(os.environ.get(n)) for n in
                    ("HF_TOKEN", "KAGGLE_API_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")},
                "note": "Presence only; account/permissions/quota not verified"}
    if args.command == "data":
        if args.action in {"inspect", "sample"}:
            source = read(root / args.registry)["sources"][args.source]
            if not source.get("local_path"):
                raise ValueError("Source local_path missing; fetch to an ignored directory first")
            folder = root / source["local_path"]
            out = root / (args.out or Path("outputs/reviews") / args.source)
            if args.action == "sample":
                return sample_sheet(folder, args.count, args.seed, out)
            result = image_inventory(folder)
            write(out / "inventory.json", result)
            return {"count": result["count"], "label_counts": result["label_counts"],
                    "unreadable": len(result["errors"]), "output": str(out / "inventory.json")}
        if args.action == "audit":
            return audit(records_from(root / args.manifest), read(root / args.registry)["sources"], root)
        return freeze(root / args.manifest, root / args.registry, root / args.out, root, args.version, args.seed)
    if args.command == "run":
        if args.action == "prepare":
            return prepare(root, root / args.config)
        if args.action == "smoke":
            return smoke(root, root / args.config)
        if args.action == "local":
            from .train import execute
            if args.resume_run:
                run_id = args.resume_run
                existing = read(run_folder(root, run_id) / "run.json")
                if existing["stage"] != "pilot" or existing["backend"] != "local":
                    raise ValueError("Only a local pilot run can be resumed with run local")
            else:
                _, created = new_run(root, args.experiment, "pilot", "local")
                run_id = created["run_id"]
            return execute(root, run_id, Path(args.data_root).resolve(), resume=bool(args.resume_run))
        if args.action == "review":
            return review(root, args.run, args.decision, args.reviewer, args.reason)
        from . import remote
        if args.action == "submit":
            return remote.submit(root, args.experiment, args.backend, args.dry_run)
        if args.action == "publish":
            return remote.publish_artifact(root, args.run)
        return getattr(remote, args.action)(root, args.run)
    if args.command == "finalize":
        from .finalize import lock_selection
        return lock_selection(root, args.runs, args.reviewer, args.reason)
    if args.command == "evaluate":
        from .finalize import evaluate_artifact
        return evaluate_artifact(root, args.run, args.data_root, args.artifact, args.split)
    if args.command == "remote-work":
        from .remote import remote_work
        persist = None
        if os.environ.get("FAVI_MODAL_VOLUME"):
            import modal
            volume = modal.Volume.from_name(os.environ["FAVI_MODAL_VOLUME"])
            persist = volume.commit
        return remote_work(root, args.run, args.cache, persist)
    return build_report(root)


def _cuda_status():
    try:
        import torch
        available = torch.cuda.is_available()
        status = {"torch_version": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
                  "available": available}
        if available:
            props = torch.cuda.get_device_properties(0)
            status.update(device=props.name, vram_gb=round(props.total_memory / 1024 ** 3, 2))
        else:
            status["reason"] = "PyTorch cannot access CUDA; sync the train extra and verify the NVIDIA driver"
        return status
    except (ImportError, OSError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def main():
    try:
        result = dispatch(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and result.get("ok") is False:
            return 1
        return 0
    except (ValueError, OSError, KeyError, ImportError, subprocess.SubprocessError, TimeoutError) as exc:
        # Subprocess/provider errors can contain secrets; never print their captured output.
        message = str(exc) if isinstance(exc, (ValueError, KeyError, ImportError)) else type(exc).__name__
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
