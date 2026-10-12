import io
import zipfile
from types import SimpleNamespace

import pytest

from favi_hub.__main__ import dispatch
from favi_hub.io import read, write
from favi_hub.remote import bundle, submit, unpack
from favi_hub.report import build_report
from favi_hub.runs import collect_result, new_run, prepare


def test_dry_run_has_no_network_or_run_side_effect(research, monkeypatch):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: pytest.fail("network operation"))
    result = submit(root, experiment["experiment_id"], "modal", dry_run=True)
    assert not result["ready"]  # Missing approved pilot is reported rather than submitted.
    assert result["network_calls"] == 0
    assert not list((root / "experiments").glob("*/*/run.json"))


def test_bundle_allowlist_excludes_secret_and_image(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    (root / ".env").write_text("SECRET=never-package", encoding="utf-8")
    (root / "private.jpg").write_bytes(b"private")
    archive = bundle(root, experiment["experiment_id"], root / "bundle.zip")
    with zipfile.ZipFile(archive) as stream:
        assert ".env" not in stream.namelist()
        assert "private.jpg" not in stream.namelist()
        assert "data.json" in stream.namelist()


def test_zip_traversal_rejected(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        unpack(payload.getvalue(), tmp_path / "safe")


def test_failed_collection_is_idempotent_and_not_a_result(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    folder, run = new_run(root, experiment["experiment_id"], "pilot", "kaggle")
    incoming = root / "incoming"
    write(incoming / "run.json", {**run, "status": "failed", "error_type": "RuntimeError"})
    collect_result(root, run["run_id"], incoming)
    collect_result(root, run["run_id"], incoming)
    assert read(folder / "run.json")["status"] == "failed"
    assert build_report(root)["runs"] == 1
    assert "not measured" in (root / "results/comparison.md").read_text(encoding="utf-8")


def test_collection_identity_mismatch_rejected(research):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")
    _, run = new_run(root, experiment["experiment_id"], "pilot", "kaggle")
    incoming = root / "incoming"
    write(incoming / "run.json", {**run, "status": "completed", "config_hash": "different"})
    with pytest.raises(ValueError, match="mismatch"):
        collect_result(root, run["run_id"], incoming)


def test_local_command_creates_local_pilot(research, monkeypatch):
    root, *_ = research
    experiment = prepare(root, root / "config.yaml")

    def fake_execute(received_root, run_id, data_root, resume=False):
        run = read(received_root / "experiments" / run_id / "run.json")
        return {"run_id": run_id, "backend": run["backend"], "resume": resume,
                "data_root": str(data_root)}

    monkeypatch.setattr("favi_hub.train.execute", fake_execute)
    args = SimpleNamespace(root=root, command="run", action="local",
                           experiment=experiment["experiment_id"], resume_run=None,
                           data_root=str(root / "images"))
    result = dispatch(args)
    assert result["backend"] == "local"
    assert result["resume"] is False
