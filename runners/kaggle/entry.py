"""Generated launcher embeds only an allowlisted code/config bundle, never secrets/data."""
import base64
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

BUNDLE = ""
RUN_ID = ""


def main():
    root = Path("/kaggle/working/project")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
        for info in archive.infolist():
            path = (root / info.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Invalid bundle path")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(info))
    # HF_TOKEN is configured once in Kaggle Secrets and granted to this kernel.
    if not os.environ.get("HF_TOKEN"):
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.9.9"], check=True)
    subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--extra", "train"], cwd=root, check=True)
    subprocess.run(["uv", "run", "--frozen", "--no-sync", "python", "-m", "favi_hub",
                    "remote-work", "--run", RUN_ID, "--cache", "/kaggle/working/cache"], cwd=root, check=True)
    folder = root / "experiments" / RUN_ID
    with zipfile.ZipFile("/kaggle/working/result-bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("run.json", "metrics.json", "calibration.json", "history.csv"):
            if (folder / name).exists():
                archive.write(folder / name, name)
    # Heavy artifacts stay in kernel output; collect downloads only result-bundle.zip.


if __name__ == "__main__":
    main()
