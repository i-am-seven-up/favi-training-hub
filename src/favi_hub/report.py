import csv
import io
from pathlib import Path

from .io import read

FIELDS = ["experiment_id", "run_id", "metric_file", "method", "stage", "backend", "status",
          "config_hash", "dataset_fingerprint", "eval_split", "macro_f1", "trainable_params",
          "peak_vram_gb", "train_seconds", "artifact_uri", "review_decision", "mock"]


def build_report(root):
    root = Path(root)
    rows, comparable = [], []
    for path in sorted((root / "experiments").glob("EXP-*/*/run.json")):
        run = read(path)
        approval = read(path.parent / "review.json") if (path.parent / "review.json").exists() else {}
        metric_paths = [p for p in (path.parent / "metrics.json", path.parent / "metrics.test.json") if p.exists()]
        if not metric_paths:
            metric_paths = [None]
        for metric_path in metric_paths:
            metrics = read(metric_path) if metric_path else {}
            row = {key: run.get(key) for key in FIELDS}
            row.update({key: metrics.get(key) for key in
                        ("macro_f1", "trainable_params", "peak_vram_gb", "train_seconds")})
            row.update(metric_file=metric_path.name if metric_path else None,
                       eval_split=metrics.get("split"), review_decision=approval.get("decision"))
            rows.append(row)
            if (run["status"] == "completed" and run["stage"] == "full"
                    and not run.get("mock") and metrics):
                comparable.append((run, metrics))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    (root / "experiments").mkdir(exist_ok=True)
    (root / "experiments/index.csv").write_text(buffer.getvalue(), encoding="utf-8")
    lines = ["# FAVI comparison", "", "Generated from run records; fixture/pilot results are not thesis evidence.",
             "", "| Run | Method | Stage | Split | Macro-F1 |", "|---|---|---|---|---|"]
    for r in rows:
        if not r.get("mock"):
            score = r["macro_f1"] if r["macro_f1"] is not None else "not measured"
            lines.append(f"| {r['run_id']} | {r['method']} | {r['stage']} | {r['eval_split']} | {score} |")
    if not rows:
        lines.extend(["", "No experiments executed."])
    for baseline, bm in comparable:
        if baseline["method"] != "frozen":
            continue
        for lora, lm in comparable:
            if (lora["method"] == "lora" and baseline["comparison_key"] == lora["comparison_key"]
                    and baseline["code_hash"] == lora["code_hash"]
                    and bm.get("split") == lm.get("split")
                    and bm.get("split_fingerprint") == lm.get("split_fingerprint")
                    and bm.get("coverage_complete") and lm.get("coverage_complete")):
                delta = lm["macro_f1"] - bm["macro_f1"]
                lines.extend(["", (f"{bm['split']} Δ Macro-F1 = {delta:+.6f}: "
                                   f"{lora['run_id']} − {baseline['run_id']}"),
                              "Hardware/time may differ; no statistical significance claim from one seed."])
    (root / "results").mkdir(exist_ok=True)
    (root / "results/comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"runs": len(rows), "index": "experiments/index.csv", "comparison": "results/comparison.md"}
