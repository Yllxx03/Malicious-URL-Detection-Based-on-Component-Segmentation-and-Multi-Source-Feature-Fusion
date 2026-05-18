from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment_registry import COMPARISON_EXPERIMENTS, ROOT_DIR, RUNS_DIR


@dataclass
class ReportRow:
    experiment_key: str
    display_name: str
    group: str
    command: str
    run_dir: Path
    status: str
    tracked_split: str
    best_epoch: str
    sequence_encoder: str
    sequence_pooling: str
    fusion_mode: str
    note: str
    tracked_metrics: dict[str, str]
    holdout_metrics: dict[str, str]


@dataclass
class ImportedRunSummary:
    status: str
    tracked_split: str
    best_epoch: str
    tracked_metrics: dict[str, str]
    holdout_metrics: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown baseline comparison report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "baseline" / "BASELINE_REPORT.md",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_DIR,
        help="Root directory that contains the experiment run folders.",
    )
    parser.add_argument(
        "--run-override",
        action="append",
        default=[],
        help="Override a run directory with KEY=PATH, e.g. main_delta_gated=runs/main_smoke.",
    )
    parser.add_argument(
        "--log-import-dir",
        type=Path,
        default=None,
        help="Optional directory containing exported console logs such as baseline数据/*.txt.",
    )
    return parser.parse_args()


def parse_run_overrides(raw_items: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --run-override value: {item}")
        key, raw_path = item.split("=", 1)
        overrides[key.strip()] = Path(raw_path.strip())
    return overrides


def resolve_run_dir(default_dir: Path, overrides: dict[str, Path], key: str) -> Path:
    if key not in overrides:
        return default_dir
    override_path = overrides[key]
    if not override_path.is_absolute():
        return (ROOT_DIR / override_path).resolve()
    return override_path


def read_latest_metrics_csv(run_dir: Path) -> list[dict[str, str]]:
    csv_files = sorted(run_dir.glob("epoch_metrics_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not csv_files:
        return []
    with csv_files[0].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_torch_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    candidate_paths = [run_dir / candidate_name for candidate_name in ("best.pt", "last.pt")]
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    for candidate_name in ("best.pt", "last.pt"):
        checkpoint_path = run_dir / candidate_name
        if checkpoint_path.exists():
            return load_torch_checkpoint(checkpoint_path)
    return None


def detect_log_import_dir() -> Path | None:
    for candidate in sorted(ROOT_DIR.glob("baseline*")):
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def find_epoch_row(rows: list[dict[str, str]], epoch: int, split: str) -> dict[str, str] | None:
    epoch_text = str(epoch)
    for row in rows:
        if row.get("epoch") == epoch_text and row.get("split") == split:
            return row
    return None


def infer_tracked_split(rows: list[dict[str, str]]) -> str:
    splits = {row.get("split", "") for row in rows}
    if "val" in splits:
        return "val"
    if "train" in splits:
        return "train"
    return "unknown"


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def extract_metric_bundle(source: dict[str, Any] | None) -> dict[str, str]:
    if not source:
        return {
            "f1": "-",
            "roc_auc": "-",
            "pr_auc": "-",
            "tpr_at_fpr_1e-2": "-",
            "tpr_at_fpr_1e-3": "-",
        }
    return {
        "f1": format_metric(source.get("f1")),
        "roc_auc": format_metric(source.get("roc_auc")),
        "pr_auc": format_metric(source.get("pr_auc")),
        "tpr_at_fpr_1e-2": format_metric(source.get("tpr_at_fpr_1e-2")),
        "tpr_at_fpr_1e-3": format_metric(source.get("tpr_at_fpr_1e-3")),
    }


def build_metric_bundle_from_log_row(row: dict[str, Any], prefix: str) -> dict[str, str]:
    return {
        "f1": format_metric(row.get(f"{prefix}_f1")),
        "roc_auc": format_metric(row.get(f"{prefix}_roc")),
        "pr_auc": format_metric(row.get(f"{prefix}_pr")),
        "tpr_at_fpr_1e-2": format_metric(row.get(f"{prefix}_tpr2")),
        "tpr_at_fpr_1e-3": format_metric(row.get(f"{prefix}_tpr3")),
    }


def load_log_imports(log_dir: Path | None) -> dict[str, ImportedRunSummary]:
    if log_dir is None or not log_dir.exists():
        return {}

    start_pattern = re.compile(r"^\[(?P<timestamp>[^\]]+)\] START (?P<key>\S+)")
    epoch_pattern = re.compile(
        r"^Epoch (?P<epoch>\d+) \| "
        r"train loss (?P<train_loss>[0-9.]+) Accuracy (?P<train_acc>[0-9.]+) "
        r"Precision (?P<train_precision>[0-9.]+) Recall (?P<train_recall>[0-9.]+) "
        r"F1 (?P<train_f1>[0-9.]+) FPR (?P<train_fpr>[0-9.]+) ROC-AUC (?P<train_roc>[0-9.]+) "
        r"PR-AUC (?P<train_pr>[0-9.]+) TPR_AT_FPR_1E-2 (?P<train_tpr2>[0-9.]+) "
        r"TPR_AT_FPR_1E-3 (?P<train_tpr3>[0-9.]+)"
        r"(?: \| val loss (?P<val_loss>[0-9.]+) Accuracy (?P<val_acc>[0-9.]+) "
        r"Precision (?P<val_precision>[0-9.]+) Recall (?P<val_recall>[0-9.]+) "
        r"F1 (?P<val_f1>[0-9.]+) FPR (?P<val_fpr>[0-9.]+) ROC-AUC (?P<val_roc>[0-9.]+) "
        r"PR-AUC (?P<val_pr>[0-9.]+) TPR_AT_FPR_1E-2 (?P<val_tpr2>[0-9.]+) "
        r"TPR_AT_FPR_1E-3 (?P<val_tpr3>[0-9.]+).*)?$"
    )

    raw_results: dict[str, list[dict[str, Any]]] = {}
    current_key: str | None = None
    for txt_path in sorted(log_dir.glob("*.txt")):
        for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            match = start_pattern.match(line)
            if match:
                current_key = match.group("key")
                raw_results.setdefault(current_key, [])
                continue

            match = epoch_pattern.match(line)
            if match is None or current_key is None:
                continue

            payload = match.groupdict()
            parsed_row: dict[str, Any] = {"epoch": int(payload["epoch"])}
            for key, value in payload.items():
                if key == "epoch" or value is None:
                    continue
                parsed_row[key] = float(value)
            raw_results[current_key].append(parsed_row)

    imported: dict[str, ImportedRunSummary] = {}
    for key, rows in raw_results.items():
        if not rows:
            continue
        val_rows = [row for row in rows if row.get("val_f1") is not None]
        if val_rows:
            best_row = max(val_rows, key=lambda row: (float(row["val_f1"]), int(row["epoch"])))
            tracked_split = "val"
            tracked_metrics = build_metric_bundle_from_log_row(best_row, "val")
        else:
            best_row = max(rows, key=lambda row: (float(row["train_f1"]), int(row["epoch"])))
            tracked_split = "train"
            tracked_metrics = build_metric_bundle_from_log_row(best_row, "train")

        imported[key] = ImportedRunSummary(
            status="imported",
            tracked_split=tracked_split,
            best_epoch=str(best_row["epoch"]),
            tracked_metrics=tracked_metrics,
            holdout_metrics=extract_metric_bundle(None),
        )

    return imported


def build_report_row(experiment, run_dir: Path, imported_runs: dict[str, ImportedRunSummary]) -> ReportRow:
    rows = read_latest_metrics_csv(run_dir) if run_dir.exists() else []
    checkpoint = load_checkpoint(run_dir) if run_dir.exists() else None
    imported_run = imported_runs.get(experiment.key)
    tracked_split = infer_tracked_split(rows)
    best_epoch = "-"
    sequence_encoder = experiment.sequence_encoder
    sequence_pooling = experiment.sequence_pooling
    fusion_mode = experiment.fusion_mode
    tracked_metrics = extract_metric_bundle(None)
    holdout_metrics = extract_metric_bundle(None)
    status = "missing"

    if checkpoint is not None:
        status = "ready"
        model_config = checkpoint.get("model_config", {})
        sequence_encoder = str(model_config.get("sequence_encoder", sequence_encoder))
        sequence_pooling = str(model_config.get("sequence_pooling", sequence_pooling))
        fusion_mode = str(model_config.get("fusion_mode", fusion_mode))
        best_epoch_value = int(checkpoint.get("epoch", -1)) + 1
        best_epoch = str(best_epoch_value) if best_epoch_value > 0 else "-"
        tracked_metrics = extract_metric_bundle(checkpoint.get("metrics"))

        if rows:
            if tracked_split == "unknown":
                tracked_split = "val"
            holdout_row = find_epoch_row(rows, best_epoch_value, "holdout")
            holdout_metrics = extract_metric_bundle(holdout_row)
    elif run_dir.exists():
        status = "partial"
        if rows:
            tracked_split = infer_tracked_split(rows)
            candidate_rows = [row for row in rows if row.get("split") == tracked_split]
            if candidate_rows:
                best_row = max(candidate_rows, key=lambda row: float(row.get("f1", "nan")))
                best_epoch = best_row.get("epoch", "-")
                tracked_metrics = extract_metric_bundle(best_row)
                try:
                    holdout_row = find_epoch_row(rows, int(best_epoch), "holdout")
                except ValueError:
                    holdout_row = None
                holdout_metrics = extract_metric_bundle(holdout_row)
    elif imported_run is not None:
        status = imported_run.status
        tracked_split = imported_run.tracked_split
        best_epoch = imported_run.best_epoch
        tracked_metrics = imported_run.tracked_metrics
        holdout_metrics = imported_run.holdout_metrics

    return ReportRow(
        experiment_key=experiment.key,
        display_name=experiment.display_name,
        group=experiment.group,
        command=experiment.command,
        run_dir=run_dir,
        status=status,
        tracked_split=tracked_split,
        best_epoch=best_epoch,
        sequence_encoder=sequence_encoder,
        sequence_pooling=sequence_pooling,
        fusion_mode=fusion_mode,
        note=experiment.note,
        tracked_metrics=tracked_metrics,
        holdout_metrics=holdout_metrics,
    )


def format_run_dir(run_dir: Path) -> str:
    try:
        return str(run_dir.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(run_dir)
    except FileNotFoundError:
        return str(run_dir)


def render_matrix_table(rows: list[ReportRow]) -> list[str]:
    lines = [
        "## Experiment Matrix",
        "",
        "| Key | Model | Group | Sequence Encoder | Pooling | Fusion | Command | Run Dir | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.experiment_key,
                    row.display_name,
                    row.group,
                    row.sequence_encoder,
                    row.sequence_pooling,
                    row.fusion_mode,
                    f"`{row.command}`",
                    f"`{format_run_dir(row.run_dir)}`",
                    row.note,
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_result_table(rows: list[ReportRow], group: str) -> list[str]:
    title = "Core Baseline Results" if group == "core" else "Ablation Results"
    lines = [
        f"## {title}",
        "",
        "| Model | Status | Split | Best Epoch | F1 | ROC-AUC | PR-AUC | TPR@1e-2 | TPR@1e-3 | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.display_name,
                    row.status,
                    row.tracked_split,
                    row.best_epoch,
                    row.tracked_metrics["f1"],
                    row.tracked_metrics["roc_auc"],
                    row.tracked_metrics["pr_auc"],
                    row.tracked_metrics["tpr_at_fpr_1e-2"],
                    row.tracked_metrics["tpr_at_fpr_1e-3"],
                    row.note,
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_holdout_table(rows: list[ReportRow], group: str) -> list[str]:
    if not any(row.holdout_metrics["f1"] != "-" for row in rows):
        return []
    title = "Core Baseline Holdout Results" if group == "core" else "Ablation Holdout Results"
    lines = [
        f"## {title}",
        "",
        "| Model | Holdout F1 | Holdout ROC-AUC | Holdout PR-AUC | Holdout TPR@1e-2 | Holdout TPR@1e-3 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.display_name,
                    row.holdout_metrics["f1"],
                    row.holdout_metrics["roc_auc"],
                    row.holdout_metrics["pr_auc"],
                    row.holdout_metrics["tpr_at_fpr_1e-2"],
                    row.holdout_metrics["tpr_at_fpr_1e-3"],
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_report(rows: list[ReportRow], log_import_dir: Path | None = None) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Baseline Comparison Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "Status is `missing` when a run directory has no checkpoint, and `partial` when metrics exist but no checkpoint was found.",
        "",
    ]
    if log_import_dir is not None:
        lines.extend(
            [
                f"Imported console logs from `{log_import_dir}`; those rows are marked as `imported`.",
                "",
            ]
        )
    lines.extend(render_matrix_table(rows))

    for group in ("core", "ablation"):
        group_rows = [row for row in rows if row.group == group]
        if not group_rows:
            continue
        lines.extend(render_result_table(group_rows, group))
        lines.extend(render_holdout_table(group_rows, group))

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    overrides = parse_run_overrides(args.run_override)
    log_import_dir = args.log_import_dir or detect_log_import_dir()
    imported_runs = load_log_imports(log_import_dir)
    rows = []
    for experiment in COMPARISON_EXPERIMENTS:
        default_run_dir = args.runs_root / experiment.default_output_dir_name
        run_dir = resolve_run_dir(default_run_dir, overrides, experiment.key)
        rows.append(build_report_row(experiment, run_dir, imported_runs))

    report = render_report(rows, log_import_dir=log_import_dir if imported_runs else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
