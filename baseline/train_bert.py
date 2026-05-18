from __future__ import annotations

import argparse
import contextlib
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise RuntimeError("baseline/train_bert.py requires `torch`. Install it with `pip install torch`.") from exc

try:
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "baseline/train_bert.py requires `transformers` and `tokenizers`. "
        "Install them with `pip install transformers tokenizers`."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from url_detect.data import _extract_parts  # noqa: E402
from url_detect.metrics import DEFAULT_FPR_TARGETS, binary_curve_metric_names, compute_binary_curve_metrics  # noqa: E402

MODEL_NAME_ALIASES = {
    "distilbert": "distilbert-base-uncased",
}


class URLTextDataset(Dataset):
    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.examples[index]


class EpochLogger:
    def __init__(self, output_dir: Path, fpr_targets: tuple[float, ...]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = output_dir / f"train_{timestamp}.log"
        self.metrics_path = output_dir / f"epoch_metrics_{timestamp}.csv"
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.metrics_handle = self.metrics_path.open("a", encoding="utf-8", newline="")
        self.metrics_writer = csv.writer(self.metrics_handle)
        self.extra_metric_names = tuple(binary_curve_metric_names(fpr_targets))
        self.metrics_writer.writerow(
            [
                "epoch",
                "split",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fpr",
                *self.extra_metric_names,
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )
        self.metrics_handle.flush()

    def write(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted, flush=True)
        self.log_handle.write(formatted + "\n")
        self.log_handle.flush()

    def write_epoch_metrics(self, epoch: int, split: str, metrics: dict[str, float | int]) -> None:
        row: list[Any] = [
            epoch,
            split,
            f"{float(metrics['loss']):.6f}",
            f"{float(metrics['accuracy']):.6f}",
            f"{float(metrics['precision']):.6f}",
            f"{float(metrics['recall']):.6f}",
            f"{float(metrics['f1']):.6f}",
            f"{float(metrics['fpr']):.6f}",
        ]
        row.extend(f"{float(metrics.get(name, math.nan)):.6f}" for name in self.extra_metric_names)
        row.extend(
            [
                int(metrics["tp"]),
                int(metrics["fp"]),
                int(metrics["tn"]),
                int(metrics["fn"]),
            ]
        )
        self.metrics_writer.writerow(row)
        self.metrics_handle.flush()

    def close(self) -> None:
        self.log_handle.close()
        self.metrics_handle.close()


class BertURLClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str,
        num_classes: int,
        dropout: float,
        freeze_encoder: bool,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        hidden_dim = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden_states[:, 0]
        return self.classifier(pooled)


def parse_fpr_targets(raw_value: str) -> tuple[float, ...]:
    if not raw_value.strip():
        return DEFAULT_FPR_TARGETS
    targets = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    return targets or DEFAULT_FPR_TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DistilBERT + MLP baseline.")
    parser.add_argument("--dataset", type=Path, default=ROOT_DIR / "data" / "train" / "dataset.csv")
    parser.add_argument("--holdout-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs" / "baseline_distilbert_mlp")
    parser.add_argument("--model", choices=tuple(MODEL_NAME_ALIASES), default="distilbert")
    parser.add_argument("--model-name", type=str, default=None, help="Override the Hugging Face model identifier.")
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--freeze", action="store_true", help="Freeze the encoder and train only the MLP head.")
    parser.add_argument("--local-files-only", action="store_true", help="Only load model/tokenizer from the local Hugging Face cache.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--fpr-targets", type=str, default="1e-2,1e-3")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    args.fpr_targets = parse_fpr_targets(args.fpr_targets)
    return args


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return MODEL_NAME_ALIASES[args.model]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def resolve_precision(device: torch.device, precision: str) -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda":
        return None, False
    if precision == "fp32":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision == "fp16":
        return torch.float16, True
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None) -> Any:
    if device.type != "cuda" or amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def read_examples(dataset_path: Path, max_samples: int | None) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if max_samples is not None and row_index >= max_samples:
                break
            parts = _extract_parts(row)
            text = parts.url_raw.strip() or parts.sequence_text
            examples.append((text, int(row["label"])))
    if not examples:
        raise ValueError(f"No rows found in dataset: {dataset_path}")
    return examples


def stratified_split(
    examples: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if val_ratio <= 0.0 or len(examples) <= 1:
        return examples, []

    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(examples):
        label_to_indices[label].append(index)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(label_to_indices):
        indices = np.asarray(label_to_indices[label], dtype=np.int64)
        rng.shuffle(indices)
        if len(indices) <= 1:
            val_count = 0
        else:
            val_count = int(round(len(indices) * val_ratio))
            val_count = max(1, min(val_count, len(indices) - 1))
        val_indices.extend(int(index) for index in indices[:val_count])
        train_indices.extend(int(index) for index in indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_examples = [examples[index] for index in train_indices]
    val_examples = [examples[index] for index in val_indices]
    return train_examples, val_examples


def make_collate_fn(tokenizer, max_length: int):
    def collate(samples: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [sample[0] for sample in samples]
        labels = torch.as_tensor([sample[1] for sample in samples], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return collate


def create_loader(
    examples: list[tuple[str, int]],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader | None:
    if not examples:
        return None
    return DataLoader(
        URLTextDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=make_collate_fn(tokenizer, max_length),
    )


def compute_class_weights(examples: list[tuple[str, int]]) -> torch.Tensor:
    labels = np.asarray([label for _, label in examples], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    total = float(np.sum(counts))
    weights = np.ones(2, dtype=np.float32)
    for class_id, count in zip(classes.tolist(), counts.tolist(), strict=False):
        weights[class_id] = total / (len(classes) * float(count))
    return torch.as_tensor(weights, dtype=torch.float32)


def build_optimizer(model: BertURLClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    parameter_groups: list[dict[str, Any]] = []
    if not args.freeze:
        encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
        if encoder_params:
            parameter_groups.append(
                {
                    "params": encoder_params,
                    "lr": args.encoder_lr,
                    "weight_decay": args.weight_decay,
                }
            )
    head_params = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    parameter_groups.append(
        {
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        }
    )
    return torch.optim.AdamW(parameter_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def compute_step_stats(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int, int, int]:
    predictions = torch.argmax(logits, dim=-1)
    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    correct = int((predictions == labels).sum().item())
    return correct, tp, fp, fn, tn


def collect_scores(
    logits: torch.Tensor,
    labels: torch.Tensor,
    score_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
) -> None:
    if logits.size(-1) != 2:
        return
    scores = torch.softmax(logits.detach(), dim=-1)[:, 1]
    score_chunks.append(scores.float().cpu().numpy())
    label_chunks.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))


def finalize_metrics(
    loss_sum: float,
    sample_count: int,
    correct: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    score_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    fpr_targets: tuple[float, ...],
) -> dict[str, float | int]:
    sample_count = max(1, sample_count)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = correct / sample_count
    fpr = fp / max(1, fp + tn)
    scores = np.concatenate(score_chunks) if score_chunks else np.empty((0,), dtype=np.float32)
    labels = np.concatenate(label_chunks) if label_chunks else np.empty((0,), dtype=np.int64)
    metrics: dict[str, float | int] = {
        "loss": loss_sum / sample_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    metrics.update(compute_binary_curve_metrics(scores, labels, fpr_targets))
    return metrics


def format_split_metrics(split: str, metrics: dict[str, float | int], fpr_targets: tuple[float, ...]) -> str:
    summary = (
        f"{split} loss {float(metrics['loss']):.4f} "
        f"Accuracy {float(metrics['accuracy']):.4f} "
        f"Precision {float(metrics['precision']):.4f} "
        f"Recall {float(metrics['recall']):.4f} "
        f"F1 {float(metrics['f1']):.4f} "
        f"FPR {float(metrics['fpr']):.4f} "
        f"ROC-AUC {float(metrics.get('roc_auc', math.nan)):.4f} "
        f"PR-AUC {float(metrics.get('pr_auc', math.nan)):.4f}"
    )
    for target in fpr_targets:
        metric_name = f"tpr_at_fpr_{target:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        metric_value = float(metrics.get(metric_name, math.nan))
        summary += f" {metric_name.upper()} {metric_value:.4f}"
    return summary


def train_one_epoch(
    model: BertURLClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    args: argparse.Namespace,
    logger: EpochLogger,
    epoch: int,
) -> dict[str, float | int]:
    model.train()
    if model.freeze_encoder:
        model.encoder.eval()

    total_steps = len(loader)
    loss_sum = 0.0
    sample_count = 0
    correct = tp = fp = fn = tn = 0
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        labels = batch.pop("labels")
        with autocast_context(device, amp_dtype):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        batch_size = labels.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        batch_correct, batch_tp, batch_fp, batch_fn, batch_tn = compute_step_stats(logits.detach(), labels)
        correct += batch_correct
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn
        collect_scores(logits, labels, score_chunks, label_chunks)

        if step % args.log_interval == 0:
            logger.write(
                f"Epoch {epoch + 1:02d} | Step {step}/{total_steps} | "
                f"loss={loss_sum / max(1, sample_count):.4f} | "
                f"encoder_lr={optimizer.param_groups[0]['lr']:.6f} | "
                f"head_lr={optimizer.param_groups[-1]['lr']:.6f}"
            )

    return finalize_metrics(loss_sum, sample_count, correct, tp, fp, fn, tn, score_chunks, label_chunks, args.fpr_targets)


def evaluate(
    model: BertURLClassifier,
    loader: DataLoader | None,
    criterion: nn.Module,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    fpr_targets: tuple[float, ...],
) -> dict[str, float | int] | None:
    if loader is None:
        return None

    model.eval()
    loss_sum = 0.0
    sample_count = 0
    correct = tp = fp = fn = tn = 0
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            with autocast_context(device, amp_dtype):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            batch_correct, batch_tp, batch_fp, batch_fn, batch_tn = compute_step_stats(logits, labels)
            correct += batch_correct
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn
            collect_scores(logits, labels, score_chunks, label_chunks)

    return finalize_metrics(loss_sum, sample_count, correct, tp, fp, fn, tn, score_chunks, label_chunks, fpr_targets)


def save_checkpoint(
    path: Path,
    model: BertURLClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    best_f1: float,
    metrics: dict[str, float | int] | None,
    args: argparse.Namespace,
    model_name: str,
) -> None:
    payload = {
        "epoch": epoch,
        "best_f1": best_f1,
        "metrics": metrics,
        "args": vars(args),
        "model_config": {
            "sequence_encoder": "distilbert",
            "sequence_pooling": args.pooling,
            "fusion_mode": "none",
            "model_name": model_name,
            "freeze_encoder": args.freeze,
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device()
    amp_dtype, use_scaler = resolve_precision(device, args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    logger = EpochLogger(args.output_dir, args.fpr_targets)
    model_name = resolve_model_name(args)

    try:
        logger.write(f"Output dir: {args.output_dir.resolve()}")
        logger.write(f"Device: {device} | model={model_name} | pooling={args.pooling} | freeze={args.freeze}")

        train_examples, val_examples = stratified_split(read_examples(args.dataset, args.max_samples), args.val_ratio, args.seed)
        holdout_examples = read_examples(args.holdout_dataset, args.max_samples) if args.holdout_dataset is not None else []
        logger.write(
            f"Loaded examples: train={len(train_examples)} | val={len(val_examples)} | holdout={len(holdout_examples)}"
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only, use_fast=True)
        except Exception as exc:  # pragma: no cover - depends on external cache/network state
            raise RuntimeError(
                f"Failed to load tokenizer `{model_name}`. "
                "If the environment is offline, pre-download the model or rerun with `--local-files-only` after caching it."
            ) from exc

        pin_memory = device.type == "cuda"
        train_loader = create_loader(
            train_examples,
            tokenizer,
            args.max_length,
            args.batch_size,
            shuffle=True,
            workers=args.workers,
            pin_memory=pin_memory,
        )
        val_loader = create_loader(
            val_examples,
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            shuffle=False,
            workers=args.workers,
            pin_memory=pin_memory,
        )
        holdout_loader = create_loader(
            holdout_examples,
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            shuffle=False,
            workers=args.workers,
            pin_memory=pin_memory,
        )
        if train_loader is None:
            raise ValueError("Training split is empty after stratified split.")

        try:
            model = BertURLClassifier(
                model_name=model_name,
                pooling=args.pooling,
                num_classes=2,
                dropout=args.dropout,
                freeze_encoder=args.freeze,
                local_files_only=args.local_files_only,
            )
        except Exception as exc:  # pragma: no cover - depends on external cache/network state
            raise RuntimeError(
                f"Failed to load model `{model_name}`. "
                "If the environment is offline, pre-download the model or rerun with `--local-files-only` after caching it."
            ) from exc
        model.to(device)

        class_weights = compute_class_weights(train_examples).to(device) if args.use_class_weights else None
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, len(train_loader), args)

        best_f1 = 0.0
        for epoch in range(args.epochs):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=scaler,
                amp_dtype=amp_dtype,
                device=device,
                args=args,
                logger=logger,
                epoch=epoch,
            )
            val_metrics = evaluate(model, val_loader, criterion, amp_dtype, device, args.fpr_targets)
            holdout_metrics = evaluate(model, holdout_loader, criterion, amp_dtype, device, args.fpr_targets)

            logger.write_epoch_metrics(epoch + 1, "train", train_metrics)
            summary_parts = [format_split_metrics("train", train_metrics, args.fpr_targets)]
            metrics_to_track = train_metrics
            if val_metrics is not None:
                logger.write_epoch_metrics(epoch + 1, "val", val_metrics)
                summary_parts.append(format_split_metrics("val", val_metrics, args.fpr_targets))
                metrics_to_track = val_metrics
            if holdout_metrics is not None:
                logger.write_epoch_metrics(epoch + 1, "holdout", holdout_metrics)
                summary_parts.append(format_split_metrics("holdout", holdout_metrics, args.fpr_targets))

            logger.write(f"Epoch {epoch + 1:02d} | " + " | ".join(summary_parts))

            current_f1 = float(metrics_to_track["f1"])
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                max(best_f1, current_f1),
                metrics_to_track,
                args,
                model_name,
            )
            if current_f1 >= best_f1:
                best_f1 = current_f1
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_f1,
                    metrics_to_track,
                    args,
                    model_name,
                )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
