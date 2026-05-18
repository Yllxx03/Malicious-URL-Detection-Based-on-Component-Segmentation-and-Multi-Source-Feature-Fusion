from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from url_detect.train_pipeline import (  # noqa: E402
    ROOT_DIR as PROJECT_ROOT,
    add_data_arguments,
    add_optimization_arguments,
    add_runtime_arguments,
    add_shared_model_arguments,
    add_transformer_arguments,
    finalize_cli_args,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Transformer + CNN + gated fusion baseline."
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / "baseline_transformer")
    add_runtime_arguments(parser)
    add_shared_model_arguments(parser, default_cnn_kernel_sizes="2,3,5", default_dropout=0.2)
    add_transformer_arguments(parser)
    add_optimization_arguments(parser, default_lr=3e-4, default_head_lr=3e-4)
    parser.set_defaults(
        sequence_encoder="transformer",
        sequence_pooling="last",
        fusion_mode="gated_sum",
        delta_layers=4,
        delta_heads=4,
        delta_ffn_dim=512,
    )
    return finalize_cli_args(parser.parse_args())


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
