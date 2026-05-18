from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from url_detect.train_pipeline import (  # noqa: E402
    ROOT_DIR as PROJECT_ROOT,
    add_data_arguments,
    add_delta_arguments,
    add_optimization_arguments,
    add_runtime_arguments,
    add_shared_model_arguments,
    add_transformer_arguments,
    finalize_cli_args,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the optimized CNN + Hybrid Delta/Attention + conditional fusion URL detector."
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / "main_delta_gated")
    add_runtime_arguments(parser)
    add_shared_model_arguments(parser, default_cnn_kernel_sizes="3,5,7", default_dropout=0.1)
    add_transformer_arguments(parser)
    add_delta_arguments(parser)
    add_optimization_arguments(parser, default_lr=2e-4, default_head_lr=5e-4)
    parser.set_defaults(
        sequence_encoder="hybrid_delta_attention",
        sequence_pooling="attention",
        fusion_mode="bilinear_gated",
        delta_layers=4,
        delta_heads=4,
        delta_ffn_dim=512,
        delta_short_conv_kernel_size=4,
        hybrid_attention_layers=1,
        hybrid_rope_fraction=0.5,
        transformer_layers=4,
        transformer_heads=8,
        transformer_ffn_dim=512,
    )
    return finalize_cli_args(parser.parse_args())


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
