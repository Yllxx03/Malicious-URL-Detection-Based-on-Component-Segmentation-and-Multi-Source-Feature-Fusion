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
    add_delta_arguments,
    add_optimization_arguments,
    add_runtime_arguments,
    add_shared_model_arguments,
    finalize_cli_args,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation: remove the gate and use ungated averaged fusion."
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / "ablation_ungated_sum")
    add_runtime_arguments(parser)
    add_shared_model_arguments(parser, default_cnn_kernel_sizes="3,5,7", default_dropout=0.1)
    add_delta_arguments(parser)
    add_optimization_arguments(parser, default_lr=2e-4, default_head_lr=5e-4)
    parser.set_defaults(
        experiment_group="ablation",
        experiment_name="ungated_sum",
        reference_experiment="main_hybrid_delta_attention",
        experiment_description="移除门控融合，用简单加和替代，用于验证门控融合相对平均融合的贡献。",
        ablation_factors=(
            "仅修改 fusion_mode：gated_sum -> ungated_sum。",
            "保持 sequence_encoder=delta_gated，避免把序列编码器影响混入本组结果。",
            "保持 sequence_pooling=mean。",
            "保持 CNN 域名分支和结构特征分支不变。",
        ),
        sequence_encoder="delta_gated",
        sequence_pooling="attention",
        fusion_mode="ungated_sum",
        delta_short_conv_kernel_size=4,
        transformer_layers=4,
        transformer_heads=8,
        transformer_ffn_dim=512,
    )
    return finalize_cli_args(parser.parse_args())


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
