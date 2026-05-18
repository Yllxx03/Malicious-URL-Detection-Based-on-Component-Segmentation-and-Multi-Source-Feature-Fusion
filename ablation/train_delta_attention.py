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
        description="Delta improvement: switch the main Delta-Gated model to attention pooling."
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / "ablation_delta_attention")
    add_runtime_arguments(parser)
    add_shared_model_arguments(parser, default_cnn_kernel_sizes="3,5,7", default_dropout=0.1)
    add_delta_arguments(parser)
    add_optimization_arguments(parser, default_lr=2e-4, default_head_lr=5e-4)
    parser.set_defaults(
        experiment_group="optimization",
        experiment_name="delta_attention",
        reference_experiment="main_hybrid_delta_attention",
        experiment_description="将 Delta-Gated 主模型的序列读出从 mean pooling 改为 attention pooling，用于验证主模型是否主要受限于读出方式。",
        ablation_factors=(
            "仅修改 sequence_pooling：mean -> attention。",
            "保持 sequence_encoder=delta_gated。",
            "保持 fusion_mode=gated_sum。",
            "保持 CNN 域名分支和结构特征分支不变。",
        ),
        sequence_encoder="delta_gated",
        sequence_pooling="attention",
        fusion_mode="gated_sum",
        transformer_layers=4,
        transformer_heads=8,
        transformer_ffn_dim=512,
    )
    return finalize_cli_args(parser.parse_args())


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
