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
        description="Ablation: keep the new training setup but replace Delta-Gated with Transformer."
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / "ablation_transformer_encoder")
    add_runtime_arguments(parser)
    add_shared_model_arguments(parser, default_cnn_kernel_sizes="3,5,7", default_dropout=0.1)
    add_transformer_arguments(parser)
    add_optimization_arguments(parser, default_lr=2e-4, default_head_lr=5e-4)
    parser.set_defaults(
        experiment_group="ablation",
        experiment_name="transformer_encoder",
        reference_experiment="main_hybrid_delta_attention",
        experiment_description="只替换序列编码器，用于验证主模型收益是否主要来自 Delta-Gated 编码器。",
        ablation_factors=(
            "仅修改 sequence_encoder：delta_gated -> transformer。",
            "保持 sequence_pooling=mean，避免把池化策略变化混入本组结果。",
            "保持 fusion_mode=gated_sum，继续保留门控融合。",
            "保持 CNN 域名分支和结构特征分支不变。",
        ),
        sequence_encoder="transformer",
        sequence_pooling="mean",
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
