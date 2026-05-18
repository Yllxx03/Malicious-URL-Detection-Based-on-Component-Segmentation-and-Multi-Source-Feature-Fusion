from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from baseline.experiment_registry import BASELINE_PRESET_LOOKUP  # noqa: E402
except ModuleNotFoundError:
    from experiment_registry import BASELINE_PRESET_LOOKUP  # noqa: E402
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


def parse_args() -> tuple[argparse.Namespace, str]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--preset", choices=tuple(BASELINE_PRESET_LOOKUP), required=True)
    bootstrap_args, _ = bootstrap.parse_known_args()
    preset = BASELINE_PRESET_LOOKUP[bootstrap_args.preset]

    parser = argparse.ArgumentParser(
        description=preset.description,
        parents=[bootstrap],
    )
    add_data_arguments(parser, default_output_dir=PROJECT_ROOT / "runs" / preset.default_output_dir_name)
    add_runtime_arguments(parser)
    add_shared_model_arguments(
        parser,
        default_cnn_kernel_sizes=preset.cnn_kernel_sizes,
        default_dropout=preset.dropout,
    )
    add_transformer_arguments(parser)
    add_delta_arguments(parser)
    add_optimization_arguments(parser, default_lr=preset.lr, default_head_lr=preset.head_lr)
    parser.set_defaults(
        sequence_encoder=preset.sequence_encoder,
        sequence_pooling=preset.sequence_pooling,
        fusion_mode=preset.fusion_mode,
        use_domain_branch=preset.use_domain_branch,
        use_struct_branch=preset.use_struct_branch,
        char_cnn_layers=preset.char_cnn_layers,
        tcn_layers=preset.tcn_layers,
        tcn_kernel_size=preset.tcn_kernel_size,
        bigru_layers=preset.bigru_layers,
        bigru_hidden_dim=preset.bigru_hidden_dim,
    )
    return finalize_cli_args(parser.parse_args()), preset.display_name


def main() -> None:
    args, display_name = parse_args()
    print(f"Running preset: {display_name}", flush=True)
    run_training(args)


if __name__ == "__main__":
    main()
