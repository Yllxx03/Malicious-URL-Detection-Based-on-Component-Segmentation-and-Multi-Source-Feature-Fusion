from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "runs"


@dataclass(frozen=True)
class BaselinePreset:
    key: str
    display_name: str
    description: str
    default_output_dir_name: str
    sequence_encoder: str
    sequence_pooling: str
    fusion_mode: str
    cnn_kernel_sizes: str
    dropout: float
    lr: float
    head_lr: float
    note: str = ""
    use_domain_branch: bool = True
    use_struct_branch: bool = True
    char_cnn_layers: int = 3
    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    bigru_layers: int = 2
    bigru_hidden_dim: int = 128

    @property
    def default_output_dir(self) -> Path:
        return RUNS_DIR / self.default_output_dir_name

    @property
    def cli_command(self) -> str:
        return f"python baseline/train_preset.py --preset {self.key}"


@dataclass(frozen=True)
class ComparisonExperiment:
    key: str
    display_name: str
    group: str
    command: str
    default_output_dir_name: str
    sequence_encoder: str
    sequence_pooling: str
    fusion_mode: str
    note: str

    @property
    def default_output_dir(self) -> Path:
        return RUNS_DIR / self.default_output_dir_name


BASELINE_PRESETS = (
    BaselinePreset(
        key="legacy_transformer_last",
        display_name="Legacy Transformer + CNN + Gate",
        description="Historical baseline: Transformer sequence encoder with last-token pooling.",
        default_output_dir_name="baseline_transformer",
        sequence_encoder="transformer",
        sequence_pooling="last",
        fusion_mode="gated_sum",
        cnn_kernel_sizes="2,3,5",
        dropout=0.2,
        lr=3e-4,
        head_lr=3e-4,
        note="Historical Transformer baseline with the original pooling, CNN kernels, and learning rates.",
    ),
    BaselinePreset(
        key="transformer_fair_mean",
        display_name="Fair Transformer Control",
        description="Fair-compute baseline: keep the new training recipe and swap the sequence encoder to Transformer.",
        default_output_dir_name="baseline_transformer_fair_mean",
        sequence_encoder="transformer",
        sequence_pooling="mean",
        fusion_mode="gated_sum",
        cnn_kernel_sizes="3,5,7",
        dropout=0.1,
        lr=2e-4,
        head_lr=5e-4,
        note="Fair Transformer control with the same training budget and CNN settings.",
    ),
    BaselinePreset(
        key="transformer_attention_gate",
        display_name="Transformer + Attention Pooling",
        description="Stronger Transformer baseline: keep the new training recipe and switch to attention pooling.",
        default_output_dir_name="baseline_transformer_attention",
        sequence_encoder="transformer",
        sequence_pooling="attention",
        fusion_mode="gated_sum",
        cnn_kernel_sizes="3,5,7",
        dropout=0.1,
        lr=2e-4,
        head_lr=5e-4,
        note="Stronger Transformer control with attention pooling.",
    ),
    BaselinePreset(
        key="charcnn_attention",
        display_name="CharCNN + Attention",
        description="Sequence-only character CNN baseline with attention pooling.",
        default_output_dir_name="baseline_charcnn_attention",
        sequence_encoder="char_cnn",
        sequence_pooling="attention",
        fusion_mode="none",
        cnn_kernel_sizes="3,5,7",
        dropout=0.1,
        lr=2e-4,
        head_lr=5e-4,
        use_domain_branch=False,
        use_struct_branch=False,
        char_cnn_layers=3,
        note="Sequence-only character CNN baseline with attention pooling.",
    ),
    BaselinePreset(
        key="tcn_mlp",
        display_name="TCN + MLP",
        description="Sequence-only temporal convolution baseline with mean pooling.",
        default_output_dir_name="baseline_tcn_mlp",
        sequence_encoder="tcn",
        sequence_pooling="mean",
        fusion_mode="none",
        cnn_kernel_sizes="3,5,7",
        dropout=0.1,
        lr=2e-4,
        head_lr=5e-4,
        use_domain_branch=False,
        use_struct_branch=False,
        tcn_layers=4,
        tcn_kernel_size=3,
        note="Sequence-only temporal convolution baseline with mean pooling.",
    ),
    BaselinePreset(
        key="bigru_attention",
        display_name="BiGRU + Attention",
        description="Sequence-only bidirectional GRU baseline with attention pooling.",
        default_output_dir_name="baseline_bigru_attention",
        sequence_encoder="bigru",
        sequence_pooling="attention",
        fusion_mode="none",
        cnn_kernel_sizes="3,5,7",
        dropout=0.1,
        lr=2e-4,
        head_lr=5e-4,
        use_domain_branch=False,
        use_struct_branch=False,
        bigru_layers=2,
        bigru_hidden_dim=128,
        note="Sequence-only bidirectional GRU baseline with attention pooling.",
    ),
)

BASELINE_PRESET_LOOKUP = {preset.key: preset for preset in BASELINE_PRESETS}

COMPARISON_EXPERIMENTS = (
    ComparisonExperiment(
        key="main_hybrid_delta_attention",
        display_name="Main Hybrid Delta-Attention",
        group="core",
        command="python train.py",
        default_output_dir_name="main_delta_gated",
        sequence_encoder="hybrid_delta_attention",
        sequence_pooling="attention",
        fusion_mode="bilinear_gated",
        note="Optimized main model: enhanced Delta blocks, one RoPE softmax attention layer, attention pooling, and conditional fusion.",
    ),
    ComparisonExperiment(
        key="legacy_transformer_last",
        display_name="Legacy Transformer + CNN + Gate",
        group="core",
        command="python baseline/train.py",
        default_output_dir_name="baseline_transformer",
        sequence_encoder="transformer",
        sequence_pooling="last",
        fusion_mode="gated_sum",
        note="Historical Transformer baseline.",
    ),
    ComparisonExperiment(
        key="transformer_fair_mean",
        display_name="Fair Transformer Control",
        group="core",
        command="python baseline/train_preset.py --preset transformer_fair_mean",
        default_output_dir_name="baseline_transformer_fair_mean",
        sequence_encoder="transformer",
        sequence_pooling="mean",
        fusion_mode="gated_sum",
        note="Fair Transformer control with the shared training recipe.",
    ),
    ComparisonExperiment(
        key="transformer_attention_gate",
        display_name="Transformer + Attention Pooling",
        group="core",
        command="python baseline/train_preset.py --preset transformer_attention_gate",
        default_output_dir_name="baseline_transformer_attention",
        sequence_encoder="transformer",
        sequence_pooling="attention",
        fusion_mode="gated_sum",
        note="Stronger Transformer baseline.",
    ),
    ComparisonExperiment(
        key="distilbert_mlp",
        display_name="DistilBERT + MLP",
        group="core",
        command="python baseline/train_bert.py --model distilbert --freeze",
        default_output_dir_name="baseline_distilbert_mlp",
        sequence_encoder="distilbert",
        sequence_pooling="cls",
        fusion_mode="none",
        note="Lightweight pretrained text baseline; requires Hugging Face DistilBERT weights and tokenizer.",
    ),
    ComparisonExperiment(
        key="charcnn_attention",
        display_name="CharCNN + Attention",
        group="core",
        command="python baseline/train_preset.py --preset charcnn_attention",
        default_output_dir_name="baseline_charcnn_attention",
        sequence_encoder="char_cnn",
        sequence_pooling="attention",
        fusion_mode="none",
        note="Character CNN sequence baseline.",
    ),
    ComparisonExperiment(
        key="tcn_mlp",
        display_name="TCN + MLP",
        group="core",
        command="python baseline/train_preset.py --preset tcn_mlp",
        default_output_dir_name="baseline_tcn_mlp",
        sequence_encoder="tcn",
        sequence_pooling="mean",
        fusion_mode="none",
        note="Temporal convolution sequence baseline.",
    ),
    ComparisonExperiment(
        key="bigru_attention",
        display_name="BiGRU + Attention",
        group="core",
        command="python baseline/train_preset.py --preset bigru_attention",
        default_output_dir_name="baseline_bigru_attention",
        sequence_encoder="bigru",
        sequence_pooling="attention",
        fusion_mode="none",
        note="Bidirectional GRU sequence baseline.",
    ),
    ComparisonExperiment(
        key="ablation_no_hybrid_attention",
        display_name="No Hybrid Attention",
        group="ablation",
        command="python ablation/train_no_hybrid_attention.py",
        default_output_dir_name="ablation_no_hybrid_attention",
        sequence_encoder="delta_gated",
        sequence_pooling="attention",
        fusion_mode="bilinear_gated",
        note="Removes the final RoPE softmax attention layer and keeps enhanced Delta blocks.",
    ),
    ComparisonExperiment(
        key="ablation_no_short_conv",
        display_name="No Short Convolution",
        group="ablation",
        command="python ablation/train_no_short_conv.py",
        default_output_dir_name="ablation_no_short_conv",
        sequence_encoder="hybrid_delta_attention",
        sequence_pooling="attention",
        fusion_mode="bilinear_gated",
        note="Disables byte-level short convolution before Delta updates.",
    ),
    ComparisonExperiment(
        key="ablation_mean_pooling",
        display_name="Mean Pooling",
        group="ablation",
        command="python ablation/train_mean_pooling.py",
        default_output_dir_name="ablation_mean_pooling",
        sequence_encoder="hybrid_delta_attention",
        sequence_pooling="mean",
        fusion_mode="bilinear_gated",
        note="Replaces attention pooling with mean pooling.",
    ),
    ComparisonExperiment(
        key="ablation_gated_sum_fusion",
        display_name="Scalar Gated Fusion",
        group="ablation",
        command="python ablation/train_gated_sum_fusion.py",
        default_output_dir_name="ablation_gated_sum_fusion",
        sequence_encoder="hybrid_delta_attention",
        sequence_pooling="attention",
        fusion_mode="gated_sum",
        note="Replaces bilinear conditional fusion with scalar branch gating.",
    ),
    ComparisonExperiment(
        key="ablation_hybrid_transformer_encoder",
        display_name="Transformer Encoder + Optimized Head",
        group="ablation",
        command="python ablation/train_hybrid_transformer_encoder.py",
        default_output_dir_name="ablation_hybrid_transformer_encoder",
        sequence_encoder="transformer",
        sequence_pooling="attention",
        fusion_mode="bilinear_gated",
        note="Replaces the hybrid Delta-Attention encoder with Transformer while keeping the optimized head.",
    ),
    ComparisonExperiment(
        key="ablation_ungated_concat",
        display_name="Enhanced Delta + Ungated Concat",
        group="ablation",
        command="python ablation/train_ungated_concat.py",
        default_output_dir_name="ablation_ungated_concat",
        sequence_encoder="delta_gated",
        sequence_pooling="attention",
        fusion_mode="ungated_concat",
        note="Removes learned fusion gates and concatenates branch features.",
    ),
    ComparisonExperiment(
        key="ablation_ungated_sum",
        display_name="Enhanced Delta + Ungated Sum",
        group="ablation",
        command="python ablation/train_ungated_sum.py",
        default_output_dir_name="ablation_ungated_sum",
        sequence_encoder="delta_gated",
        sequence_pooling="attention",
        fusion_mode="ungated_sum",
        note="Replaces learned fusion with a simple averaged sum.",
    ),
)

COMPARISON_EXPERIMENT_LOOKUP = {experiment.key: experiment for experiment in COMPARISON_EXPERIMENTS}
