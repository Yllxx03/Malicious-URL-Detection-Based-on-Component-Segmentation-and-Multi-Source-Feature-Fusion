# Malicious URL Detection Based on Component Segmentation and Multi-Source Feature Fusion

本项目是面向恶意 URL 检测的研究型系统，围绕组件分割与多源特征融合方法开展二分类识别研究，基于 PyTorch 构建模型训练与评测流程，通过对 URL 的域名、路径、查询参数等组件进行结构化拆分，并融合字符级序列特征、域名分支特征和统计结构特征，实现对可疑链接的高精度识别。系统支持主模型训练、基线对比、消融实验、分布外（OOD）鲁棒性评测和推理时延测试，可用于网络钓鱼识别、恶意链接拦截、邮件与短信安全过滤、浏览器安全防护等场景。

## 项目目录结构

```text

├── src/                      # 核心源码目录
│   └── url_detect/           # 数据处理、模型定义、指标计算与训练管线
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── train_pipeline.py
├── data/                     # 项目数据目录
│   └── train/                # 主训练数据
│       └── dataset.csv
├── baseline/                 # 基线实验目录
│   ├── __init__.py
│   ├── experiment_registry.py
│   ├── generate_report.py
│   ├── train.py
│   ├── train_bert.py
│   └── train_preset.py
├── ablation/                 # 消融实验目录
│   ├── __init__.py
│   ├── train_delta_attention.py
│   ├── train_delta_attention_deep.py
│   ├── train_delta_attention_longseq.py
│   ├── train_gated_sum_fusion.py
│   ├── train_hybrid_transformer_encoder.py
│   ├── train_mean_pooling.py
│   ├── train_no_hybrid_attention.py
│   ├── train_no_short_conv.py
│   ├── train_transformer_encoder.py
│   ├── train_ungated_concat.py
│   └── train_ungated_sum.py
├── OOD/                      # 分布外数据准备、鲁棒性评测与适配实验目录
│   ├── __init__.py
│   ├── evaluate_ood.py
│   ├── finetune_main_adapt.py
│   ├── generate_ood_data.py
│   ├── mine_hard_negatives.py
│   ├── prepare_adaptation_data.py
│   ├── prepare_checkpoints.py
│   ├── prepare_real_ood_data.py
│   ├── run_real_adaptation.sh
│   └── tune_main_checkpoint.py
├── benchmark_latency.py
├── train.py
└── README.md
```

## 项目简介

主模型围绕 URL 的多视角表示构建，统一使用三路输入：

1. 域名分支：字符级 byte 编码 + 多尺度 `CNN`
2. 序列分支：`hybrid_delta_attention`、`transformer`、`char_cnn`、`tcn` 或 `bigru`
3. 结构分支：14 维统计特征 + `MLP`

默认主模型配置为：

- `sequence_encoder=hybrid_delta_attention`
- `sequence_pooling=attention`
- `fusion_mode=bilinear_gated`
- `delta_short_conv_kernel_size=4`
- `hybrid_attention_layers=1`
- `hybrid_rope_fraction=0.5`
- `cnn_kernel_sizes=3,5,7`
- `dropout=0.1`

训练和评估阶段重点关注以下指标：

- `accuracy`
- `precision`
- `recall`
- `f1`
- `fpr`
- `roc_auc`
- `pr_auc`
- `tpr_at_fpr_1e-2`
- `tpr_at_fpr_1e-3`

## 环境依赖

- Python 3.10+
- `numpy`
- `torch`
- `transformers`
- `tokenizers`

示例安装：

```bash
pip install numpy torch transformers tokenizers
```

## 快速开始

### 1. 准备数据

默认训练集路径为 `data/train/dataset.csv`。最少包含两列：

```csv
url_raw,label
http://example.com,0
http://secure-login-verify.example.xyz,1
```

### 2. 训练主模型

```bash
python train.py --output-dir runs/main_hybrid_delta_attention
```

常用示例：

```bash
python train.py --max-samples 20000 --epochs 1 --workers 0
python train.py --holdout-dataset data/external/holdout.csv --output-dir runs/main_holdout
```

### 3. 运行 Baseline 对比

```bash
python baseline/train.py
python baseline/train_preset.py --preset transformer_attention_gate
python baseline/generate_report.py --output baseline/BASELINE_REPORT.md
```

详细说明见 [baseline/README.md](baseline/README.md)。

### 4. 运行消融实验

```bash
python ablation/train_no_short_conv.py
python ablation/train_mean_pooling.py
python ablation/train_gated_sum_fusion.py
```

详细说明见 [ablation/README.md](ablation/README.md)。

### 5. 运行 OOD 评测

```bash
python OOD/prepare_checkpoints.py --overwrite
python OOD/prepare_real_ood_data.py
python OOD/evaluate_ood.py --device cuda
```

详细说明见 [OOD/README.md](OOD/README.md)。

## 实验模块说明

### 主模型

根目录 [train.py](train.py) 是主模型训练入口，底层训练逻辑位于 [src/url_detect/train_pipeline.py](src/url_detect/train_pipeline.py)。该训练管线支持：

- 单卡、`DataParallel` 和 `DDP`
- 自动构建缓存数据集
- 按指标自动选择 `best.pt`
- 保存 `train_*.log`、`epoch_metrics_*.csv`、`best.pt`、`last.pt`

### Baseline 对比

`baseline/` 用于和主模型做公平对比，当前包含：

- Transformer 系列对照
- DistilBERT 轻量预训练基线
- CharCNN、TCN、BiGRU 等序列模型基线

可直接查看：

- [baseline/README.md](baseline/README.md)
- [baseline/BASELINE_REPORT.md](baseline/BASELINE_REPORT.md)

### 消融实验

`ablation/` 用于解释主模型提升来自哪些组件，重点覆盖：

- 去除 Hybrid Attention
- 去除短卷积
- 用 Mean Pooling 替代 Attention Pooling
- 用简单门控或无门控替代条件融合

可直接查看：

- [ablation/README.md](ablation/README.md)
- [ablation/消融实验结果分析.md](ablation/消融实验结果分析.md)

### OOD 鲁棒性实验

`OOD/` 用于评估模型在真实分布外样本上的泛化能力，并提供适配与 hard negative mining 流程。当前仓库已经附带：

- 真实 OOD 数据划分说明：[OOD/data/REAL_OOD_DATA_SUMMARY.md](OOD/data/REAL_OOD_DATA_SUMMARY.md)
- 真实 OOD 结果报告：[OOD/results/REAL_OOD_RESULTS.md](OOD/results/REAL_OOD_RESULTS.md)

当前 OOD 报告中：

- `Strong baseline` 在 `real_ood_test` 上的默认 `F1` 为 `0.742857`
- `Main` 在 `TPR@FPR=1e-3` 上保持当前报告内最佳表现

## 推理时延测试

```bash
python benchmark_latency.py --checkpoint runs/main_hybrid_delta_attention/best.pt --cache-dir runs/main_hybrid_delta_attention/cache --split val --device cuda
```

如需强制重建缓存：

```bash
python benchmark_latency.py --checkpoint runs/main_hybrid_delta_attention/best.pt --dataset data/train/dataset.csv --cache-dir runs/main_hybrid_delta_attention/cache --split val --rebuild-cache
```

## 项目声明

- 项目名称：基于组件分割与多源特征融合的恶意URL检测研究
- 项目作者：Yang Langxuan
- 开发语言：Python
- 框架：PyTorch
- 核心技术：URL 组件分割、多源特征融合、恶意链接检测
