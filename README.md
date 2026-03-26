# ChainEdgeLLM — Multi-dimensional Threat Activation Dataset Generator & Classifier
# 多维威胁激活状态数据集生成器与分类器

> **English** | [中文](#中文文档)

---

A complete pipeline for generating labeled hidden-state datasets from LLM decoder layers and training threat detection classifiers for decentralized edge computing security.

## Project Overview

This project consists of two main modules:

1. **threat_dataset** — Generates labeled datasets by capturing hidden states from LLM decoder layers and simulating 5 types of threats/attacks
2. **classify** — Trains and evaluates classical ML and neural network classifiers to detect threats from hidden-state vectors, using a **dual-track strategy** for both fast iteration and maximum performance

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STEP 1: Data Generation                             │
│                      (threat_dataset module)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │ Load Model  │───→│  Register   │───→│   Process   │───→│  Generate  │  │
│   │  + Dataset  │    │    Hooks    │    │ Instructions│    │  Threats   │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └────────────┘  │
│                                                                │            │
│                                                                ↓            │
│   Output: Parquet files (hidden_state_vector, label, layer_index...)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 2: Threat Classification                          │
│                         (classify module)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Track 1 (Agile Baseline)          Track 2 (Full Performance)             │
│   ┌──────────────────────┐          ┌──────────────────────────┐           │
│   │ Stratified Sample    │          │ Full Data (1.6M records)  │           │
│   │ (e.g. 100k records)  │          │ Streaming DataLoader      │           │
│   │ PCA 128-dim          │          │ Raw 4096-dim vectors      │           │
│   │ Classical ML + MLP   │          │ MLP_Light / MLP_Deep      │           │
│   │ Fast feedback        │          │ CNN_1D                    │           │
│   └──────────────────────┘          └──────────────────────────┘           │
│                                                                │            │
│                                                                ↓            │
│   Output: Metrics CSV, Confusion Matrices, Classification Reports          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module 1: threat_dataset (Data Generation)

Generate multi-dimensional threat activation datasets from LLM decoder layers.

### Directory Structure

```
threat_dataset/
├── __init__.py          # Package metadata
├── __main__.py          # python -m threat_dataset entry
├── config.py            # Constants, label map, Parquet schema, CLI args, RunConfig
├── logging_setup.py     # Dual-output logging (console INFO + file DEBUG) with RSS monitoring
├── model_loader.py      # HF model/tokenizer loading, decoder-layer discovery, inspection pass
├── hooks.py             # Forward-hook factory, LayerStateCapture, last-token extraction
├── threats.py           # 5 threat-label generators (pure NumPy)
├── buffer_io.py         # RecordBuffer with periodic Parquet flush (pyarrow)
└── main.py              # CLI orchestration loop
```

### CLI Usage

```bash
python -m threat_dataset [OPTIONS]
```

#### All CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `meta-llama/Meta-Llama-3-8B` | HuggingFace/ModelScope model identifier |
| `--dataset-size` | int | `5000` | Number of Alpaca instructions to process |
| `--output-dir` | str | `output` | Directory for Parquet part files |
| `--dtype` | str | `bf16` | Model precision: `bf16` or `fp16` |
| `--buffer-size` | int | `10000` | Records to buffer before flushing to disk |
| `--device` | str | `cuda` | PyTorch device (`cuda`, `cpu`, etc.) |
| `--log-dir` | str | `logs` | Directory for log files |
| `--source` | str | `huggingface` | Model source: `huggingface` or `modelscope` |
| `--step` | str | `all` | `download`, `inference`, or `all` |
| `--dataset-name` | str | — | Override default dataset identifier |

### Quick Start Examples

```bash
# Small test (5 instructions)
python -m threat_dataset --model meta-llama/Meta-Llama-3-8B --dataset-size 5 --step all

# ModelScope source (China)
python -m threat_dataset --model LLM-Research/Meta-Llama-3-8B --dataset-size 5 --source modelscope --step all

# Download model only
python -m threat_dataset --model meta-llama/Meta-Llama-3-8B --step download

# Full production run
python -m threat_dataset \
    --model LLM-Research/Meta-Llama-3-8B \
    --dataset-size 10000 \
    --step inference \
    --source modelscope \
    --output-dir /root/autodl-tmp/data
```

### Output Schema (Parquet)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | String | Unique ID, e.g. `00001_L015_T2` |
| `model_name` | String | Short model name, e.g. `Meta-Llama-3-8B` |
| `instruction_id` | Int32 | Index of the source instruction |
| `layer_index` | Int32 | Decoder layer index (0-based) |
| `label` | Int32 | Threat label (0–4) |
| `label_name` | String | Human-readable label name |
| `hidden_state_vector` | List\<Float32\> | Feature vector of length `hidden_size` (4096 for Llama3-8B) |

### Label Definitions

| Label | Name | Description |
|-------|------|-------------|
| 0 | `honest` | Genuine decoder output |
| 1 | `silent_precision_downgrade` | Simulated INT4 quantization round-trip |
| 2 | `identity_forgery` | Layer input passed through as output (skip attack) |
| 3 | `random_noise` | Gaussian noise matching output statistics (Byzantine fault) |
| 4 | `adversarial_perturbation` | Honest output + 5% std Gaussian noise |

### Resume Capability

The tool supports resuming interrupted runs via `resume_state.json` in the output directory — simply re-run the same command to continue from the last completed instruction.

---

## Module 2: classify (Threat Classification)

Train and evaluate ML classifiers to detect threats from hidden-state vectors.

### Directory Structure

```
classify/
├── __init__.py          # Package metadata
├── __main__.py          # python -m classify entry
├── main.py              # CLI and dual-track classification pipeline
├── nn_models.py         # PyTorch NN architectures + streaming Parquet dataset
├── sampling.py          # Stratified sampling utility
└── logging_setup.py     # Logging configuration (logs/ in project root)
```

### Dual-Track Strategy

| | Track 1 — Agile Baseline | Track 2 — Full Performance |
|---|---|---|
| **Data** | Stratified sample (e.g. 100k) | Full dataset (1.6M+) |
| **Features** | PCA 128-dim | Raw 4096-dim |
| **Models** | LR, RF, SVM, KNN, LightGBM, MLP | MLP_Light, MLP_Deep, CNN_1D |
| **Loading** | In-memory pandas | Streaming DataLoader (constant RAM) |
| **Purpose** | Fast feedback, baseline F1 | Unlock full data potential |

### CLI Usage

```bash
python -m classify [OPTIONS]
```

#### All CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-dir` | str | `/root/autodl-tmp/data/` | Directory containing .parquet data files |
| `--task` | str | `cross-layer` | `per-layer` or `cross-layer` |
| `--pca-dim` | int | `128` | PCA components for classical ML (0 to disable) |
| `--val-size` | float | `0.15` | Validation set ratio |
| `--test-size` | float | `0.15` | Test set ratio |
| `--seed` | int | `42` | Random seed |
| `--sample` | int | `0` | Stratified sample size for Track 1 (0 = all data) |
| `--track` | str | `auto` | `classical`, `nn-stream`, or `auto` (both) |
| `--nn-epochs` | int | `50` | Max epochs for neural network training |
| `--nn-batch-size` | int | `512` | Batch size for neural network training |
| `--nn-lr` | float | `1e-3` | Learning rate for neural networks |

#### `--task` modes

- `cross-layer`: Single classifier trained on hidden states from all layers mixed together
- `per-layer`: One classifier per decoder layer, each only sees its own layer's data

#### `--track` modes

- `classical`: Track 1 only — stratified sample + PCA + traditional ML
- `nn-stream`: Track 2 only — full data streaming + raw 4096-dim + neural networks
- `auto`: Run both tracks sequentially (default)

### Neural Network Architectures (Track 2)

| Name | Architecture | Purpose |
|------|-------------|---------|
| `MLP_Light` | `4096 → 512 → 5` | Pipeline validator, fastest |
| `MLP_Deep` | `4096 → 1024 → 256 → 64 → 5` | Main workhorse, BatchNorm + Dropout |
| `CNN_1D` | Conv1d sliding window on 4096-dim | Local activation pattern extractor |

### Classical ML Classifiers (Track 1)

| Classifier | Notes |
|------------|-------|
| LogisticRegression | With StandardScaler |
| RandomForest | 200 estimators |
| SVM_RBF | RBF kernel, One-vs-Rest |
| KNN | k=5 neighbors |
| LightGBM | If installed |

### Quick Start Examples

```bash
# Track 1 only: 100k stratified sample, fast baseline
python -m classify --data-dir /root/autodl-tmp/data --sample 100000 --track classical

# Track 2 only: full data streaming neural networks
python -m classify --data-dir /root/autodl-tmp/data --track nn-stream --nn-epochs 30

# Both tracks (recommended)
python -m classify --data-dir /root/autodl-tmp/data --sample 100000 --track auto

# Per-layer mode
python -m classify --data-dir /root/autodl-tmp/data --task per-layer --sample 100000 --track classical
```

### Output Files

Results are saved to `{data-dir}/classify_results/`, logs to `logs/` in the project root.

```
classify_results/
├── summary_{task}.csv                        # Full metrics summary (all models, all splits)
├── cm_{tag}_{classifier}_{split}.png         # Confusion matrices
├── report_{tag}_{classifier}_test.txt        # Classification reports
└── layer_{N}/                                # Per-layer results (per-layer task only)
    ├── cm_layer_{N}_{classifier}_{split}.png
    └── report_layer_{N}_{classifier}_test.txt

logs/
└── classify_{YYYYMMDD_HHMMSS}.log            # DEBUG-level log with timestamps + RSS
```

---

## Complete Pipeline Example

```bash
# Step 1: Generate dataset
python -m threat_dataset \
    --model LLM-Research/Meta-Llama-3-8B \
    --dataset-size 10000 \
    --step all \
    --source modelscope \
    --output-dir /root/autodl-tmp/data \
    --dtype bf16

# Step 2: Dual-track classification (100k baseline + full streaming NN)
python -m classify \
    --data-dir /root/autodl-tmp/data \
    --task cross-layer \
    --sample 100000 \
    --track auto
```

---

## Troubleshooting

**OOM (Out of Memory)**
- Track 1 uses PCA 128-dim — memory safe for any dataset size with `--sample`
- Track 2 uses streaming DataLoader — RAM stays constant regardless of dataset size
- Try `--dtype fp16` or a smaller model for data generation

**Slow throughput during data generation**
- Batch size is fixed at 1 (no padding) for correctness
- Expected: ~2–4 instructions/second on A100 with 7B model

**Type / compatibility errors**
- Requires `torch>=2.1` for bfloat16 support
- Add custom layer paths to `LAYER_ATTR_PATHS` in `config.py` for non-standard architectures

---

# 中文文档

> [English](#chainedgellm--multi-dimensional-threat-activation-dataset-generator--classifier) | **中文**

---

用于在去中心化边缘计算场景中生成大型语言模型解码层威胁激活数据集，并训练威胁检测分类器的完整流水线。

## 项目概述

本项目包含两个核心模块：

1. **threat_dataset** — 通过挂载 LLM 各 Decoder 层的 Forward Hook，截获隐藏状态并生成 5 种威胁/攻击类型的标记数据集
2. **classify** — 采用**双轨并行策略**，训练并评估经典机器学习和神经网络分类器，从隐藏状态向量中检测威胁

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         第一步：数据生成                                     │
│                      (threat_dataset 模块)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│   │  加载模型   │───→│   注册Hook  │───→│   处理指令  │───→│   生成威胁  │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └────────────┘  │
│   输出：Parquet 文件 (hidden_state_vector, label, layer_index...)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      第二步：威胁分类 (classify 模块)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   轨道一（敏捷基线）                    轨道二（全量冲刺）                   │
│   ┌──────────────────────┐          ┌──────────────────────────┐           │
│   │ 分层抽样（如10万条）  │          │ 全量数据（160万条）        │           │
│   │ PCA 128维降维         │          │ 流式 DataLoader           │           │
│   │ 传统ML + MLP          │          │ 原始 4096维向量            │           │
│   │ 快速反馈基线          │          │ MLP_Light/Deep + CNN_1D   │           │
│   └──────────────────────┘          └──────────────────────────┘           │
│   输出：指标 CSV、混淆矩阵、分类报告                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 模块一：threat_dataset (数据生成)

### 目录结构

```
threat_dataset/
├── config.py            # 常量、标签映射、Parquet Schema、CLI 参数
├── logging_setup.py     # 双输出日志（控制台 INFO + 文件 DEBUG）+ 内存监控
├── model_loader.py      # HF 模型/Tokenizer 加载、Decoder 层发现
├── hooks.py             # Forward Hook 工厂、最后 Token 提取
├── threats.py           # 5 种威胁标签生成函数（纯 NumPy）
├── buffer_io.py         # RecordBuffer，定期分片写入 Parquet
└── main.py              # CLI 主循环编排
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `meta-llama/Meta-Llama-3-8B` | 模型标识符 |
| `--dataset-size` | `5000` | 处理的指令数量 |
| `--output-dir` | `output` | Parquet 输出目录 |
| `--dtype` | `bf16` | 模型精度：`bf16` 或 `fp16` |
| `--buffer-size` | `10000` | 落盘前最大缓冲记录数 |
| `--source` | `huggingface` | 模型来源：`huggingface` 或 `modelscope` |
| `--step` | `all` | `download`、`inference` 或 `all` |

### 快速开始

```bash
# 小规模测试（国内 ModelScope 源）
python -m threat_dataset \
    --model LLM-Research/Meta-Llama-3-8B \
    --dataset-size 5 \
    --source modelscope \
    --step all

# 完整生产运行（支持断点续传）
python -m threat_dataset \
    --model LLM-Research/Meta-Llama-3-8B \
    --dataset-size 10000 \
    --step inference \
    --source modelscope \
    --output-dir /root/autodl-tmp/data
```

### 标签定义

| 标签 | 名称 | 描述 |
|------|------|------|
| 0 | `honest` | 诚实执行，真实输出 |
| 1 | `silent_precision_downgrade` | 模拟 INT4 量化截断误差 |
| 2 | `identity_forgery` | 跳层逃逸，直接返回上层输入 |
| 3 | `random_noise` | 同均值/方差的随机高斯噪声（拜占庭故障） |
| 4 | `adversarial_perturbation` | 诚实输出 + 5% 标准差高斯扰动 |

---

## 模块二：classify (威胁分类)

### 目录结构

```
classify/
├── main.py              # CLI 和双轨分类流水线
├── nn_models.py         # PyTorch 神经网络架构 + 流式 Parquet 数据集
├── sampling.py          # 分层抽样工具
└── logging_setup.py     # 日志配置（输出到项目根目录 logs/）
```

### 双轨并行策略

| | 轨道一（敏捷基线） | 轨道二（全量冲刺） |
|---|---|---|
| **数据量** | 分层抽样（如10万条） | 全量（160万条+） |
| **特征** | PCA 128维 | 原始 4096维 |
| **模型** | LR、RF、SVM、KNN、LightGBM | MLP_Light、MLP_Deep、CNN_1D |
| **加载方式** | Pandas 内存加载 | 流式 DataLoader（内存恒定） |
| **目的** | 快速拿到基线 F1，排雷 | 释放全量数据潜力 |

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `/root/autodl-tmp/data/` | Parquet 数据目录 |
| `--task` | `cross-layer` | `per-layer` 或 `cross-layer` |
| `--pca-dim` | `128` | PCA 降维维度（0 禁用） |
| `--val-size` | `0.15` | 验证集比例 |
| `--test-size` | `0.15` | 测试集比例 |
| `--seed` | `42` | 随机种子 |
| `--sample` | `0` | 轨道一分层抽样数量（0 = 全量） |
| `--track` | `auto` | `classical`、`nn-stream` 或 `auto`（双轨） |
| `--nn-epochs` | `50` | 神经网络最大训练轮数 |
| `--nn-batch-size` | `512` | 神经网络批大小 |
| `--nn-lr` | `1e-3` | 神经网络学习率 |

### 神经网络架构（轨道二）

| 名称 | 结构 | 用途 |
|------|------|------|
| `MLP_Light` | `4096 → 512 → 5` | 管道验证，速度最快 |
| `MLP_Deep` | `4096 → 1024 → 256 → 64 → 5` | 主力模型，BatchNorm + Dropout |
| `CNN_1D` | Conv1d 滑动卷积 | 局部激活模式提取，参数量更少 |

### 快速开始

```bash
# 轨道一：10万条快速基线
python -m classify --data-dir /root/autodl-tmp/data --sample 100000 --track classical

# 轨道二：全量流式神经网络
python -m classify --data-dir /root/autodl-tmp/data --track nn-stream --nn-epochs 30

# 双轨并行（推荐）
python -m classify --data-dir /root/autodl-tmp/data --sample 100000 --track auto
```

### 输出文件

结果保存到 `{data-dir}/classify_results/`，日志保存到项目根目录 `logs/`。

```
classify_results/
├── summary_{task}.csv                        # 全量指标汇总
├── cm_{tag}_{classifier}_{split}.png         # 混淆矩阵
├── report_{tag}_{classifier}_test.txt        # 分类报告
└── layer_{N}/                                # 逐层结果（per-layer 任务）

logs/
└── classify_{YYYYMMDD_HHMMSS}.log            # DEBUG 级别日志，含时间戳和内存占用
```

---

## 完整流水线示例

```bash
# 第一步：生成数据集
python -m threat_dataset \
    --model LLM-Research/Meta-Llama-3-8B \
    --dataset-size 10000 \
    --step all \
    --source modelscope \
    --output-dir /root/autodl-tmp/data \
    --dtype bf16

# 第二步：双轨分类
python -m classify \
    --data-dir /root/autodl-tmp/data \
    --task cross-layer \
    --sample 100000 \
    --track auto
```

---

## 常见问题

**显存不足（OOM）**
- 轨道一使用 PCA 128维 + 分层抽样，内存安全
- 轨道二使用流式 DataLoader，内存占用恒定
- 数据生成时尝试 `--dtype fp16` 或换用更小的模型

**推理速度慢**
- Batch Size 强制为 1（无填充）以确保正确性
- A100 上 7B 模型预计约 2–4 条指令/秒

**类型兼容问题**
- 需要 `torch>=2.1` 以支持 bfloat16
- 非标准架构请在 `config.py` 的 `LAYER_ATTR_PATHS` 中添加层路径

---

## License

本项目代码遵循相应开源许可证。原始模型（Meta-Llama-3-8B 等）受其自身许可证约束。

## Acknowledgments

- 基础模型：[Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) by Meta AI
- 指令数据集：[Alpaca GPT-4](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en) 基于 Stanford Alpaca
