# ChainEdgeLLM — Multi-dimensional Threat Activation Dataset Generator
# 多维威胁激活状态数据集生成器

> **English** | [中文](#中文文档)

---

Generate labeled hidden-state datasets from LLM decoder layers for training inference-state discriminators in decentralized edge computing scenarios.

## System Architecture

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

**Data flow:**

```
CLI args → RunConfig → load model (bf16) + tokenizer
                          → discover decoder layers
                          → inspection pass (verify tensor structure)
                          → load Alpaca instructions
                          → FOR EACH instruction:
                              tokenize → model.forward (no_grad, batch=1)
                                → hooks capture last-token hidden states per layer
                                → generate 5 threat variants per layer
                                → buffer records → flush to Parquet when threshold reached
```

Each instruction produces `num_layers × 5` records (1 honest + 4 threat variants per layer).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run (small test)

```bash
# Test with 5 instructions to verify everything works
python -m threat_dataset --model meta-llama/Meta-Llama-3-8B --dataset-size 5 --step "download" 

# 使用魔塔，注意 model 参数换成了 LLM-Research/Meta-Llama-3-8B
python -m threat_dataset --model LLM-Research/Meta-Llama-3-8B --dataset-size 5 --step "all" --source "modelscope"

# Or with Qwen
python -m threat_dataset --model Qwen/Qwen1.5-7B --dataset-size 5 --step "download"
```

### 3. Full dataset generation

```bash
python -m threat_dataset \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset-size 5000 \
    --output-dir output \
    --dtype bf16 \
    --buffer-size 10000 \
    --device cuda
```

### 4. Verify output

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("output/")
print(df.shape)
print(df["label"].value_counts())
print(df.iloc[0]["hidden_state_vector"][:5])  # first 5 elements

# NaN/Inf check
assert not df["hidden_state_vector"].apply(
    lambda v: any(np.isnan(v)) or any(np.isinf(v))
).any()
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `meta-llama/Meta-Llama-3-8B` | HuggingFace model identifier |
| `--dataset-size` | `5000` | Number of Alpaca instructions to process |
| `--output-dir` | `output` | Directory for Parquet part files |
| `--dtype` | `bf16` | Model precision (`bf16` or `fp16`) |
| `--buffer-size` | `10000` | Records to buffer before flushing to disk |
| `--device` | `cuda` | PyTorch device |
| `--log-dir` | `logs` | Directory for log files |

## Output Schema (Parquet)

| Column | Type | Description |
|---|---|---|
| `sample_id` | String | Unique ID, e.g. `00001_L015_T2` |
| `model_name` | String | Short model name, e.g. `Meta-Llama-3-8B` |
| `instruction_id` | Int32 | Index of the source instruction |
| `layer_index` | Int32 | Decoder layer index (0-based) |
| `label` | Int32 | Threat label (0–4) |
| `label_name` | String | Human-readable label name |
| `hidden_state_vector` | List\<Float32\> | Feature vector of length `hidden_size` |

### Label Definitions

| Label | Name | Description |
|---|---|---|
| 0 | `honest` | Genuine decoder output |
| 1 | `silent_precision_downgrade` | Simulated INT4 quantization round-trip |
| 2 | `identity_forgery` | Layer input passed through as output (skip attack) |
| 3 | `random_noise` | Gaussian noise matching output statistics |
| 4 | `adversarial_perturbation` | Honest output + 5% std Gaussian noise |

## Troubleshooting

### OOM (Out of Memory)

- The tool extracts only the **last token's** hidden state per layer (not the full sequence), so GPU memory should be dominated by the model itself.
- If OOM still occurs, try `--dtype fp16` (slightly smaller than bf16 on some hardware) or use a smaller model.
- Reduce `--buffer-size` if CPU RAM is tight (each buffered record holds a `hidden_size`-length float32 list).

### Type / Compatibility Errors

- **bfloat16 → NumPy**: The code explicitly casts to `float32` before `.numpy()`. If you see a dtype error, check your PyTorch version (`torch>=2.1` required).
- **Hook tensor structure**: The tool runs an **inspection pass** on layer 0 before the main loop. Check the log output for `input[0]_shape` and `output[0]_shape` — they should both be `[1, seq_len, hidden_dim]`. If not, your transformers version may wrap tensors differently.
- **Layer discovery fails**: The tool tries `model.model.layers` then `model.layers`. For non-standard architectures, you may need to add the correct attribute path to `LAYER_ATTR_PATHS` in `config.py`.

### Slow Throughput

- Batch size is intentionally fixed at 1 (no padding) to ensure `[:, -1, :]` always captures the true last token. This is a correctness-over-speed trade-off.
- For 5000 instructions on a 7B model with an A100, expect ~2–4 instructions/second.

---

# 中文文档

> [English](#chainedgellm--multi-dimensional-threat-activation-dataset-generator) | **中文**

---

在去中心化边缘计算场景中，通过挂载 LLM 各 Decoder 层的 Forward Hook，截获诚实前向传播的中间隐状态，并生成 4 种恶意/异常伪造状态，输出结构化 Parquet 特征数据集，用于训练轻量级推理状态判别器。

## 系统架构

```
threat_dataset/
├── __init__.py          # 包元数据
├── __main__.py          # python -m threat_dataset 入口
├── config.py            # 常量、标签映射、Parquet Schema、CLI 参数、RunConfig
├── logging_setup.py     # 双输出日志（控制台 INFO + 文件 DEBUG）+ 内存监控
├── model_loader.py      # HF 模型/Tokenizer 加载、Decoder 层发现、结构探查
├── hooks.py             # Forward Hook 工厂、LayerStateCapture、最后 Token 提取
├── threats.py           # 5 种威胁标签生成函数（纯 NumPy）
├── buffer_io.py         # RecordBuffer，定期分片写入 Parquet（pyarrow）
└── main.py              # CLI 主循环编排
```

**数据流向：**

```
CLI 参数 → RunConfig → 加载模型（bf16）+ Tokenizer
                          → 发现 Decoder 层
                          → 结构探查（校验张量结构）
                          → 加载 Alpaca 指令集
                          → 逐条处理指令：
                              Tokenize → model.forward（no_grad，batch=1）
                                → Hook 截获各层最后 Token 隐状态
                                → 每层生成 5 种威胁变体
                                → 缓冲记录 → 达到阈值时写入 Parquet
```

每条指令产生 `层数 × 5` 条记录（每层 1 条诚实 + 4 条威胁变体）。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 小规模测试

```bash
# 用 5 条指令验证流程是否正常
python -m threat_dataset --model meta-llama/Meta-Llama-3-8B --dataset-size 5 --step "download"

# 使用 Qwen 模型
python -m threat_dataset --model Qwen/Qwen1.5-7B --dataset-size 5 --step "download"
```

### 3. 完整数据集生成

```bash
python -m threat_dataset \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset-size 5000 \
    --output-dir output \
    --dtype bf16 \
    --buffer-size 10000 \
    --device cuda
```

### 4. 验证输出

```python
import pandas as pd
import numpy as np

df = pd.read_parquet("output/")
print(df.shape)
print(df["label"].value_counts())
print(df.iloc[0]["hidden_state_vector"][:5])  # 前 5 个元素

# NaN/Inf 检查
assert not df["hidden_state_vector"].apply(
    lambda v: any(np.isnan(v)) or any(np.isinf(v))
).any()
```

### CLI 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `meta-llama/Meta-Llama-3-8B` | HuggingFace 模型标识符 |
| `--dataset-size` | `5000` | 处理的 Alpaca 指令数量 |
| `--output-dir` | `output` | Parquet 分片文件输出目录 |
| `--dtype` | `bf16` | 模型精度（`bf16` 或 `fp16`） |
| `--buffer-size` | `10000` | 落盘前最大缓冲记录数 |
| `--device` | `cuda` | PyTorch 运行设备 |
| `--log-dir` | `logs` | 日志文件目录 |

## 输出数据结构（Parquet Schema）

| 列名 | 类型 | 说明 |
|---|---|---|
| `sample_id` | String | 唯一标识符，如 `00001_L015_T2` |
| `model_name` | String | 模型短名称，如 `Meta-Llama-3-8B` |
| `instruction_id` | Int32 | 原始指令在数据集中的索引 |
| `layer_index` | Int32 | Decoder 层索引（从 0 开始） |
| `label` | Int32 | 威胁分类标签（0–4） |
| `label_name` | String | 标签文本名称 |
| `hidden_state_vector` | List\<Float32\> | 长度为 `hidden_size` 的特征向量 |

### 标签定义

| 标签 | 名称 | 描述 |
|---|---|---|
| 0 | `honest` | 诚实执行，直接保存真实输出 |
| 1 | `silent_precision_downgrade` | 模拟 INT4 对称量化后反量化的截断误差 |
| 2 | `identity_forgery` | 跳层逃逸，直接返回上层输入 |
| 3 | `random_noise` | 与输出同均值/方差的随机高斯噪声（拜占庭故障） |
| 4 | `adversarial_perturbation` | 诚实输出 + 5% 标准差高斯扰动（微小投毒） |

## 常见问题排查

### 显存不足（OOM）

- 本工具仅在 Hook 内部提取**最后一个 Token** 的隐状态（而非完整序列），显存主要被模型本身占用。
- 仍然 OOM 时，可尝试 `--dtype fp16`，或换用参数量更小的模型。
- CPU 内存紧张时，调低 `--buffer-size`（每条缓冲记录包含一个 `hidden_size` 长度的 float32 向量）。

### 类型兼容问题

- **bfloat16 → NumPy**：代码在 `.numpy()` 前已显式转为 `float32`。若仍报 dtype 错误，请确认 PyTorch 版本 ≥ 2.1。
- **Hook 张量结构异常**：主循环前会对 Layer 0 执行**结构探查（Inspection Pass）**，日志中会打印 `input[0]_shape` 和 `output[0]_shape`，二者应均为 `[1, seq_len, hidden_dim]`。如不符，说明当前 transformers 版本的张量封装方式有差异。
- **层路径发现失败**：工具依次尝试 `model.model.layers` 和 `model.layers`。非标准架构请在 `config.py` 的 `LAYER_ATTR_PATHS` 中添加正确路径。

### 推理速度慢

- Batch Size 强制为 1（禁止 Padding），以确保 `[:, -1, :]` 始终取到真实的最后 Token，这是正确性优先于速度的设计决策。
- 在 A100 上用 7B 模型处理 5000 条指令，预计速度约为 2–4 条/秒。
