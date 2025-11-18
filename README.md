# MMF-GAT 行业股票预测模型

基于多尺度时间特征提取和图注意力网络的行业股票收益率预测模型。

## 📋 项目简介

本项目实现了一个用于行业股票收益率预测的深度学习模型，结合了以下核心技术：

- **多尺度时间编码器（Multi-Scale Time Encoder）**：使用共享参数的 Transformer 编码器处理不同时间窗口（20、40、80天）的K线数据
- **DWT增强（Discrete Wavelet Transform）**：使用小波变换增强时间序列特征
- **动态注意力门控（Dynamic Attention Gate）**：自适应融合多尺度特征
- **图注意力网络（GAT）**：建模行业间的关系，利用行业关联性提升预测效果
- **学习压缩层（Learning Compression Layer）**：压缩时间特征，为GAT提供输入

模型将未来30天的收益率分为5个分位数（Q1-Q5），实现分类预测。

## 🏗️ 模型架构

```
输入: 行业K线数据 (20/40/80天窗口)
  ↓
DWT增强 (可选)
  ↓
多尺度Transformer编码器 (共享参数)
  ↓
动态注意力门控融合
  ↓
学习压缩层 (LCL)
  ↓
GAT图注意力网络 (行业关系建模)
  ↓
分类预测头
  ↓
输出: 5分位数分类 (Q1-Q5)
```

### 核心组件

1. **DWTEnhancement** (`components/dwt_enhancement.py`)
   - 使用小波变换（db4）增强时间序列特征
   - 提取多频率信息

2. **MultiScaleTimeEncoder** (`components/time_encoder.py`)
   - 共享参数的Transformer编码器
   - 处理不同长度的时间窗口

3. **DynamicAttentionGate** (`components/dynamic_gate.py`)
   - 自适应权重分配
   - 融合多尺度特征

4. **GAT** (`components/gat_layer.py`)
   - 图注意力网络
   - 建模行业间关系（基于申万行业分类）

5. **IndustryStockModel** (`components/model.py`)
   - 整合所有组件的完整模型

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. **收集行业数据**（可选，如果已有数据可跳过）

```python
from components.industry_data_get import collect_industry_data

# 收集行业K线数据
collect_industry_data(output_dir="./data")
```

2. **准备数据文件**

确保 `data/` 目录下包含以下文件：
- `industry_kline_data.json`: 行业K线数据
- `industry_relation.csv`: 行业关系数据（包含 `industry` 和 `sw_industry` 列）

数据格式示例：
- `industry_kline_data.json`: `{"行业名称": [["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", ...], ...]}`
- `industry_relation.csv`: 包含 `industry`（行业名称）和 `sw_industry`（申万行业分类）列

## 📖 使用方法

### 训练模型

#### 1. 使用默认配置训练

```bash
python train.py
```

#### 2. 使用自定义配置文件

```bash
python train.py --config config/my_config.yaml
```

#### 3. 命令行参数覆盖

```bash
# 修改训练参数
python train.py --batch_size 64 --lr 0.0002 --epochs 100

# 修改模型参数
python train.py --d_model 256 --nhead 16 --use_dwt

# 使用K折验证
python train.py --use_kfold --n_splits 5

# 指定设备
python train.py --device cuda
```

### 配置文件说明

配置文件采用 YAML 格式，主要包含以下部分：

```yaml
# 数据配置
data:
  data_dir: "./data"
  window_20: 20
  window_40: 40
  window_80: 80
  future_days: 30
  num_classes: 5
  use_kfold: true
  n_splits: 3

# 模型配置
model:
  input_features: 7
  use_dwt: true
  time_encoder:
    d_model: 128
    nhead: 8
  gat:
    hidden_features: 128
    out_features: 64
    num_heads: 8
    num_layers: 2

# 训练配置
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  device: "auto"  # auto, cuda, cpu, mps
```

详细配置说明请参考 `config/default_config.yaml`。

### 推理预测

```bash
python predict.py --checkpoint ./checkpoints/best_model.pth --config config/default_config.yaml
```

## 📁 项目结构

```
transformer/
├── components/              # 核心组件模块
│   ├── __init__.py
│   ├── model.py            # 主模型
│   ├── data_loader.py      # 数据加载和预处理
│   ├── trainer.py          # 训练器
│   ├── config_loader.py    # 配置加载
│   ├── visualizer.py       # 可视化工具
│   ├── metrics.py          # 金融指标计算
│   ├── validator.py        # K折验证
│   ├── time_encoder.py     # 时间编码器
│   ├── dwt_enhancement.py  # DWT增强
│   ├── dynamic_gate.py     # 动态门控
│   ├── gat_layer.py        # GAT层
│   └── industry_data_get.py # 数据收集
├── config/                 # 配置文件
│   └── default_config.yaml
├── data/                   # 数据目录
│   ├── industry_kline_data.json
│   └── industry_relation.csv
├── checkpoints/            # 模型检查点
├── visualizations/         # 可视化结果
├── logs/                   # 日志文件
├── predictions/            # 预测结果
├── example_train.py        # 训练示例脚本
├── predict.py              # 推理脚本
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 🔧 核心功能

### 1. 数据加载 (`components/data_loader.py`)

- `IndustryDataLoader`: 加载和预处理行业K线数据
- `IndustryDataset`: PyTorch数据集类
- 支持多时间窗口（20/40/80天）
- 自动构建行业关系邻接矩阵

### 2. 模型训练 (`components/trainer.py`)

- `Trainer`: 完整的训练流程
- 支持标准训练/验证分割
- 支持时间序列K折交叉验证
- 自动计算金融指标（IC、RankIC、多空收益等）
- 学习率调度和梯度裁剪

### 3. 可视化 (`components/visualizer.py`)

- 训练曲线可视化
- 混淆矩阵
- K折验证结果
- 注意力权重热力图
- 分位数收益分析

### 4. 金融指标 (`components/metrics.py`)

- IC (Information Coefficient)
- RankIC
- 多空组合收益
- 分位数收益分析

## 📊 评估指标

模型评估使用以下指标：

1. **分类指标**
   - 准确率（Accuracy）
   - 混淆矩阵

2. **金融指标**
   - IC（信息系数）：预测值与真实收益率的相关系数
   - RankIC：预测排名与收益率排名的相关系数
   - 多空组合收益：做多高预测值组合，做空低预测值组合的收益

## 🎯 使用示例

### 基本训练流程

```python
from components.data_loader import IndustryDataLoader, IndustryDataset
from components.model import IndustryStockModel
from components.trainer import Trainer
from components.config_loader import load_config_with_cli, get_device

# 1. 加载配置
config, args = load_config_with_cli()

# 2. 加载数据
data_loader = IndustryDataLoader(
    data_dir=config.data.data_dir,
    window_sizes=[20, 40, 80],
    future_days=30
)
samples, targets, adj_matrix = data_loader.prepare_data()
dataset = IndustryDataset(samples, targets)

# 3. 创建模型
model = IndustryStockModel(
    input_features=7,
    time_encoder_dim=128,
    compression_dim=64,
    gat_hidden_dim=128,
    gat_output_dim=64,
    num_classes=5
)

# 4. 训练
trainer = Trainer(model, device=get_device("auto"))
history = trainer.train(train_loader, val_loader, adj_matrix, num_epochs=50)
```

### K折交叉验证

```python
# 在配置文件中设置 use_kfold: true
fold_results = trainer.k_fold_validate(
    dataset=dataset,
    adj_matrix=adj_matrix,
    n_splits=5,
    num_epochs=30
)
```

## ⚙️ 配置参数说明

### 数据配置

- `data_dir`: 数据目录路径
- `window_20/40/80`: 时间窗口大小
- `future_days`: 预测未来天数
- `num_classes`: 分类类别数（5分位数）
- `use_kfold`: 是否使用K折验证
- `n_splits`: K折折数

### 模型配置

- `input_features`: 输入特征数（K线特征：开盘、收盘、最高、最低、成交量、成交额、收益率）
- `use_dwt`: 是否使用DWT增强
- `time_encoder.d_model`: 时间编码器维度
- `time_encoder.nhead`: 注意力头数
- `gat.num_layers`: GAT层数
- `gat.num_heads`: GAT注意力头数

### 训练配置

- `batch_size`: 批大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `use_scheduler`: 是否使用学习率调度
- `max_grad_norm`: 梯度裁剪阈值
- `device`: 计算设备（auto/cuda/cpu/mps）

## 🔍 常见问题

### Q: 如何选择时间窗口大小？

A: 默认使用20/40/80天的多尺度窗口。可以根据数据特点调整，建议保持多尺度以捕获不同时间周期的特征。

### Q: 如何调整模型大小？

A: 修改配置文件中的 `time_encoder.d_model`、`gat.hidden_features` 等参数。更大的模型需要更多计算资源，但可能获得更好的性能。

### Q: 如何使用GPU训练？

A: 设置 `training.device: "cuda"` 或使用命令行参数 `--device cuda`。

### Q: K折验证和标准训练的区别？

A: K折验证更适合时间序列数据，可以更好地评估模型泛化能力。标准训练速度更快，适合快速迭代。

## 📝 注意事项

1. **数据格式**：确保K线数据格式正确，包含必要的字段
2. **行业关系**：`industry_relation.csv` 需要包含 `industry` 和 `sw_industry` 列
3. **内存管理**：大批量训练时注意内存使用，可适当减小 `batch_size`
4. **随机种子**：设置 `experiment.seed` 以保证可复现性

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**：本项目为学术研究项目，不构成投资建议。使用本模型进行实际投资决策需自行承担风险。

