# 横截面局部训练技术文档

## 概述

本文档详细说明了MMF-GAT模型的横截面局部训练架构，该架构解决了原有训练方式中的时间对齐和特征覆盖问题。

---

## 核心问题

### 问题1：时间对齐问题

**原有训练方式**：
```python
# Batch中的样本来自不同时期
Sample 1: 行业A, 2023-01-01 到 2023-03-21  (牛市)
Sample 2: 行业B, 2023-06-15 到 2023-09-03  (熊市)
Sample 3: 行业C, 2023-11-20 到 2024-02-07  (震荡)
```

**问题**：
- GAT理论要求同一时间点的横截面数据
- 混合不同市场状态导致学习到错误的行业关联

### 问题2：特征覆盖问题

**原有实现**：
```python
# 同一行业在batch中多次出现
industry_indices = [5, 12, 5, 23]  # 行业5出现2次

# 在_process_subgraph中
for i, idx in enumerate(industry_indices):
    subgraph_features[idx] = compressed_features[i]  # 后者覆盖前者！
```

**问题**：
- 只有最后出现的样本特征被保留
- 前面样本的计算被浪费

---

## 解决方案：横截面局部训练

### 架构设计

```
┌──────────────────────────────────────────────────────┐
│          横截面数据（同一时间窗口）                  │
│  时间步t: 所有86个行业的[t-80:t]数据                │
└──────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │    度数采样器                │
        │  选择k=12个中心节点          │
        └─────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────┐
│  中心节点 (12个): 采样的重要行业                    │
│  邻居节点 (~15个): 中心节点的1-hop邻居              │
│  掩码节点 (~60个): 其他未被采样的行业               │
└──────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │    节点特征填充              │
        │  - 中心+邻居: 门控融合      │
        │  - 掩码节点: 纯嵌入         │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  完整86节点图GAT传播        │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  提取中心+邻居输出          │
        │  计算损失并反向传播         │
        └─────────────────────────────┘
```

---

## 核心组件

### 1. 度数采样器 (`degree_sampler.py`)

**DegreeBasedSampler类**:

```python
sampler = DegreeBasedSampler(adj_matrix, temperature=1.0)

# 采样中心节点（度数高的行业更容易被选中）
center_nodes = sampler.sample(k=12)  # [5, 12, 23, 45, ...]

# 获取1-hop邻居
neighbors = sampler.get_neighbors(center_nodes)  # [1, 3, 8, ...]
```

**参数说明**:
- `temperature`:
  - 1.0: 严格按度数采样
  - >1.0: 更均匀（弱化度数差异）
  - <1.0: 更陡峭（强化度数差异）

**采样概率**:
```python
degrees = adj_matrix.sum(dim=1)  # 每个节点的度数
probs = softmax(degrees / temperature)
```

---

### 2. 节点级门控层 (`node_level_gate.py`)

**NodeLevelGate类**:

```python
gate = NodeLevelGate(feature_dim=64, hidden_dim=64)

# 自适应融合
fused_features, gates = gate(time_features, embeddings)
# gates ∈ [0, 1]: 0=完全使用嵌入, 1=完全使用时间特征
```

**融合公式**:
```
g = sigmoid(MLP(concat(time_features, embeddings)))
output = g × time_features + (1-g) × embeddings
```

**优势**:
- 每个节点独立学习融合权重
- 模型自主决定依赖时间特征还是嵌入
- 可解释性强（可通过门控值分析）

---

### 3. 横截面局部数据集 (`cross_sectional_dataset.py`)

**CrossSectionalLocalDataset类**:

```python
dataset = CrossSectionalLocalDataset(
    cross_sectional_data=cross_sec_data,  # 横截面数据
    adj_matrix=adj_matrix,
    num_centers=12,  # 每次采样12个中心
    sampler_type='degree',
    samples_per_timestep=8  # 每时间步采样8次
)
```

**数据组织**:
```
- 外层: 时间步 (~900个)
- 内层: 每时间步8次采样
- 总样本: 900 × 8 = 7,200
```

**返回格式**:
```python
{
    'sequence': [num_active, 80, 7],      # 中心+邻居的序列
    'target': [num_active],                # 标签
    'mask': [num_active, 80],              # 时间掩码
    'industry_idx': [num_active],          # 行业索引
    'node_mask': [86],                     # bool, 哪些节点有输入
    'center_mask': [num_active],           # bool, 哪些是中心
    'time_index': scalar                   # 时间步索引
}
```

---

### 4. 模型修改 (`model.py`)

**新增参数**:
```python
IndustryStockModel(
    ...,
    use_node_gate=True,         # 节点级门控
    gate_hidden_dim=64,         # 门控MLP维度
    cross_sectional_mode=True   # 横截面模式
)
```

**新增方法**:
```python
def _process_cross_sectional_subgraph(
    compressed_features,  # [num_active, 64] 有输入节点的时间特征
    industry_indices,     # [num_active] 有输入节点的索引
    adj_matrix,           # [86, 86] 完整邻接矩阵
    node_mask             # [86] bool, True=有输入
):
    # 1. 初始化所有86个节点的嵌入
    all_embeddings = industry_embeddings(range(86))

    # 2. 填充有输入节点的时间特征
    all_time_features[active_indices] = compressed_features

    # 3. 节点级门控融合
    fused_features[active_indices], gates = node_gate(
        all_time_features[active_indices],
        all_embeddings[active_indices]
    )

    # 4. 掩码节点保持纯嵌入
    fused_features[masked_indices] = all_embeddings[masked_indices]

    # 5. 完整86节点图GAT
    gat_output = gat(fused_features, adj_matrix)  # [86, gat_output_dim]

    # 6. 提取有输入节点的输出
    return gat_output[industry_indices], gates
```

---

### 5. 训练器修改 (`trainer.py`)

**新增方法**:
```python
def train_epoch_cross_sectional(dataloader, adj_matrix, epoch):
    """横截面局部训练epoch"""

    # 时间步追踪
    current_time_step = -1
    time_step_losses = []

    # 门控值统计
    all_gate_values = []

    for batch in dataloader:
        time_idx = batch['time_index'].item()

        # 检测新时间步
        if time_idx != current_time_step:
            if current_time_step >= 0:
                # 记录上一时间步统计
                print(f"Time step {current_time_step}: "
                      f"Loss={np.mean(time_step_losses):.4f}")
            current_time_step = time_idx
            time_step_losses = []

        # 前向传播（传递node_mask）
        predictions, _, gates = model(
            ...,
            node_mask=batch['node_mask']
        )

        # 收集门控值
        if gates is not None:
            all_gate_values.append(gates)

        # 正常的反向传播...

    # 返回包含门控统计的指标
    return {..., 'gate_mean': ..., 'favor_time_ratio': ...}
```

---

## 配置说明

### `config/default_config.yaml`

```yaml
data:
  # ⭐ 横截面局部训练配置
  use_cross_sectional_training: true
  num_center_nodes: 12       # 每次采样的中心节点数
  sampler_type: "degree"     # 采样器类型
  sampler_temperature: 1.0   # 度数采样温度
  samples_per_timestep: null # 每时间步采样次数（null=自动）

  # K折验证建议关闭
  use_kfold: false

model:
  # ⭐ 节点级门控配置
  use_node_gate: true
  gate_hidden_dim: 64
  cross_sectional_mode: true

  # 行业嵌入
  num_industries: 86
  use_industry_embedding: true
  embedding_fusion_alpha: 1.0  # 仅在use_node_gate=false时生效
```

---

## 使用方法

### 基本训练

```bash
# 使用默认配置（横截面局部训练）
python train.py
```

### 参数调整

```bash
# 调整中心节点数
python train.py --num_center_nodes 16

# 调整采样器温度
python train.py --sampler_temperature 0.8

# 使用顺序采样
python train.py --sampler_type sequential
```

### 对比实验

**实验1: 横截面局部训练（推荐）**
```yaml
data:
  use_cross_sectional_training: true
model:
  use_node_gate: true
  cross_sectional_mode: true
```

**实验2: 传统训练（Baseline）**
```yaml
data:
  use_cross_sectional_training: false
model:
  use_node_gate: false
  cross_sectional_mode: false
```

---

## Epoch定义

**时间步优先模式**:

```
Epoch 1:
  Time Step 0: (所有86个行业在t=0的数据)
    Sample 1: 采样中心[5,12,23...] + 邻居
    Sample 2: 采样中心[8,15,34...] + 邻居
    ...
    Sample 8: 采样中心[77,81,85...] + 邻居

  Time Step 1: (所有86个行业在t=1的数据)
    Sample 9: ...
    ...

  ...

  Time Step 899:
    Sample 7193-7200

Epoch 2: (重新开始)
  Time Step 0: (重新采样，可能选择不同的中心节点)
    ...
```

**统计保证**:
- 每个时间步采样8次 × 每次12个中心 = 96个中心位置
- 86个行业 → 每个行业平均被选为中心 ≈ 1.1次/时间步
- 加上作为邻居的机会 → 每个行业充分参与训练

---

## 参数统计

### 新增参数

```python
# 节点级门控
gate_params = (64*2)*64 + 64*1 = 8,320

# 行业嵌入（已有）
embedding_params = 86*64 = 5,504

# 总新增
total_new = 13,824 参数

# 占比
ratio = 13,824 / 661,608 ≈ 2.1%
```

### 计算复杂度

- 嵌入查找: O(86) - 可忽略
- 门控MLP: O(num_active × 64²) ≈ O(27 × 4096) - 很小
- GAT (86节点): 与之前相同
- **总结**: 几乎不增加计算开销

---

## 预期效果

### 优势

✅ **理论正确性**: 横截面数据，GAT学习同一市场状态下的行业关系
✅ **训练效率**: 每批只处理~27个节点的时间特征
✅ **参数学习**: 行业嵌入通过GAT梯度持续更新
✅ **自适应融合**: 节点级门控让模型自主决定融合策略
✅ **完整信息**: GAT在完整86节点图上传播消息
✅ **可解释性**: 门控值显示模型的融合偏好

### 改进对比

| 指标 | 传统训练 | 横截面局部训练 |
|------|---------|---------------|
| 时间对齐 | ❌ 混合时期 | ✅ 同一时间点 |
| 特征覆盖 | ❌ 会覆盖 | ✅ 无覆盖 |
| 理论正确性 | ⚠️ 有问题 | ✅ 符合GAT假设 |
| 训练效率 | 一般 | ✅ 更高效 |
| 可解释性 | 一般 | ✅ 门控值分析 |

---

## 门控值分析

### 查看门控统计

训练时会自动打印：

```
Epoch 1/20
Time step 0: Loss=1.5234, Acc=28.3%
Time step 1: Loss=1.4876, Acc=31.2%
...

Epoch Statistics:
  Loss: 1.4523
  Accuracy: 32.5%
  Gate Mean: 0.647      # 平均门控值
  Gate Std: 0.182       # 门控值标准差
  Favor Time Ratio: 0.73  # 73%节点更依赖时间特征
```

### 门控值解释

- `gate_mean` ≈ 1.0: 模型强烈依赖时间特征
- `gate_mean` ≈ 0.5: 平衡使用时间特征和嵌入
- `gate_mean` ≈ 0.0: 模型强烈依赖行业嵌入

**典型现象**:
- 训练初期: gate_mean ≈ 0.5 （探索阶段）
- 训练中期: gate_mean 上升（学会使用时间特征）
- 训练后期: 稳定在0.6-0.8（自适应平衡）

---

## 故障排查

### Q1: 训练速度很慢

**原因**: 每个时间步多次采样导致总样本数增加

**解决方案**:
```yaml
data:
  samples_per_timestep: 5  # 减少采样次数（默认8）
  num_center_nodes: 8      # 减少中心节点数（默认12）
```

### Q2: 内存不足

**原因**: 完整86节点图GAT

**解决方案**:
```yaml
model:
  gat:
    num_layers: 1  # 减少GAT层数（默认2）
    hidden_features: 64  # 减少隐藏维度（默认128）
```

### Q3: 门控值统计缺失

**原因**: use_node_gate=false

**解决方案**:
```yaml
model:
  use_node_gate: true
```

### Q4: 准确率与传统模式差距大

**原因**: 数据量变化（横截面模式样本数不同）

**解决方案**:
- 增加训练epochs
- 调整学习率
- 检查数据预处理

---

## 最佳实践

### 推荐配置

**生产环境**:
```yaml
data:
  use_cross_sectional_training: true
  num_center_nodes: 12
  sampler_type: "degree"
  sampler_temperature: 1.0

model:
  use_node_gate: true
  gate_hidden_dim: 64
  cross_sectional_mode: true

training:
  num_epochs: 30
  batch_size: 1  # 横截面模式固定为1
  learning_rate: 0.00005
```

**实验对比**:
```yaml
# Baseline 1: 传统训练
data.use_cross_sectional_training: false
model.cross_sectional_mode: false

# Baseline 2: 横截面全图（所有86个行业）
# 修改CrossSectionalLocalDataset使num_centers=86

# Baseline 3: 横截面局部 + 固定alpha（无门控）
model.use_node_gate: false
model.embedding_fusion_alpha: 0.7
```

---

## 未来改进方向

1. **动态中心节点数**: 根据时间步动态调整
2. **多尺度采样**: 同时采样不同粒度的子图
3. **注意力机制采样**: 基于注意力权重选择中心节点
4. **时序一致性**: 跨时间步的嵌入正则化

---

**文档版本**: v1.0
**最后更新**: 2025-11
**维护者**: MMF-GAT Team
