# 阶段1&2完成总结 - 横截面局部训练核心组件

## 已完成组件清单

### 1. ✅ 度数采样器 (`components/degree_sampler.py`)
**文件大小**: 249行

**核心类**：
- `DegreeBasedSampler`: 基于度数的概率采样（主要使用）
- `SequentialSampler`: 顺序采样（baseline对比）

**关键功能**：
```python
# 基于度数采样中心节点
sampler = DegreeBasedSampler(adj_matrix, temperature=1.0)
center_nodes = sampler.sample(k=12)  # 采样12个中心
neighbors = sampler.get_neighbors(center_nodes)  # 获取1-hop邻居
stats = sampler.get_sampling_statistics()  # 采样统计
```

**参数说明**：
- `temperature`: 控制采样平滑度
  - 1.0: 严格按度数
  - >1.0: 更均匀
  - <1.0: 更陡峭（强化度数差异）

---

### 2. ✅ 节点级门控层 (`components/node_level_gate.py`)
**文件大小**: 202行

**核心类**：
- `NodeLevelGate`: 节点级自适应门控（主要使用）
- `GlobalGate`: 全局门控（所有节点共享）
- `AdaptiveGate`: 多头门控

**功能**：
```python
gate = NodeLevelGate(feature_dim=64, hidden_dim=64)
fused, gates = gate(time_features, embeddings)
# fused: [num_nodes, 64] 融合特征
# gates: [num_nodes, 1] 门控值 ∈ [0, 1]
```

**融合公式**：
```
output = g * time_features + (1-g) * embeddings
其中 g = sigmoid(MLP([time_features; embeddings]))
```

---

### 3. ✅ 模型主体修改 (`components/model.py`)
**修改量**: +120行

**新增参数**：
```python
IndustryStockModel(
    ...,
    use_node_gate=True,  # 节点级门控
    gate_hidden_dim=64,  # 门控MLP维度
    cross_sectional_mode=True  # 横截面模式
)
```

**新增方法**：
- `_process_cross_sectional_subgraph()`: 完整86节点图处理
  - 输入：部分节点有时间特征
  - 处理：节点级门控融合
  - 输出：所有节点的GAT特征

**关键流程**：
```python
# 1. 初始化86个节点的嵌入
all_embeddings = self.industry_embeddings(torch.arange(86))

# 2. 填充有输入节点的时间特征
all_time_features[active_indices] = compressed_features

# 3. 节点级门控融合
fused_features, gates = self.node_gate(
    all_time_features[active_indices],
    all_embeddings[active_indices]
)

# 4. 掩码节点使用纯嵌入
final_features[masked_indices] = all_embeddings[masked_indices]

# 5. 完整图GAT
gat_output = self.gat(final_features, adj_matrix)  # [86, gat_output_dim]

# 6. 提取batch输出
batch_output = gat_output[active_indices]
```

---

### 4. ✅ 横截面局部数据集 (`components/cross_sectional_dataset.py`)
**文件大小**: 258行

**核心功能**：
```python
dataset = CrossSectionalLocalDataset(
    cross_sectional_data=cross_sec_data,  # 横截面数据
    adj_matrix=adj_matrix,
    num_centers=12,  # 每次采样12个中心
    sampler_type='degree',  # 度数采样
    samples_per_timestep=8  # 每时间步采样8次
)

# 数据组织
# - 外层: 时间步 (900个)
# - 内层: 每步8次采样
# - 总样本: 900 × 8 = 7,200
```

**返回格式**：
```python
sample = dataset[idx]
# {
#     'sequence': [num_active, 80, 7],      # 中心+邻居序列
#     'target': [num_active],                # 标签
#     'mask': [num_active, 80],              # 时间掩码
#     'industry_idx': [num_active],          # 行业索引
#     'node_mask': [86],                     # bool, 哪些节点有输入
#     'center_mask': [num_active],           # bool, 哪些是中心
#     'time_index': scalar                   # 时间步索引
# }
```

---

## 数据流示意图

```
横截面数据 (t=100):
┌─────────────────────────────────────────┐
│ 行业0   行业1   行业2   ...   行业85   │
│ [80,7]  [80,7]  [80,7]  ...  [80,7]    │
│ 同一时间窗口的所有行业数据              │
└─────────────────────────────────────────┘
              ↓
        度数采样器
              ↓
┌──────────────────────────────────────────┐
│ 中心节点: [5, 12, 23, 45, 67, 78, ...]  │ k=12
│ 邻居节点: [1, 3, 8, 10, 15, 20, ...]    │ ~15个
│ 掩码节点: [0, 2, 4, 6, 7, 9, ...]       │ ~60个
└──────────────────────────────────────────┘
              ↓
      节点特征填充
              ↓
┌──────────────────────────────────────────┐
│ 中心+邻居: 门控融合(时间 + 嵌入)       │
│ 掩码节点: 纯嵌入                        │
└──────────────────────────────────────────┘
              ↓
      完整86节点GAT
              ↓
┌──────────────────────────────────────────┐
│ 提取中心+邻居的输出 → 计算损失         │
│ 掩码节点的嵌入通过梯度间接更新         │
└──────────────────────────────────────────┘
```

---

## Epoch定义

**时间步优先模式**：
```
Epoch 1:
  Time Step 0:
    Sample 1: centers=[5,12,23,...] + neighbors
    Sample 2: centers=[8,15,34,...] + neighbors
    ...
    Sample 8: centers=[77,81,85,...] + neighbors
  Time Step 1:
    Sample 9: centers=...
    ...
  ...
  Time Step 899:
    Sample 7193-7200

Epoch 2: (重新开始)
  Time Step 0: (重新采样，可能不同的中心节点)
    ...
```

**统计保证**：
- 每个时间步采样8次
- 每次12个中心 → 96个中心位置
- 86个行业 → 平均每个行业被选为中心 ≈ 1.1次
- 加上作为邻居的机会 → 每个时间步每个行业都被多次使用

---

## 参数统计

### 新增参数量
```python
# 节点级门控 (NodeLevelGate)
gate_params = (64*2) * 64 + 64 * 1 = 8,256 + 64 = 8,320

# 行业嵌入 (已有)
embedding_params = 86 * 64 = 5,504

# 总新增
total_new = 8,320 + 5,504 = 13,824 参数

# 占总模型参数
ratio = 13,824 / 661,608 ≈ 2.1%
```

### 计算复杂度
- 嵌入查找: O(86) - 可忽略
- 门控MLP: O(num_active * 64^2) ≈ O(27 * 4096) - 很小
- GAT (86节点): 与之前相同
- **结论**: 几乎不增加计算开销

---

## 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 采样策略 | 基于度数 | 重要节点(科技、金融)更多训练机会 |
| 门控类型 | 节点级 | 每个节点自适应决定融合比例 |
| 掩码处理 | 纯嵌入 | 不参与损失，但参与GAT消息传递 |
| Epoch定义 | 时间步优先 | 确保时间维度完整遍历 |
| 中心数量 | 12 | 平衡效率(计算)和覆盖率(信息) |

---

## 下一步

### 阶段3: 训练流程 (进行中)
- [ ] 修改 `trainer.py` 支持横截面训练
- [ ] 时间步计数器
- [ ] 门控值记录和分析

### 阶段4: 配置和文档
- [ ] 更新 `config/default_config.yaml`
- [ ] 修改 `train.py`
- [ ] 创建技术文档

---

**当前状态**: 阶段1&2 完成 ✅
**下一任务**: 修改训练器 (`trainer.py`)
**预计剩余工作量**: 2-3小时
