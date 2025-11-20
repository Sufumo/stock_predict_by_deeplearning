# 训练时Batch数据格式分析

## 一、训练时的数据组织方式

### 1. DataLoader配置

在 `train.py` 中：

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,  # 默认64
    shuffle=True,  # 训练集打乱
    num_workers=config.training.num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size,  # 默认64
    shuffle=False,  # 验证集不打乱
    num_workers=config.training.num_workers
)
```

### 2. 数据采样方式

在 `data_loader.py` 的 `prepare_sequences` 方法中：

```python
for industry_idx, industry_name in enumerate(self.industry_list):
    data = self.parse_kline_data(industry_name)
    # ...
    for i in range(len(data) - max_window - future_days + 1):
        # 滑动窗口采样
        seq = data[i:i+max_window]
        all_sequences.append(seq_normalized)
        all_targets.append(target)
        all_industry_indices.append(industry_idx)  # 记录行业索引
```

**采样特点**：
- 每个行业按时间顺序生成多个样本（滑动窗口）
- 样本按行业顺序排列：行业0的所有样本 → 行业1的所有样本 → ...
- 如果某个行业有1000个时间步，会生成约920个样本（1000 - 80 - 30 + 1）

### 3. Batch组成示例

假设有3个行业，每个行业有100个样本：

```
数据集顺序：
[行业0样本0, 行业0样本1, ..., 行业0样本99,
 行业1样本0, 行业1样本1, ..., 行业1样本99,
 行业2样本0, 行业2样本1, ..., 行业2样本99]

如果 shuffle=True（训练集）：
- Batch可能包含：行业1样本50, 行业2样本30, 行业0样本10, ...
- 同一batch中可能包含多个不同行业的样本

如果 shuffle=False（验证集）：
- Batch 1: [行业0样本0-63]  ← 64个样本，全部是行业0！
- Batch 2: [行业0样本64-99, 行业1样本0-36]  ← 混合
- Batch 3: [行业1样本37-99, 行业2样本0-26]  ← 混合
```

## 二、训练时的问题分析

### ⚠️ 问题1：验证集Batch中可能出现同一行业的多个样本

**场景**：
- 验证集 `shuffle=False`
- Batch size = 64
- 某个行业有超过64个样本

**结果**：
- Batch中可能包含64个相同行业的样本（但时间窗口不同）
- 这些样本在 `_process_subgraph` 中会映射到同一个GAT节点
- **只有最后一个样本的特征被保留，前面的被覆盖**

### ⚠️ 问题2：训练集虽然打乱，但仍可能包含同一行业的多个样本

**场景**：
- 训练集 `shuffle=True`
- 但某个行业的样本数量很多（如1000个）
- Batch size = 64

**结果**：
- 虽然打乱，但仍有概率在同一batch中出现多个相同行业的样本
- 如果出现，同样会有特征覆盖问题

### ⚠️ 问题3：影响分析

1. **信息丢失**：
   - 同一batch中多个相同行业的样本，只有最后一个的特征被使用
   - 前面的样本对梯度更新没有贡献

2. **训练效率低**：
   - 64个样本可能只贡献了10-20个有效样本的信息
   - 梯度估计不准确

3. **验证指标偏差**：
   - 验证集batch中如果全是同一行业，所有样本预测相同
   - 验证准确率可能被高估或低估

## 三、解决方案

### 方案1：修复 `_process_subgraph`（推荐）

为每个样本创建独立的GAT节点，而不是按行业聚合：

```python
def _process_subgraph(self, compressed_features: torch.Tensor,
                     industry_indices: torch.Tensor,
                     adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    修复版本：为每个样本创建独立节点
    """
    batch_size = compressed_features.shape[0]
    device = compressed_features.device
    
    # ⭐ 为每个样本创建独立节点
    subgraph_features = compressed_features.clone()  # [batch_size, compression_dim]
    
    # ⭐ 构建邻接矩阵：基于行业关系
    subgraph_adj = torch.zeros(batch_size, batch_size, device=device, dtype=torch.float32)
    for i in range(batch_size):
        industry_i = industry_indices[i].item()
        subgraph_adj[i, i] = 1.0  # 自连接
        for j in range(i + 1, batch_size):
            industry_j = industry_indices[j].item()
            if adj_matrix[industry_i, industry_j] > 0:  # 如果行业有关联
                subgraph_adj[i, j] = 1.0
                subgraph_adj[j, i] = 1.0
    
    # GAT处理
    gat_output = self.gat(subgraph_features, subgraph_adj)  # [batch_size, gat_output_dim]
    
    return gat_output  # 直接返回，每个样本对应一个输出
```

### 方案2：修改DataLoader采样策略

确保每个batch中每个行业最多只有一个样本：

```python
class BalancedIndustryDataset(Dataset):
    """平衡的数据集，确保每个batch中每个行业最多一个样本"""
    def __init__(self, ...):
        # 按行业分组样本
        self.industry_samples = {}
        for idx, industry_idx in enumerate(industry_indices):
            if industry_idx not in self.industry_samples:
                self.industry_samples[industry_idx] = []
            self.industry_samples[industry_idx].append(idx)
    
    def __getitem__(self, idx):
        # 实现平衡采样逻辑
        ...
```

### 方案3：使用横截面模式训练（实验性）

使用 `prepare_cross_sectional_data` 生成的数据：
- 每个batch包含所有86个行业（每个行业一个样本）
- 天然避免了同一行业多个样本的问题
- 但需要修改训练逻辑

## 四、当前状态

### ✅ 已实现

1. **横截面数据模式**：`prepare_cross_sectional_data` 方法
   - 每个时间步包含所有86个行业的数据
   - 用于回测和预测

2. **回测功能**：`backtest_strategy` 函数
   - 选择预测为前20%（类别4）的行业
   - 计算真实收益率
   - 绘制累计收益率折线图

### ⚠️ 待修复

1. **训练时的GAT节点共享问题**：
   - 需要修复 `_process_subgraph` 方法
   - 为每个样本创建独立节点

2. **验证集batch组织**：
   - 当前验证集可能包含同一行业的多个样本
   - 建议修复 `_process_subgraph` 后重新训练

## 五、建议

1. **立即修复**：修改 `components/model.py` 中的 `_process_subgraph` 方法
2. **重新训练**：使用修复后的模型重新训练
3. **验证效果**：对比修复前后的训练和验证指标
4. **横截面回测**：使用横截面模式进行回测，验证模型实际表现

