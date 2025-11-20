# 模型输入维度和K折验证详解

## 📊 一、模型输入维度

### 1.1 批次结构

**错误理解**：`n(批次) × 86(行业) × 80(时间步) × 7(特征)`

**正确理解**：`[batch_size, 80, 7]`

### 1.2 详细说明

#### 数据准备阶段

1. **样本生成**：
   - 从86个行业的K线数据中，使用滑动窗口提取样本
   - 每个样本是一个80天的序列，包含7个特征
   - **总样本数** = 所有行业能生成的窗口数之和（约315,370个）

2. **样本结构**：
   ```python
   # 每个样本的结构
   sample = {
       'sequence': [80, 7],      # 80天 × 7特征
       'target': int,             # 标签（0-4，5分位数）
       'mask': [80],              # 掩码（全1）
       'industry_idx': int        # 行业索引（0-85）
   }
   ```

#### 批次构建阶段

1. **DataLoader的工作**：
   - PyTorch的`DataLoader`将样本打包成批次
   - 每个批次包含`batch_size`个样本（如32或64）
   - 批次中的样本可能来自**不同行业**，也可能来自**同一行业的不同时间窗口**

2. **批次形状**：
   ```python
   batch = {
       'sequence': [batch_size, 80, 7],      # 如 [32, 80, 7]
       'target': [batch_size],                # 如 [32]
       'mask': [batch_size, 80],              # 如 [32, 80]
       'industry_idx': [batch_size]           # 如 [32]，值在0-85之间
   }
   ```

3. **关键点**：
   - ❌ **不是**每个批次都包含所有86个行业
   - ✅ 每个批次包含`batch_size`个样本，每个样本来自某个行业
   - ✅ 86是**行业总数**，用于构建GAT图的邻接矩阵`[86, 86]`
   - ✅ `industry_idx`告诉模型每个样本属于哪个行业，用于GAT子图采样

### 1.3 模型前向传播流程

```python
# 输入批次
sequences = [batch_size, 80, 7]  # 例如 [32, 80, 7]

# 1. 提取多尺度窗口（在trainer.py中）
x_80 = sequences                    # [32, 80, 7] - 全部80天
x_40 = sequences[:, -40:, :]       # [32, 40, 7] - 最后40天
x_20 = sequences[:, -20:, :]       # [32, 20, 7] - 最后20天

# 2. 分别通过DWT增强（如果启用）
x_20_enhanced = dwt(x_20)          # [32, 20, 7]
x_40_enhanced = dwt(x_40)          # [32, 40, 7]
x_80_enhanced = dwt(x_80)          # [32, 80, 7]

# 3. 分别通过Transformer编码（共享参数）
feat_20 = transformer(x_20_enhanced)  # [32, 128]
feat_40 = transformer(x_40_enhanced)  # [32, 128]
feat_80 = transformer(x_80_enhanced)   # [32, 128]

# 4. 动态门控融合
time_features = gate(feat_20, feat_40, feat_80)  # [32, 128]

# 5. 学习压缩层
compressed = lcl(time_features)  # [32, 64]

# 6. GAT处理（使用industry_idx和adj_matrix）
gat_features = gat(compressed, industry_idx, adj_matrix)  # [32, 64]

# 7. 预测
predictions = predictor(gat_features)  # [32, 5]
```

## 🔀 二、K折交叉验证（Fold 1, 2, 3）

### 2.1 时间序列K折验证

**Fold 1, 2, 3确实代表时间上的划分**，这是**时间序列K折交叉验证**（Time Series K-Fold Cross-Validation）。

### 2.2 划分方式

假设总样本数 N = 315,370，`min_train_size = 0.4`，`n_splits = 3`：

```
总验证集大小 = N × (1 - 0.4) = 315,370 × 0.6 = 189,222
每折验证集大小 = 189,222 / 3 = 63,074
第一个训练集结束位置 = N × 0.4 = 126,148
```

**Fold 1**:
- 训练集: 样本索引 [0, 126,148) → 126,148个样本（前40%）
- 验证集: 样本索引 [126,148, 189,222) → 63,074个样本（40%-60%）

**Fold 2**:
- 训练集: 样本索引 [0, 189,222) → 189,222个样本（前60%）
- 验证集: 样本索引 [189,222, 252,296) → 63,074个样本（60%-80%）

**Fold 3**:
- 训练集: 样本索引 [0, 252,296) → 252,296个样本（前80%）
- 验证集: 样本索引 [252,296, 315,370) → 63,074个样本（80%-100%）

### 2.3 为什么这样划分？

1. **时间序列特性**：
   - 金融数据具有时间依赖性
   - 不能随机打乱，必须保持时间顺序
   - 训练集必须在验证集之前（避免未来信息泄露）

2. **模拟真实场景**：
   - Fold 1: 用2021-2023的数据训练，预测2023-2024
   - Fold 2: 用2021-2024的数据训练，预测2024-2025
   - Fold 3: 用2021-2025的数据训练，预测2025-2026
   - 每个fold的训练集逐渐增大，模拟随着时间推移获得更多历史数据

3. **评估模型泛化能力**：
   - 测试模型在不同时间段的性能
   - 检查模型是否过拟合到特定时间段
   - 提供更可靠的性能估计

### 2.4 代码实现

```python
# components/validator.py - TimeSeriesKFold
def split(self, X: np.ndarray):
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    val_size = int(n_samples * (1 - self.min_train_size) / self.n_splits)
    first_train_end = int(n_samples * self.min_train_size)
    
    for fold in range(self.n_splits):
        val_start = first_train_end + fold * val_size
        val_end = val_start + val_size
        
        if fold == self.n_splits - 1:
            val_end = n_samples
        
        train_indices = indices[:val_start]      # 训练集：从开始到验证集开始
        val_indices = indices[val_start:val_end]  # 验证集：连续的时间段
        
        yield train_indices, val_indices
```

## 🔄 三、20/40/80日窗口的处理

### 3.1 窗口提取

从80日的完整序列中提取三个子窗口：

```python
# trainer.py - train_epoch()
x_80 = sequences                    # [batch_size, 80, 7] - 全部80天
x_40 = sequences[:, -40:, :]       # [batch_size, 40, 7] - 最后40天
x_20 = sequences[:, -20:, :]       # [batch_size, 20, 7] - 最后20天
```

### 3.2 多尺度特征提取

1. **DWT增强**（可选）：
   ```python
   x_20_enhanced = dwt_enhancement(x_20)  # [batch_size, 20, 7]
   x_40_enhanced = dwt_enhancement(x_40)  # [batch_size, 40, 7]
   x_80_enhanced = dwt_enhancement(x_80)  # [batch_size, 80, 7]
   ```

2. **Transformer编码**（共享参数）：
   ```python
   # 使用同一个Transformer编码器处理不同长度的序列
   feat_20 = time_encoder(x_20_enhanced)  # [batch_size, 128]
   feat_40 = time_encoder(x_40_enhanced)  # [batch_size, 128]
   feat_80 = time_encoder(x_80_enhanced)  # [batch_size, 128]
   ```

3. **动态门控融合**：
   ```python
   # 自适应融合三个时间尺度的特征
   time_features = dynamic_gate(feat_20, feat_40, feat_80)  # [batch_size, 128]
   ```

### 3.3 为什么使用多尺度？

- **20日窗口**：捕获短期波动和趋势
- **40日窗口**：捕获中期趋势
- **80日窗口**：捕获长期趋势和周期性
- **融合**：结合不同时间尺度的信息，提高预测准确性

## 📈 四、完整数据流示例

### 4.1 数据准备

```
86个行业 × 每个行业约3,667个样本 = 315,370个总样本
每个样本: [80天, 7特征]
```

### 4.2 K折划分（3折）

```
Fold 1:
  Train: 126,148个样本（时间上最早的40%）
  Val:   63,074个样本（接下来的20%）

Fold 2:
  Train: 189,222个样本（时间上最早的60%）
  Val:   63,074个样本（接下来的20%）

Fold 3:
  Train: 252,296个样本（时间上最早的80%）
  Val:   63,074个样本（最后的20%）
```

### 4.3 批次处理

```
每个epoch:
  - 遍历所有训练样本
  - 每32个样本组成一个batch
  - 每个batch: [32, 80, 7]
  - 从80日中提取20/40/80三个窗口
  - 分别处理并融合
  - 通过GAT建模行业关系
  - 输出预测 [32, 5]
```

## 🎯 五、关键要点总结

1. **输入维度**：
   - ✅ 每个批次：`[batch_size, 80, 7]`
   - ❌ 不是：`[batch_size, 86, 80, 7]`
   - 86是行业总数，用于GAT图，不是批次维度

2. **Fold划分**：
   - ✅ Fold 1, 2, 3是**时间上的划分**
   - ✅ 训练集逐渐增大，验证集在时间上晚于训练集
   - ✅ 符合时间序列预测的实际场景

3. **多尺度处理**：
   - ✅ 从80日序列中提取20/40/80三个窗口
   - ✅ 分别通过DWT和Transformer编码
   - ✅ 使用动态门控融合多尺度特征

4. **GAT图**：
   - ✅ 邻接矩阵：`[86, 86]`（所有行业的关系）
   - ✅ 每个样本有`industry_idx`，用于GAT子图采样
   - ✅ 只处理batch中的行业及其邻居，提高效率

## 📝 六、常见误解澄清

### 误解1：每个批次包含所有86个行业
**正确**：每个批次包含`batch_size`个样本，这些样本可能来自不同行业，也可能来自同一行业的不同时间窗口。

### 误解2：Fold是随机划分的
**正确**：Fold是按时间顺序严格划分的，训练集总是在验证集之前，避免未来信息泄露。

### 误解3：20/40/80日是三个独立的输入
**正确**：它们是从同一个80日序列中提取的三个子窗口，共享同一个Transformer编码器，最后融合成一个特征向量。

---

**总结**：模型输入是`[batch_size, 80, 7]`，Fold 1/2/3是时间序列K折验证，代表时间上的划分，训练集逐渐增大以模拟真实场景。


