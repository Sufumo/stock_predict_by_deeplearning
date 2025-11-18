# MMF-GAT 项目全流程详解

基于执行日志和代码分析，本文档详细解释整个项目的训练流程、数据集划分、模型架构和训练方法。

## 📊 一、数据集说明

### 1.1 数据来源
- **行业数量**: 86个行业
- **数据格式**: 每个行业的K线数据（开盘、收盘、最高、最低、成交量、成交额、收益率）
- **数据文件**: 
  - `industry_kline_data.json`: 行业K线历史数据
  - `industry_relation.csv`: 行业关系数据（申万行业分类）

### 1.2 样本生成过程

#### 输入特征（7维）
每个时间步的特征向量：
```
[开盘价, 收盘价, 最高价, 最低价, 成交量, 成交额, 收益率]
```

#### 时间窗口提取
对于每个行业的时间序列数据：
- **滑动窗口**: 从时间序列中提取长度为80的时间窗口
- **多尺度提取**: 从80天窗口中提取3个子窗口
  - `x_20`: 最后20天（短期）
  - `x_40`: 最后40天（中期）
  - `x_80`: 全部80天（长期）

#### 标签生成（5分位数分类）
1. **计算未来收益率**: 对于每个80天窗口，计算未来30天的收益率
   ```
   future_return = (未来30天后的收盘价 - 当前收盘价) / 当前收盘价
   ```

2. **全局分位数计算**: 收集所有样本的未来收益率，计算全局分位数阈值
   ```
   quantiles = [20%, 40%, 60%, 80%]
   ```

3. **标签分配**:
   - `Q1 (标签0)`: 收益率 ≤ 20%分位数（最低20%）
   - `Q2 (标签1)`: 20% < 收益率 ≤ 40%分位数
   - `Q3 (标签2)`: 40% < 收益率 ≤ 60%分位数
   - `Q4 (标签3)`: 60% < 收益率 ≤ 80%分位数
   - `Q5 (标签4)`: 收益率 > 80%分位数（最高20%）

#### 最终数据集
- **总样本数**: 315,370个样本
- **标签分布**: 每个类别63,074个样本（完美平衡，因为使用分位数划分）
- **特征维度**: [315370, 80, 7]
  - 315370: 样本数
  - 80: 时间步数（最大窗口）
  - 7: 特征维度

### 1.3 行业关系图（邻接矩阵）

**邻接矩阵构建规则**:
- 如果两个行业属于同一个申万行业分类（`sw_industry`），则连接权重为1
- 每个行业与自己连接（自连接）
- **邻接矩阵大小**: [86, 86]

## 🔀 二、数据集划分

### 2.1 时间序列K折交叉验证（当前使用）

**特点**: 严格保持时间顺序，避免未来信息泄露

#### 划分方式（3折，min_train_size=0.4）

假设总样本数 N = 315,370：

```
总验证集大小 = N × (1 - 0.4) = 315,370 × 0.6 = 189,222
每折验证集大小 = 189,222 / 3 = 63,074
第一个训练集结束位置 = N × 0.4 = 126,148
```

**Fold 1**:
- 训练集: [0, 126,148) → 126,148个样本
- 验证集: [126,148, 189,222) → 63,074个样本

**Fold 2**:
- 训练集: [0, 189,222) → 189,222个样本
- 验证集: [189,222, 252,296) → 63,074个样本

**Fold 3**:
- 训练集: [0, 252,296) → 252,296个样本
- 验证集: [252,296, 315,370) → 63,074个样本

**关键点**:
- ✅ 训练集总是在验证集之前（时间顺序）
- ✅ 每个fold的训练集逐渐增大
- ✅ 验证集不重叠
- ✅ 符合时间序列预测的实际场景

### 2.2 标准训练/验证分割（备选方案）

如果 `use_kfold: false`:
- **训练集**: 80% (252,296个样本)
- **验证集**: 20% (63,074个样本)
- **随机打乱**: 使用随机种子保证可复现

## 🏗️ 三、模型架构（MMF-GAT）

### 3.1 模型总览

```
输入: [batch_size, 80, 7] (80天K线数据)
  ↓
【阶段1: 多尺度时间特征提取】
  ↓
DWT增强 (可选)
  ↓
多尺度Transformer编码器 (共享参数)
  ├─ x_20 → Transformer → feat_20 [batch, 128]
  ├─ x_40 → Transformer → feat_40 [batch, 128]
  └─ x_80 → Transformer → feat_80 [batch, 128]
  ↓
动态注意力门控融合
  ↓
融合特征 [batch, 128]
  ↓
【阶段2: 行业关系建模】
  ↓
学习压缩层 (LCL)
  ↓
压缩特征 [batch, 64]
  ↓
GAT图注意力网络 (86个行业节点)
  ↓
行业增强特征 [batch, 64]
  ↓
分类预测头
  ↓
输出: [batch, 5] (5个分位数的概率分布)
```

### 3.2 模型组件详解

#### 3.2.1 DWT增强（可选）
- **功能**: 使用小波变换（db4）提取多频率特征
- **输入**: [batch, 80, 7]
- **输出**: [batch, 80, 7] (增强后的特征)

#### 3.2.2 多尺度Transformer编码器
- **架构**: 共享参数的Transformer Encoder
- **参数**:
  - `d_model`: 128
  - `nhead`: 8
  - `num_layers`: 2
  - `dim_feedforward`: 512
- **处理**:
  - `x_20` → `feat_20` [batch, 128]
  - `x_40` → `feat_40` [batch, 128]
  - `x_80` → `feat_80` [batch, 128]

#### 3.2.3 动态注意力门控
- **功能**: 自适应融合三个时间尺度的特征
- **机制**: 学习权重分配，动态决定各尺度的重要性
- **输出**: [batch, 128]

#### 3.2.4 学习压缩层（LCL）
- **功能**: 压缩时间特征维度，为GAT准备输入
- **输入**: [batch, 128]
- **输出**: [batch, 64]

#### 3.2.5 GAT图注意力网络
- **功能**: 建模行业间关系，利用行业关联性
- **输入**: 
  - 节点特征: [batch, 64] (压缩后的时间特征)
  - 邻接矩阵: [86, 86] (行业关系图)
  - 行业索引: [batch] (每个样本对应的行业)
- **处理**:
  - 子图采样: 只处理batch中的行业及其1跳邻居
  - 图注意力: 聚合邻居行业的信息
- **输出**: [batch, 64] (行业增强特征)

#### 3.2.6 分类预测头
- **架构**: 2层MLP
  ```
  Linear(64 → 32) → ReLU → Dropout → Linear(32 → 5)
  ```
- **输出**: [batch, 5] (5个分位数的logits)

### 3.3 模型参数统计
- **总参数量**: 655,848个参数
- **主要参数来源**:
  - Transformer编码器
  - GAT层
  - 分类头

## 🚀 四、训练流程

### 4.1 训练方法

#### 优化器
- **类型**: Adam
- **学习率**: 0.0001
- **权重衰减**: 0.0001 (L2正则化)

#### 损失函数
- **类型**: CrossEntropyLoss
- **任务**: 5分类（5分位数）

#### 学习率调度
- **类型**: ReduceLROnPlateau
- **策略**: 当验证损失不再下降时降低学习率
- **参数**:
  - `mode`: 'min'
  - `factor`: 0.5
  - `patience`: 5
  - `min_lr`: 0.00001

#### 梯度裁剪
- **阈值**: 1.0
- **目的**: 防止梯度爆炸

### 4.2 训练步骤（每个Epoch）

#### 训练阶段（train_epoch）

```python
for batch in train_loader:
    # 1. 数据准备
    sequences = batch['sequence']      # [batch_size, 80, 7]
    targets = batch['target']           # [batch_size]
    industry_indices = batch['industry_idx']  # [batch_size]
    
    # 2. 提取多尺度窗口
    x_20 = sequences[:, -20:, :]        # [batch, 20, 7]
    x_40 = sequences[:, -40:, :]        # [batch, 40, 7]
    x_80 = sequences                    # [batch, 80, 7]
    
    # 3. 前向传播
    predictions, _ = model(
        x_20, x_40, x_80,
        mask_20, mask_40, mask_80,
        adj_matrix, industry_indices
    )
    
    # 4. 计算损失
    loss = CrossEntropyLoss(predictions, targets)
    
    # 5. 反向传播
    loss.backward()
    
    # 6. 梯度裁剪
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 7. 更新参数
    optimizer.step()
    
    # 8. 统计指标
    accuracy = (predictions.argmax(dim=1) == targets).mean()
```

#### 验证阶段（validate）

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        # 前向传播（不计算梯度）
        predictions, _ = model(...)
        
        # 计算损失和准确率
        loss = criterion(predictions, targets)
        accuracy = ...
        
        # 收集预测和真实值（用于金融指标）
        all_predictions.append(predictions)
        all_targets.append(targets)
```

### 4.3 K折交叉验证流程

```python
for fold in range(1, 4):  # 3折
    # 1. 划分训练集和验证集（按时间顺序）
    train_indices, val_indices = tscv.split(indices)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    # 2. 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # 3. 重新初始化模型（每折重新训练）
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.0001)
    
    # 4. 训练当前折
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, adj_matrix)
        val_metrics = trainer.validate(val_loader, adj_matrix)
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            save_checkpoint(...)
    
    # 5. 加载最佳模型，进行最终评估
    model.load_state_dict(best_checkpoint)
    final_metrics = trainer.validate(val_loader, adj_matrix)
    
    # 6. 记录结果
    fold_results['val_acc'].append(final_metrics['accuracy'])
    fold_results['val_IC'].append(final_metrics['IC'])
```

### 4.4 评估指标

#### 分类指标
- **准确率（Accuracy）**: 预测正确的样本比例
- **混淆矩阵**: 5×5矩阵，显示各类别的预测分布

#### 金融指标
- **IC (Information Coefficient)**: 预测值与真实收益率的相关系数
  ```
  IC = corr(predicted_scores, actual_returns)
  ```
- **RankIC**: 预测排名与收益率排名的相关系数
  ```
  RankIC = corr(rank(predicted_scores), rank(actual_returns))
  ```
- **多空组合收益**: 做多高预测值组合，做空低预测值组合的收益

## 📈 五、执行日志分析

### 5.1 数据加载阶段
```
加载了 86 个行业的数据
Total samples: 315,370
Number of industries: 86
Label distribution: [63074 63074 63074 63074 63074]
```
✅ 数据加载成功，标签分布完美平衡

### 5.2 模型创建阶段
```
Total parameters: 655,848
```
✅ 模型创建成功，参数量合理

### 5.3 训练阶段（K折验证）

#### Fold 1
```
Train samples: 126,148
Val samples: 63,074
Epoch 1/1
  Train Loss: nan, Acc: 20.90%
  Val Loss: nan, Acc: 19.42%
```

#### Fold 2
```
Train samples: 189,222
Val samples: 63,074
Epoch 1/1 (训练中...)
```

### 5.4 ⚠️ 发现的问题

**Loss = NaN**: 这是一个严重问题，可能的原因：
1. **学习率过大**: 导致梯度爆炸
2. **数据异常**: 包含NaN或Inf值
3. **数值不稳定**: 某些操作导致数值溢出
4. **初始化问题**: 模型参数初始化不当

**建议修复**:
- 检查数据预处理，确保没有NaN值
- 降低学习率（如0.00001）
- 添加梯度裁剪（已配置，但可能需要调整）
- 检查模型初始化

## 🔄 六、完整数据流

### 6.1 数据流向

```
原始K线数据 (JSON)
  ↓
IndustryDataLoader.parse_kline_data()
  ↓
特征提取 [时间步, 7]
  ↓
滑动窗口提取 (80天窗口)
  ↓
计算未来30天收益率
  ↓
全局分位数计算
  ↓
标签分配 (0-4)
  ↓
IndustryDataset
  ↓
DataLoader (batch_size=32)
  ↓
模型前向传播
  ↓
预测结果 [batch, 5]
  ↓
损失计算 + 反向传播
  ↓
参数更新
```

### 6.2 模型前向传播详细流程

```
输入: x_20 [32, 20, 7], x_40 [32, 40, 7], x_80 [32, 80, 7]
  ↓
DWT增强 (可选)
  ↓
MultiScaleTimeEncoder
  ├─ Transformer(x_20) → feat_20 [32, 128]
  ├─ Transformer(x_40) → feat_40 [32, 128]
  └─ Transformer(x_80) → feat_80 [32, 128]
  ↓
DynamicAttentionGate
  ↓
time_features [32, 128]
  ↓
LearningCompressionLayer
  ↓
compressed_features [32, 64]
  ↓
GAT (子图采样)
  ├─ 提取batch中的行业索引
  ├─ 构建子图（行业+1跳邻居）
  ├─ 图注意力聚合
  └─ batch_gat_features [32, 64]
  ↓
Predictor (MLP)
  ↓
predictions [32, 5]
```

## 📝 七、关键配置参数

### 7.1 数据配置
```yaml
window_20: 20      # 短期窗口
window_40: 40      # 中期窗口
window_80: 80      # 长期窗口
future_days: 30    # 预测未来天数
num_classes: 5     # 5分位数分类
use_kfold: true    # 使用K折验证
n_splits: 3        # 3折
```

### 7.2 模型配置
```yaml
input_features: 7           # K线特征数
time_encoder_dim: 128       # 时间编码器维度
compression_dim: 64         # 压缩后维度
gat_hidden_dim: 128         # GAT隐藏层
gat_output_dim: 64          # GAT输出维度
num_heads: 8                # 注意力头数
num_gat_layers: 2           # GAT层数
use_dwt: true               # 使用DWT增强
```

### 7.3 训练配置
```yaml
batch_size: 32              # 批大小
num_epochs: 50              # 训练轮数（K折中每折的轮数）
learning_rate: 0.0001       # 学习率
weight_decay: 0.0001        # L2正则化
max_grad_norm: 1.0          # 梯度裁剪阈值
```

## 🎯 八、总结

### 8.1 项目特点
1. **多尺度时间特征**: 同时使用20/40/80天窗口捕获不同时间周期
2. **行业关系建模**: 通过GAT利用行业间的关联性
3. **时间序列验证**: 使用K折交叉验证，严格保持时间顺序
4. **金融指标评估**: 不仅关注分类准确率，还关注IC、RankIC等金融指标

### 8.2 训练集、验证集、预测集

- **训练集**: 每个fold中时间上早于验证集的样本
  - Fold 1: 126,148个样本
  - Fold 2: 189,222个样本
  - Fold 3: 252,296个样本

- **验证集**: 每个fold中用于评估模型性能的样本
  - 每个fold: 63,074个样本
  - 用于选择最佳模型和早停

- **预测集**: 在实际应用中，预测集是未来的数据
  - 当前项目中，验证集可以视为"预测集"
  - 用于评估模型的泛化能力

### 8.3 训练方法
- **方法**: 时间序列K折交叉验证
- **优化**: Adam优化器 + 学习率调度
- **正则化**: L2权重衰减 + Dropout + 梯度裁剪
- **评估**: 分类准确率 + 金融指标（IC、RankIC）

---

**注意**: 当前训练中出现 `loss=nan` 的问题，需要检查数据预处理和模型初始化，确保训练稳定进行。

