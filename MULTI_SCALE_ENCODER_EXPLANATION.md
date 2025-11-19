# 多尺度时间编码器机制详解

## 🎯 一、核心设计思想

项目使用**共享参数的Transformer编码器**来处理不同时间窗口（20日、40日、80日）的数据。这种设计的优势：

1. ✅ **参数共享**：三个尺度共用同一套Transformer参数，减少模型参数量
2. ✅ **多尺度信息融合**：同时捕获短期、中期、长期的时间模式
3. ✅ **灵活处理**：Transformer天然支持可变长度序列

## 📊 二、数据准备阶段

### 2.1 从80天数据中提取多尺度窗口

在 `components/trainer.py` 的训练循环中（第165-177行），从完整的80天序列中提取三个不同尺度的窗口：

```python
# 输入：完整的80天序列
sequences = batch['sequence']  # [batch_size, 80, 7]

# 提取三个不同尺度的窗口
x_80 = sequences                    # [batch_size, 80, 7] - 全部80天
x_40 = sequences[:, -40:, :]       # [batch_size, 40, 7] - 最后40天
x_20 = sequences[:, -20:, :]       # [batch_size, 20, 7] - 最后20天

# 对应的掩码
mask_80 = masks                    # [batch_size, 80]
mask_40 = masks[:, -40:]           # [batch_size, 40]
mask_20 = masks[:, -20:]           # [batch_size, 20]
```

### 2.2 窗口关系可视化

```
时间轴: 0    20    40    60    80
        |-----|-----|-----|-----|
x_80:   [=======================]  (全部80天)
x_40:              [============]  (最后40天)
x_20:                      [====]  (最后20天)
```

**关键点**：
- 三个窗口**嵌套关系**：x_20 ⊆ x_40 ⊆ x_80
- 都使用**相同的时间点**（最后20天重叠）
- 但**时间跨度不同**：20天、40天、80天

## 🔄 三、共享Transformer编码器架构

### 3.1 MultiScaleTimeEncoder 结构

在 `components/time_encoder.py` 中，`MultiScaleTimeEncoder` 使用**单个共享编码器**：

```python
class MultiScaleTimeEncoder(nn.Module):
    def __init__(self, ...):
        # ⭐ 只创建一个共享的Transformer编码器
        self.shared_encoder = SharedTransformerEncoder(...)
    
    def forward(self, x_20, x_40, x_80, ...):
        # 使用同一个编码器处理三个不同长度的序列
        encoded_20 = self.shared_encoder(x_20, mask_20)  # [batch_size, 128]
        encoded_40 = self.shared_encoder(x_40, mask_40)  # [batch_size, 128]
        encoded_80 = self.shared_encoder(x_80, mask_80)  # [batch_size, 128]
        
        return encoded_20, encoded_40, encoded_80
```

### 3.2 SharedTransformerEncoder 处理流程

`SharedTransformerEncoder` 的关键在于能够处理**可变长度序列**：

```python
class SharedTransformerEncoder(nn.Module):
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, features]
        # seq_len可以是20、40或80
        
        # 1. 输入投影：将7个特征投影到d_model维度
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 2. LayerNorm归一化
        x = self.input_norm(x)
        
        # 3. 转换为Transformer格式：(seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # 4. ⭐ 位置编码：根据实际序列长度动态选择
        x = self.pos_encoder(x)  # 只使用前seq_len个位置编码
        
        # 5. ⭐ Transformer编码：处理可变长度序列
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        # encoded: [seq_len, batch_size, d_model]
        
        # 6. ⭐ 全局平均池化：将不同长度序列压缩为固定长度向量
        if mask is not None:
            # 掩码平均池化（考虑有效时间步）
            pooled = masked_mean_pooling(encoded, mask)
        else:
            # 简单平均池化
            pooled = encoded.mean(dim=0)
        
        # 返回固定长度特征向量
        return pooled  # [batch_size, d_model]
```

## 🔑 四、Transformer处理可变长度序列的关键机制

### 4.1 位置编码（Positional Encoding）

位置编码支持可变长度序列的核心机制：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # 预计算所有可能位置的位置编码
        pe = torch.zeros(max_len, d_model)
        # ... 计算sin/cos位置编码 ...
        self.register_buffer('pe', pe)  # [max_len, d_model]
    
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        # ⭐ 动态选择：只使用前seq_len个位置编码
        x = x + self.pe[:x.size(0), :]  # 根据实际长度选择
        return x
```

**工作原理**：
- 预计算最大长度（max_len=5000）的位置编码
- 前向传播时，根据实际序列长度动态选择：
  - seq_len=20 → 使用 pe[0:20]
  - seq_len=40 → 使用 pe[0:40]
  - seq_len=80 → 使用 pe[0:80]

### 4.2 自注意力机制（Self-Attention）

Transformer的自注意力机制天然支持可变长度：

```python
# 在TransformerEncoderLayer内部
# Q, K, V的形状都是 [seq_len, batch_size, d_model]

# 对于不同长度的序列：
# seq_len=20: Q, K, V都是 [20, batch_size, d_model]
# seq_len=40: Q, K, V都是 [40, batch_size, d_model]
# seq_len=80: Q, K, V都是 [80, batch_size, d_model]

# 注意力计算：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
# 注意力矩阵形状：[seq_len, seq_len]
# - seq_len=20 → [20, 20]
# - seq_len=40 → [40, 40]
# - seq_len=80 → [80, 80]
```

**关键特性**：
- ✅ 注意力权重矩阵大小随序列长度变化
- ✅ 每个时间步都能关注到序列内的所有时间步
- ✅ 不需要固定长度输入

### 4.3 掩码机制（Padding Mask）

处理不同长度序列时，使用掩码标记有效时间步：

```python
# 创建padding mask
if mask is not None:
    # mask: [batch_size, seq_len], 1表示有效，0表示无效
    src_key_padding_mask = (mask == 0)  # True表示需要mask的位置
    
    # 在注意力计算中，masked位置会被设置为-inf
    # 经过softmax后变为0，不参与计算
```

**作用**：
- 虽然实际序列长度不同，但可以通过mask统一处理
- 确保只有有效时间步参与计算

### 4.4 全局平均池化（Global Average Pooling）

将不同长度的序列压缩为固定长度的特征向量：

```python
# 掩码平均池化
if mask is not None:
    # encoded: [seq_len, batch_size, d_model]
    # mask: [batch_size, seq_len]
    
    # 扩展mask维度
    mask_expanded = mask.unsqueeze(-1).transpose(0, 1)  # [seq_len, batch_size, 1]
    
    # 掩码加权求和
    encoded_masked = encoded * mask_expanded  # [seq_len, batch_size, d_model]
    sum_features = encoded_masked.sum(dim=0)  # [batch_size, d_model]
    sum_mask = mask.sum(dim=1, keepdim=True)   # [batch_size, 1]
    
    # 平均池化
    pooled = sum_features / (sum_mask + 1e-8)  # [batch_size, d_model]
else:
    # 简单平均
    pooled = encoded.mean(dim=0)  # [batch_size, d_model]
```

**结果**：
- 输入：`[seq_len, batch_size, d_model]`（seq_len可以是20/40/80）
- 输出：`[batch_size, d_model]`（固定长度，128维）

## 🔀 五、多尺度特征融合

### 5.1 编码后的特征

经过共享Transformer编码器后，三个不同尺度的序列都被编码为**相同维度的特征向量**：

```python
# 编码结果
feat_20 = shared_encoder(x_20)  # [batch_size, 128] - 20天模式
feat_40 = shared_encoder(x_40)  # [batch_size, 128] - 40天模式
feat_80 = shared_encoder(x_80)  # [batch_size, 128] - 80天模式
```

**特征含义**：
- `feat_20`：捕获短期趋势（最近20天的模式）
- `feat_40`：捕获中期趋势（最近40天的模式）
- `feat_80`：捕获长期趋势（最近80天的模式）

### 5.2 动态注意力门控融合

使用 `DynamicAttentionGate` 动态融合三个尺度的特征：

```python
class DynamicAttentionGate(nn.Module):
    def forward(self, feat_20, feat_40, feat_80):
        # 1. 拼接三个特征
        concat_feat = torch.cat([feat_20, feat_40, feat_80], dim=1)
        # [batch_size, 128*3] = [batch_size, 384]
        
        # 2. 计算动态权重（根据当前特征自适应）
        weights = self.gate_network(concat_feat)  # [batch_size, 3]
        # weights[:, 0]: 20天权重
        # weights[:, 1]: 40天权重
        # weights[:, 2]: 80天权重
        
        # 3. 加权求和
        weighted_sum = (feat_20 * weights[:, 0:1] + 
                       feat_40 * weights[:, 1:2] + 
                       feat_80 * weights[:, 2:3])
        
        # 4. 进一步融合
        fused_feat = self.fusion_layer(concat_feat)
        
        # 5. 残差连接
        output = weighted_sum + fused_feat  # [batch_size, 128]
        
        return output
```

**动态权重的作用**：
- ✅ **自适应融合**：根据当前样本的特征，动态决定三个尺度的权重
- ✅ **场景适应**：不同市场状态可能需要不同尺度的重要性
  - 波动大时 → 短期（20天）权重高
  - 趋势明显时 → 长期（80天）权重高

## 📈 六、完整前向传播流程

### 6.1 模型前向传播（model.py）

```python
def forward(self, x_20, x_40, x_80, ...):
    # 1. DWT增强（可选）
    x_20_enhanced = self.dwt_enhancement(x_20)
    x_40_enhanced = self.dwt_enhancement(x_40)
    x_80_enhanced = self.dwt_enhancement(x_80)
    
    # 2. ⭐ 多尺度Transformer编码（共享参数）
    feat_20, feat_40, feat_80 = self.time_encoder(
        x_20_enhanced, x_40_enhanced, x_80_enhanced,
        mask_20, mask_40, mask_80
    )
    # 输出都是 [batch_size, 128]
    
    # 3. ⭐ 动态门控融合
    time_features = self.dynamic_gate(feat_20, feat_40, feat_80)
    # [batch_size, 128]
    
    # 4. 后续处理（压缩、GAT、预测）
    compressed = self.compression_layer(time_features)
    gat_features = self.gat(compressed, ...)
    predictions = self.predictor(gat_features)
    
    return predictions
```

### 6.2 数据流可视化

```
输入数据:
x_20: [batch_size, 20, 7]  ──┐
x_40: [batch_size, 40, 7]  ──┼──> 共享Transformer编码器
x_80: [batch_size, 80, 7]  ──┘

编码后:
feat_20: [batch_size, 128] ──┐
feat_40: [batch_size, 128] ──┼──> 动态门控融合
feat_80: [batch_size, 128] ──┘

融合后:
time_features: [batch_size, 128] ──> 后续处理
```

## 🎓 七、为什么可以共享参数？

### 7.1 Transformer的特性

1. **位置无关性**：通过位置编码，Transformer可以处理任意位置的序列
2. **长度无关性**：自注意力机制不依赖固定长度
3. **模式学习**：学习的是时间模式，而不是特定长度

### 7.2 参数共享的优势

**优点**：
- ✅ **减少参数量**：三个尺度共用一套参数，参数量减少2/3
- ✅ **正则化效果**：参数共享起到正则化作用，防止过拟合
- ✅ **知识迁移**：不同尺度共享时间模式知识

**潜在问题**：
- ⚠️ 不同尺度可能需要不同的模式，但通过动态门控可以缓解

### 7.3 训练时的梯度更新

```python
# 前向传播
feat_20 = shared_encoder(x_20)
feat_40 = shared_encoder(x_40)
feat_80 = shared_encoder(x_80)

# 融合
time_features = dynamic_gate(feat_20, feat_40, feat_80)

# 损失计算
loss = criterion(predictions, targets)

# 反向传播
loss.backward()
# ⭐ 梯度会同时更新shared_encoder的参数
# 因为三个尺度都使用了shared_encoder
```

**梯度更新**：
- 每个batch中，三个尺度（20、40、80）都会使用shared_encoder
- 反向传播时，梯度会**累加**到shared_encoder的参数上
- 这样shared_encoder学习到的是**多尺度通用的时间模式**

## 📝 八、代码位置总结

| 功能 | 文件位置 | 关键代码 |
|------|---------|---------|
| 多尺度窗口提取 | `components/trainer.py` | `train_epoch()` 第170-172行 |
| 共享编码器定义 | `components/time_encoder.py` | `MultiScaleTimeEncoder` 第140-190行 |
| Transformer编码 | `components/time_encoder.py` | `SharedTransformerEncoder` 第35-137行 |
| 位置编码 | `components/time_encoder.py` | `PositionalEncoding` 第11-32行 |
| 动态融合 | `components/dynamic_gate.py` | `DynamicAttentionGate` 第10-82行 |
| 模型前向传播 | `components/model.py` | `forward()` 第107-170行 |

## 🎯 九、总结

1. **数据准备**：从80天序列中提取20、40、80天的嵌套窗口
2. **共享编码**：使用同一个Transformer编码器处理三个不同长度的序列
3. **可变长度处理**：
   - 位置编码：动态选择前N个位置编码
   - 自注意力：注意力矩阵大小随序列长度变化
   - 全局池化：将不同长度序列压缩为固定长度向量
4. **特征融合**：通过动态注意力门控机制，自适应融合三个尺度的特征
5. **参数共享**：减少参数量，提高泛化能力，通过梯度累加学习多尺度模式

**核心思想**：Transformer的灵活性使得同一个编码器可以处理不同长度的序列，通过全局池化统一输出维度，再通过动态门控融合多尺度信息。

