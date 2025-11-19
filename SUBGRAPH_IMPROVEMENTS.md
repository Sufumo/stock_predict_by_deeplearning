# GAT子图采样改进说明

## 改进概述

本次改进将GAT子图采样中的"邻居节点使用零向量"升级为"邻居节点使用可学习的行业嵌入",并添加了完整的可视化工具来验证改进效果。

## 核心改进

### 1. 可学习的行业嵌入 (Industry Embeddings)

**改进位置**: `components/model.py`

**新增内容**:
```python
# 为每个行业学习一个固定的嵌入向量
self.industry_embeddings = nn.Embedding(num_industries, compression_dim)
nn.init.normal_(self.industry_embeddings.weight, mean=0.0, std=0.01)
```

**作用**:
- 为86个行业各学习一个64维的嵌入向量
- 捕获行业的固有特性(如周期性、波动性、行业属性等)
- 独立于时间序列特征,提供稳定的行业表示

### 2. 改进的子图特征填充

**改进位置**: `components/model.py` - `_process_subgraph` 方法

**原逻辑**:
```python
# 所有节点初始化为零向量
subgraph_features = torch.zeros(num_subgraph_nodes, compression_dim)

# 只有batch中的节点使用时间特征
for i, orig_idx in enumerate(industry_indices):
    subgraph_features[sub_idx] = compressed_features[i]

# 邻居节点保持零向量 ❌
```

**新逻辑**:
```python
# ⭐ 使用行业嵌入初始化所有节点
subgraph_embeddings = self.industry_embeddings(subgraph_nodes_tensor)
subgraph_features = subgraph_embeddings.clone()

# ⭐ Batch节点可选择完全替换或融合
for i, orig_idx in enumerate(industry_indices):
    if embedding_fusion_alpha < 1.0:
        # 融合模式: 时间特征 + 行业嵌入
        subgraph_features[sub_idx] = (
            alpha * compressed_features[i] +
            (1 - alpha) * subgraph_features[sub_idx]
        )
    else:
        # 替换模式: 完全使用时间特征(默认)
        subgraph_features[sub_idx] = compressed_features[i]

# 邻居节点保持行业嵌入 ✅
```

### 3. 子图结构说明

**当前实现(保持不变)**:
```
给定batch: [行业5, 行业12, 行业23, 行业45]

Step 1: 获取batch中的中心节点
  中心节点 = {5, 12, 23, 45}

Step 2: 添加1-hop邻居
  - 行业5的邻居: {1, 3, 7, ...}
  - 行业12的邻居: {10, 13, 15, ...}
  - ...

  子图节点 = 中心节点 ∪ 所有邻居

Step 3: 构建诱导子图
  - 提取子图节点之间的所有边
  - 包含:
    * 中心节点之间的边 (如果存在)
    * 中心节点到邻居的边
    * 邻居之间的边 (如果都在子图中)

Step 4: 特征填充
  - 中心节点: 使用时间序列特征 (来自Transformer编码器)
  - 邻居节点: 使用可学习的行业嵌入 ⭐
```

**边的来源**:
- 所有边都来自 `data/industry_relation_cleaned.csv`
- 基于真实的行业关联关系(如"电池"↔"光伏设备")
- 确保子图中的边都是有意义的

## 可视化工具

### 1. 子图结构可视化

**方法**: `Visualizer.plot_subgraph_structure()`

**功能**:
- 使用networkx绘制子图的网络结构
- 红色节点: batch中心节点
- 蓝色节点: 1-hop邻居节点
- 边: 行业之间的关联关系
- 标签: 行业名称

**用途**: 验证子图采样的合理性

### 2. 行业嵌入相似度

**方法**: `Visualizer.plot_embedding_similarity()`

**功能**:
- 计算87个行业嵌入的余弦相似度
- 绘制相似度热力图
- 只显示top-30最具代表性的行业

**用途**:
- 验证嵌入是否学到了有意义的行业聚类
- 检查相似行业(如"电池"与"新能源车")是否在嵌入空间中接近

### 3. 注意力权重分析

**方法**: `Visualizer.plot_subgraph_attention_summary()`

**功能**:
- 提取GAT的注意力权重
- 左图: 注意力权重热力图
- 右图: 每个节点接收到的平均注意力

**用途**:
- 理解GAT学到的行业关系模式
- 验证中心节点和邻居节点的信息传递

## 配置参数

### config/default_config.yaml

```yaml
model:
  # 行业嵌入配置
  num_industries: 86  # 行业总数
  use_industry_embedding: true  # 是否使用可学习的行业嵌入
  embedding_fusion_alpha: 1.0  # 时间特征融合权重
    # 1.0 = 完全使用时间特征(batch节点)
    # 0.5 = 时间特征和嵌入各占50%
    # 0.0 = 完全使用嵌入
```

**参数说明**:
- `num_industries`: 固定为86(根据数据集)
- `use_industry_embedding`:
  - `true`: 使用行业嵌入(推荐) ✅
  - `false`: 回退到零向量模式
- `embedding_fusion_alpha`:
  - 只影响batch中心节点
  - 邻居节点始终使用纯嵌入
  - 推荐使用1.0(完全替换模式)

## 改进优势

### 1. 解决零特征问题
**之前**: 邻居节点使用零向量,无实际语义信息
**现在**: 邻居节点使用学习到的行业嵌入,有意义的特征表示

### 2. 增强信息传播
**之前**: GAT在聚合邻居信息时,邻居只提供结构信息
**现在**: 邻居既提供结构信息,又提供行业特性信息

### 3. 学习行业特性
- 嵌入可以捕获行业的固有属性
- 例如:周期性行业(房地产、基建)可能形成一个聚类
- 成长型行业(科技、新能源)可能形成另一个聚类

### 4. 提高泛化能力
- 行业嵌入在整个训练过程中共享学习
- 即使某些行业在batch中出现较少,也能学到好的表示
- 减少了数据稀疏性的影响

### 5. 可解释性
- 通过可视化嵌入相似度,可以理解模型学到的行业关系
- 验证模型是否符合领域知识
- 帮助发现潜在的行业关联模式

## 使用方法

### 训练模型
```bash
python train.py --use_industry_embedding
```

### 调整融合权重
```bash
python train.py --embedding_fusion_alpha 0.7
```

### 禁用行业嵌入(回退模式)
修改配置文件:
```yaml
model:
  use_industry_embedding: false
```

## 预期效果

### 模型性能
- 训练损失: 预期更平滑的收敛曲线
- 验证指标: 预期IC和RankIC略有提升
- 稳定性: 减少训练波动

### 可视化结果
1. **子图结构**:
   - 应该看到清晰的中心-邻居结构
   - 行业之间的连接应该符合真实关联

2. **嵌入相似度**:
   - 相似行业(如"化学制品"与"化学原料")应该聚在一起
   - 不同类别行业(如"金融"与"制造")应该分开

3. **注意力权重**:
   - 中心节点应该对相关邻居有较高注意力
   - 邻居对中心节点的贡献应该合理

## 技术细节

### 参数量增加
```python
# 行业嵌入参数
num_params = num_industries * compression_dim
           = 86 * 64
           = 5,504 个参数
```
相比整个模型(约66万参数),增加量很小(不到1%)。

### 计算开销
- 嵌入查找: O(num_subgraph_nodes) - 非常快
- 额外内存: 86 * 64 * 4 bytes = 22KB - 可忽略
- 训练时间: 几乎无影响

### 梯度流动
```
时间序列数据 → Transformer → LCL → 时间特征
                                    ↓
                            (融合/替换)
                                    ↓
行业ID → Embedding → 行业嵌入 → 子图特征 → GAT → 预测
         ↑                                      ↓
         └────────── 反向传播更新 ────────────┘
```

行业嵌入通过GAT的梯度进行更新,学习对预测任务有用的行业表示。

## 对比实验建议

### Baseline (零向量邻居)
```yaml
model:
  use_industry_embedding: false
```

### 改进版本 (嵌入邻居)
```yaml
model:
  use_industry_embedding: true
  embedding_fusion_alpha: 1.0
```

### 对比指标
1. 训练损失曲线
2. 验证IC/RankIC
3. 最终预测准确率
4. 模型收敛速度

## 常见问题

### Q: 为什么邻居节点不直接使用它们的时间序列特征?
A: 因为batch中没有这些邻居行业的数据。每个batch只包含部分行业的时间序列数据,邻居节点不在batch中,所以无法获得它们当前的时间序列特征。行业嵌入提供了一个稳定的替代方案。

### Q: embedding_fusion_alpha应该设置为多少?
A: 推荐使用默认值1.0(完全替换)。这样batch节点使用准确的时间特征,邻居节点使用稳定的嵌入,两者职责分明。如果想探索融合效果,可以尝试0.7-0.9之间的值。

### Q: 如何验证嵌入学得好不好?
A:
1. 查看嵌入相似度可视化,看相似行业是否聚集
2. 观察训练过程中IC/RankIC是否提升
3. 使用t-SNE降维可视化嵌入空间

### Q: 这个改进对小batch_size有帮助吗?
A: 是的!小batch_size时,中心节点少,邻居节点相对更多,行业嵌入的作用更明显。

## 未来改进方向

1. **动态嵌入**: 让嵌入随时间变化(如使用LSTM)
2. **多层嵌入**: 为不同GAT层使用不同的嵌入
3. **外部特征融合**: 将行业的宏观特征(如行业规模、增长率)融入嵌入
4. **预训练嵌入**: 使用行业描述文本预训练嵌入

## 总结

本次改进通过引入可学习的行业嵌入,解决了GAT子图采样中邻居节点零特征的问题,同时保持了扩展诱导子图的结构优势。改进后的模型能够更充分地利用图结构信息,提升预测性能和可解释性。

**核心思想**:
- Batch中心节点 = 精确的时间特征(来自当前数据)
- 邻居节点 = 稳定的行业嵌入(学习行业固有属性)
- 两者结合 = 既有动态信息又有结构信息

---

**实现日期**: 2025-11
**版本**: v1.0
**状态**: ✅ 已完成并集成到main分支
