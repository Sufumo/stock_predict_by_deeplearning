# 横截面局部训练实施进度

## 已完成（阶段1：核心组件开发）

### ✅ 1. 基于度数的采样器 (`components/degree_sampler.py`)

**功能**：
- `DegreeBasedSampler`: 基于行业关系图度数的概率采样
  - 度数高的行业（科技、金融等）更容易被选为中心节点
  - 支持温度参数调节采样平滑度
  - 提供采样统计功能
- `SequentialSampler`: 顺序采样（作为baseline对比）
- `get_neighbors()`: 获取1-hop邻居

**文件大小**: ~200行

### ✅ 2. 节点级门控层 (`components/node_level_gate.py`)

**功能**：
- `NodeLevelGate`: 为每个节点独立学习融合权重
  - 通过MLP计算门控值 g ∈ [0, 1]
  - 输出 = g * 时间特征 + (1-g) * 嵌入
  - 提供门控值统计分析
- `GlobalGate`: 全局门控（所有节点共享权重）
- `AdaptiveGate`: 自适应门控（多头版本）

**文件大小**: ~180行

### ✅ 3. 模型主体修改 (`components/model.py`)

**新增参数**：
- `use_node_gate`: 是否使用节点级门控
- `gate_hidden_dim`: 门控MLP隐藏维度
- `cross_sectional_mode`: 是否使用横截面局部训练模式

**新增组件**：
- `self.node_gate`: NodeLevelGate实例

**新增方法**：
- `_process_cross_sectional_subgraph()`: 横截面局部模式的GAT处理
  - 构建完整的86节点图
  - 中心+邻居节点：门控融合（时间特征+嵌入）
  - 掩码节点：纯嵌入
  - 在完整图上运行GAT
  - 只提取中心+邻居的输出用于损失计算

**修改方法**：
- `forward()`: 添加`node_mask`参数，返回门控值
- 根据模式选择调用`_process_subgraph`或`_process_cross_sectional_subgraph`

**代码行数**: +100行

---

## 进行中（阶段2：数据处理）

### 🔄 4. 横截面局部数据集类

**待实现**: `components/data_loader.py`
- `CrossSectionalLocalDataset` 类
- 集成度数采样器
- 时间步追踪逻辑
- 返回格式：包含`node_mask`、`center_mask`、`time_index`

---

## 待实施（阶段3-4）

### ⏳ 5. 训练器修改 (`components/trainer.py`)
- 时间步计数器
- 支持横截面batch
- 门控值记录和分析

### ⏳ 6. 训练脚本和配置
- `train.py`: 模式选择逻辑
- `config/default_config.yaml`: 新增配置项

### ⏳ 7. 文档和可视化
- `CROSS_SECTIONAL_LOCAL_TRAINING.md`: 技术文档
- `visualizer.py`: 门控值可视化、采样覆盖率统计

---

## 下一步行动

继续实施阶段2：
1. 创建`CrossSectionalLocalDataset`类
2. 测试数据加载流程
3. 进入阶段3修改训练器

---

**最后更新**: 实施中
**当前阶段**: 阶段1完成，阶段2进行中
