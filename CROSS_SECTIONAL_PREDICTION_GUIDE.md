# 横截面预测与回测功能使用指南

## 一、功能概述

新增了横截面数据模式和回测功能，用于：
1. **横截面预测**：每个时间步包含所有86个行业的数据，生成行业排名
2. **回测策略**：选择预测为前20%（类别4）的行业持有，计算收益率
3. **可视化**：绘制累计收益率折线图

## 二、横截面数据模式

### 1. 数据格式

在 `data_loader.py` 中新增 `prepare_cross_sectional_data` 方法：

```python
cross_sectional_batches = data_loader_obj.prepare_cross_sectional_data(
    window_sizes=[20, 40, 80],
    future_days=30
)
```

**返回格式**：
- 列表，每个元素是一个时间步的数据
- 每个时间步包含：
  - `sequences`: [num_industries, 80, 7] - 所有行业的序列数据
  - `targets`: [num_industries] - 真实标签
  - `masks`: [num_industries, 80] - 掩码
  - `industry_indices`: [num_industries] - 行业索引（0-85）
  - `time_index`: 时间步索引

### 2. 特点

- ✅ **每个时间步包含所有行业**：天然避免了同一行业多个样本的问题
- ✅ **时间对齐**：所有行业使用相同的时间窗口
- ✅ **适合回测**：可以模拟真实交易场景

## 三、使用方法

### 1. 基本预测

```bash
python predict.py \
    --config config/default_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --mode cross_sectional \
    --output predictions/cross_sectional_predictions.csv
```

### 2. 预测 + 回测

```bash
python predict.py \
    --config config/default_config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --mode cross_sectional \
    --backtest \
    --output predictions/cross_sectional_predictions.csv
```

### 3. 参数说明

- `--mode`: 预测模式
  - `standard`: 标准模式（原有逻辑）
  - `cross_sectional`: 横截面模式（新增）
- `--backtest`: 是否运行回测策略
- `--output`: 输出文件路径

## 四、输出文件

### 1. 预测结果 (`predictions.csv`)

包含列：
- `time_index`: 时间步索引
- `industry_idx`: 行业索引
- `industry_name`: 行业名称
- `predicted_class`: 预测类别（0-4）
- `true_class`: 真实类别（0-4）
- `confidence`: 预测置信度
- `Q1_prob` ~ `Q5_prob`: 各类别概率
- `real_return`: 真实收益率

### 2. 回测结果 (`backtest_results.csv`)

包含列：
- `time_index`: 时间步索引
- `num_selected`: 选中的行业数
- `period_return`: 该期收益率
- `cumulative_return`: 累计收益率
- `selected_industries`: 选中的行业名称（前5个）

### 3. 可视化图表 (`backtest_results.png`)

- 累计收益率折线图
- 包含统计信息（最终收益率、平均收益率等）

## 五、回测策略说明

### 1. 策略逻辑

1. **每个时间步**：
   - 对所有86个行业进行预测
   - 选择预测类别为4（前20%）的行业
   - 等权重持有这些行业

2. **收益率计算**：
   - 使用真实收益率（`real_return`）
   - 等权重平均：`period_return = mean(selected_industries.real_return)`
   - 累计收益率：`cumulative_return *= (1 + period_return)`

3. **可视化**：
   - 绘制累计收益率曲线
   - 显示统计信息

### 2. 示例输出

```
回测统计
============================================================
总时间步数: 850
最终累计收益率: 15.23%
平均每期收益率: 0.0179%
收益率标准差: 0.0234%
平均选中行业数: 17.2
```

## 六、代码示例

### Python代码中使用

```python
from components.data_loader import IndustryDataLoader
from components.model import IndustryStockModel
from predict import predict_cross_sectional, backtest_strategy, plot_backtest_results
import torch

# 1. 加载数据和模型
data_loader_obj = IndustryDataLoader(data_dir="./data", ...)
model = load_model("checkpoints/best_model.pth", config, device)

# 2. 准备横截面数据
cross_sectional_batches = data_loader_obj.prepare_cross_sectional_data()

# 3. 预测
predictions_df = predict_cross_sectional(
    model, cross_sectional_batches, adj_matrix, device, industry_names
)

# 4. 回测
backtest_df = backtest_strategy(
    predictions_df, data_loader_obj, industry_names, top_percentile=4
)

# 5. 可视化
plot_backtest_results(backtest_df, "predictions/backtest_results.png")
```

## 七、注意事项

1. **数据对齐**：
   - 横截面模式要求所有行业有相同的时间范围
   - 如果某个行业数据不足，会在该时间步跳过

2. **内存使用**：
   - 横截面模式会生成大量数据
   - 建议分批处理或使用较小的数据集

3. **回测假设**：
   - 等权重持有
   - 不考虑交易成本
   - 不考虑滑点

4. **模型修复**：
   - 当前模型在batch中有同一行业多个样本时会有问题
   - 建议先修复 `_process_subgraph` 方法（见 `TRAINING_BATCH_ANALYSIS.md`）

## 八、与标准模式的区别

| 方面 | 标准模式 | 横截面模式 |
|------|---------|-----------|
| **数据组织** | 按行业顺序，滑动窗口 | 按时间步，每个时间步包含所有行业 |
| **Batch组成** | 可能包含同一行业的多个样本 | 每个时间步一个batch，包含所有行业 |
| **适用场景** | 训练、一般预测 | 回测、行业排名 |
| **GAT节点** | 可能共享（有问题） | 每个样本独立（无问题） |

## 九、未来改进

1. **支持自定义策略**：
   - 不等权重持有
   - 考虑交易成本
   - 动态调整持仓

2. **更多回测指标**：
   - 夏普比率
   - 最大回撤
   - 胜率

3. **可视化增强**：
   - 多策略对比
   - 行业持仓热力图
   - 收益率分布图

