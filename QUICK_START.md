# MMF-GAT 快速入门指南

## 🚀 5分钟快速开始

### 第一步: 验证环境

```bash
python verify_setup.py
```

**预期输出**: 所有检查项显示 ✓

---

### 第二步: 开始训练

```bash
# 使用默认配置(推荐)
python train.py
```

**训练配置**:
- Batch size: 64
- Epochs: 1 (可在config中修改)
- Learning rate: 5e-5
- K-fold: 3折验证
- 行业嵌入: 启用 ✅

**输出文件**:
- `checkpoints/fold_X_best_model.pth` - 每折的最佳模型
- `visualizations/*.png` - 可视化图表

---

### 第三步: 查看结果

训练完成后,检查以下可视化:

1. **训练曲线** (`visualizations/kfold_results.png`)
   - 每折的损失和准确率
   - 验证性能对比

2. **行业嵌入相似度** (`visualizations/industry_embedding_similarity.png`)
   - 查看哪些行业被模型认为相似
   - 验证是否符合领域知识

3. **子图结构示例** (`visualizations/subgraph_structure_example.png`)
   - 理解GAT如何聚合邻居信息
   - 红色=中心节点, 蓝色=邻居节点

---

## 🔧 常用命令

### 调整训练参数

```bash
# 增加训练轮数
python train.py --num_epochs 50

# 调整batch size
python train.py --batch_size 128

# 修改学习率
python train.py --learning_rate 0.0001

# 组合多个参数
python train.py --num_epochs 30 --batch_size 128 --learning_rate 0.0001
```

### 行业嵌入实验

```bash
# 默认模式: 完全使用时间特征(中心节点)
python train.py --embedding_fusion_alpha 1.0

# 融合模式: 70%时间特征 + 30%嵌入
python train.py --embedding_fusion_alpha 0.7

# 基线对比: 禁用行业嵌入(邻居用零向量)
python train.py --use_industry_embedding False
```

### 使用标准分割(非K折)

```bash
# 修改config/default_config.yaml:
data:
  use_kfold: false
  train_ratio: 0.8
  val_ratio: 0.2

# 然后运行
python train.py
```

---

## 📊 理解输出

### 训练日志示例

```
============================================================
MMF-GAT Industry Stock Prediction Training
============================================================
Random seed set to: 42
Using device: mps

============================================================
Step 1: Loading Data
============================================================
Total samples: 12450
Number of industries: 86
Label distribution: [2490 2490 2490 2490 2490]

============================================================
Step 2: Creating Model
============================================================
Total parameters: 661,608

============================================================
Step 4: Training
============================================================
Using 3-Fold Cross-Validation

Fold 1/3
--------
Train samples: 6225, Val samples: 6225
Epoch 1/1: 100%|████████| 98/98 [00:45<00:00]
  Train Loss: 1.5234, Acc: 28.3%
  Val Loss: 1.4876, Acc: 31.2%
  Val IC: 0.0234, RankIC: 0.0187

...
```

### 关键指标解读

- **Accuracy**: 分类准确率(随机基线=20%)
- **IC**: 信息系数,越高越好(>0.02较好)
- **RankIC**: 秩信息系数,衡量排序能力
- **Sharpe Ratio**: 夏普比率(>1较好,>2优秀)

---

## 🎯 实验建议

### 新手推荐流程

1. **运行默认配置** (1分钟)
   ```bash
   python train.py
   ```
   了解模型基本性能

2. **查看可视化结果** (5分钟)
   - 打开`visualizations/`目录
   - 检查嵌入相似度是否合理
   - 理解子图结构

3. **对比实验** (10分钟)
   ```bash
   # 实验1: 有嵌入(默认)
   python train.py --use_industry_embedding True

   # 实验2: 无嵌入(基线)
   python train.py --use_industry_embedding False
   ```
   比较两者的IC和准确率

4. **调优参数** (30分钟+)
   - 增加epochs到30-50
   - 调整batch_size(32/64/128)
   - 尝试不同学习率(1e-5到1e-4)

---

## 🛠️ 故障排查

### 常见问题

**Q1: 训练速度很慢**
```bash
# 解决方法:
# 1. 减小batch_size
python train.py --batch_size 32

# 2. 减少epochs
python train.py --num_epochs 1

# 3. 禁用K折验证
# 修改config: use_kfold: false
```

**Q2: 准确率停留在20%左右**
- 这是正常的,因为只训练1个epoch
- 增加训练轮数:
  ```bash
  python train.py --num_epochs 30
  ```

**Q3: 内存不足**
```bash
# 减小batch size
python train.py --batch_size 16

# 或减少GAT层数(修改config)
model:
  gat:
    num_layers: 1
```

**Q4: 可视化文件没生成**
- 检查`config/default_config.yaml`:
  ```yaml
  visualization:
    plot_training_curves: true
    plot_confusion_matrix: true
  ```

**Q5: 找不到CUDA/GPU**
```bash
# 系统会自动选择可用设备
# 查看当前设备: 训练开始时会显示
# "Using device: cuda" 或 "mps" 或 "cpu"

# 强制使用CPU
python train.py --device cpu
```

---

## 📁 文件说明

### 配置文件
- `config/default_config.yaml` - 所有超参数

### 输出文件
- `checkpoints/` - 训练好的模型
- `visualizations/` - 可视化图表
- `logs/` - 训练日志(如果启用)

### 数据文件
- `data/industry_kline_data_cleaned.json` - K线数据
- `data/industry_relation_cleaned.csv` - 行业关系图
- `data/industry_list.json` - 86个行业名称

---

## 🔍 进阶功能

### 推理预测

```bash
# 使用训练好的模型进行预测
python predict.py --checkpoint checkpoints/fold_1_best_model.pth
```

### 自定义配置文件

```bash
# 1. 复制默认配置
cp config/default_config.yaml config/my_config.yaml

# 2. 修改参数
# 编辑my_config.yaml...

# 3. 使用自定义配置
python train.py --config config/my_config.yaml
```

### 提取注意力权重

```python
# 在train.py中添加:
from components.visualizer import Visualizer

# 训练后提取注意力
# (需要修改GAT forward方法返回attention_weights)
vis.plot_subgraph_attention_summary(
    attention_weights=...,
    subgraph_nodes=...,
    batch_nodes=...,
    industry_names=...
)
```

---

## 📚 详细文档

- 完整实现说明: `IMPLEMENTATION_COMPLETE.md`
- 子图改进细节: `SUBGRAPH_IMPROVEMENTS.md`
- 模型输入格式: `MODEL_INPUT_EXPLANATION.md`
- 多尺度编码器: `MULTI_SCALE_ENCODER_EXPLANATION.md`

---

## ✅ 检查清单

开始训练前确认:

- [ ] 运行`python verify_setup.py`全部通过
- [ ] 数据文件存在(86个行业)
- [ ] `config/default_config.yaml`配置合理
- [ ] 磁盘空间充足(用于保存模型和可视化)
- [ ] 理解输出目录结构

---

**准备好了吗? 开始训练!**

```bash
python train.py
```

**祝好运!** 🍀
