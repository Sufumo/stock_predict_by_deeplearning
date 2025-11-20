# 训练问题检查清单与Checkpoint机制

## 一、已实现的Checkpoint机制

### 1. 每个Epoch自动保存

**功能**：
- ✅ 每个epoch结束后自动保存checkpoint
- ✅ 保存路径：`checkpoints/epoch_{epoch_num}.pth`
- ✅ 包含完整训练状态：模型、优化器、调度器、训练历史

**保存内容**：
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_metrics['loss'],
    'train_accuracy': train_metrics['accuracy'],
    'val_loss': val_metrics['loss'],
    'val_accuracy': val_metrics['accuracy'],
    'train_history': train_history,
    'val_history': val_history,
    'best_val_acc': best_val_acc,
    # 金融指标（如果存在）
    'val_IC': ...,
    'val_RankIC': ...,
    # 门控统计（如果存在）
    'gate_mean': ...,
    'gate_std': ...
}
```

**自动清理**：
- 只保留最近5个epoch的checkpoint
- 自动删除旧的checkpoint以节省空间

### 2. 最佳模型保存

**功能**：
- ✅ 当验证准确率提升时保存最佳模型
- ✅ 保存路径：`checkpoints/best_model.pth`
- ✅ 包含最佳性能指标

### 3. 错误恢复Checkpoint

**功能**：
- ✅ 训练出错时自动保存错误checkpoint
- ✅ 验证出错时自动保存错误checkpoint
- ✅ 保存路径：`checkpoints/epoch_{epoch_num}_error.pth` 或 `epoch_{epoch_num}_val_error.pth`

## 二、训练时可能遇到的问题

### 1. 内存不足（MPS/CUDA OOM）

**症状**：
```
RuntimeError: MPS backend out of memory
或
RuntimeError: CUDA out of memory
```

**原因**：
- Batch size太大
- 横截面模式下batch合并导致内存放大
- 模型参数太多

**解决方案**：
- ✅ 已添加异常处理，自动清理GPU缓存
- ✅ 建议减小batch_size到4-8
- ✅ 建议减小num_center_nodes到6-8
- ✅ 使用CPU训练（如果GPU内存不足）

**预防措施**：
- 监控内存使用
- 使用梯度累积代替大batch size

### 2. NaN/Inf损失值

**症状**：
```
⚠ Warning: NaN/Inf detected in training loss
```

**原因**：
- 学习率过高
- 梯度爆炸
- 数值不稳定
- 输入数据异常

**解决方案**：
- ✅ 已添加NaN检测和自动checkpoint保存
- ✅ 降低学习率（当前：5e-5）
- ✅ 启用梯度裁剪（max_grad_norm=1.0）
- ✅ 检查输入数据是否包含异常值

**预防措施**：
- 使用梯度裁剪
- 使用学习率调度器
- 检查数据预处理

### 3. 验证准确率异常低

**症状**：
```
⚠ Warning: Very low validation accuracy: 5.0%
```

**原因**：
- 学习率过高导致训练不稳定
- 过拟合
- 模型容量不足
- 数据标签问题

**解决方案**：
- ✅ 已添加低准确率警告
- ✅ 降低学习率
- ✅ 增加正则化（weight_decay）
- ✅ 检查数据标签分布

### 4. 梯度爆炸

**症状**：
```
⚠ Warning: Very large gradient norm: 100.0
```

**原因**：
- 学习率过高
- 网络深度太深
- 激活函数选择不当

**解决方案**：
- ✅ 已添加梯度裁剪（max_grad_norm=1.0）
- ✅ 已添加梯度异常检测
- ✅ 降低学习率
- ✅ 使用梯度裁剪

### 5. Batch格式错误

**症状**：
```
AttributeError: 'list' object has no attribute 'to'
```

**原因**：
- 横截面模式下batch是列表格式
- validate方法未正确处理

**解决方案**：
- ✅ 已修复：validate方法自动检测batch格式
- ✅ 支持列表和tensor两种格式

### 6. 训练中断

**症状**：
- 程序崩溃
- 系统重启
- 手动中断

**解决方案**：
- ✅ 每个epoch自动保存checkpoint
- ✅ 可以从任意epoch的checkpoint恢复
- ✅ 支持断点续训

## 三、Checkpoint使用指南

### 1. 从Checkpoint恢复训练

```python
# 加载checkpoint
checkpoint = torch.load('checkpoints/epoch_5.pth', weights_only=False)

# 恢复模型
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复调度器（如果存在）
if checkpoint['scheduler_state_dict'] is not None:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# 恢复训练历史
train_history = checkpoint['train_history']
val_history = checkpoint['val_history']

# 继续训练
start_epoch = checkpoint['epoch'] + 1
```

### 2. 检查训练进度

```python
import torch

checkpoint = torch.load('checkpoints/epoch_10.pth', weights_only=False)

print(f"Epoch: {checkpoint['epoch']}")
print(f"Train Loss: {checkpoint['train_loss']:.4f}")
print(f"Train Acc: {checkpoint['train_accuracy']:.2f}%")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Val Acc: {checkpoint['val_accuracy']:.2f}%")
print(f"Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
```

### 3. 分析训练历史

```python
checkpoint = torch.load('checkpoints/epoch_10.pth', weights_only=False)

train_history = checkpoint['train_history']
val_history = checkpoint['val_history']

# 绘制训练曲线
import matplotlib.pyplot as plt

plt.plot(train_history['loss'], label='Train Loss')
plt.plot(val_history['loss'], label='Val Loss')
plt.legend()
plt.show()
```

## 四、配置选项

### 在config/default_config.yaml中：

```yaml
training:
  save_dir: "./checkpoints"
  save_best_only: true  # 是否只保存最佳模型
  save_every_epoch: true  # ⭐ 是否每个epoch都保存checkpoint
```

### 在代码中：

```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    adj_matrix=adj_matrix_tensor,
    num_epochs=config.training.num_epochs,
    save_path=str(save_path),
    use_cross_sectional=use_cross_sectional,
    save_dir=config.training.save_dir,  # checkpoint保存目录
    save_every_epoch=True  # 是否每个epoch保存
)
```

## 五、Checkpoint文件说明

### 文件命名规则：

1. **正常checkpoint**: `epoch_{epoch_num}.pth`
   - 每个epoch的正常保存
   - 包含完整训练状态

2. **最佳模型**: `best_model.pth`
   - 验证准确率最高时的模型
   - 用于推理和部署

3. **错误checkpoint**: `epoch_{epoch_num}_error.pth`
   - 训练出错时保存
   - 包含错误信息

4. **验证错误checkpoint**: `epoch_{epoch_num}_val_error.pth`
   - 验证出错时保存
   - 包含错误信息

5. **NaN警告checkpoint**: `epoch_{epoch_num}_nan.pth`
   - 检测到NaN/Inf时保存
   - 包含警告信息

## 六、最佳实践

### 1. 定期检查Checkpoint

```bash
# 查看checkpoint文件
ls -lh checkpoints/

# 检查最新checkpoint
python -c "
import torch
ckpt = torch.load('checkpoints/epoch_10.pth', weights_only=False)
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Val Acc: {ckpt[\"val_accuracy\"]:.2f}%')
"
```

### 2. 监控训练指标

- 定期检查训练loss是否下降
- 检查验证准确率是否提升
- 注意NaN/Inf警告
- 监控内存使用

### 3. 备份重要Checkpoint

```bash
# 备份最佳模型
cp checkpoints/best_model.pth checkpoints/backup_best_model.pth

# 备份特定epoch
cp checkpoints/epoch_10.pth checkpoints/backup_epoch_10.pth
```

### 4. 清理旧Checkpoint

```bash
# 删除旧的checkpoint（保留最近5个）
ls -t checkpoints/epoch_*.pth | tail -n +6 | xargs rm
```

## 七、故障排除

### 问题1：Checkpoint文件太大

**原因**：保存了完整的训练历史

**解决方案**：
- 修改代码，只保存最近N个epoch的历史
- 或者不保存历史，只保存当前状态

### 问题2：Checkpoint保存失败

**原因**：
- 磁盘空间不足
- 权限问题
- 文件被占用

**解决方案**：
- ✅ 已添加异常处理，不会中断训练
- 检查磁盘空间
- 检查文件权限

### 问题3：从Checkpoint恢复后训练不稳定

**原因**：
- 优化器状态不匹配
- 学习率调度器状态不匹配

**解决方案**：
- ✅ 已保存优化器和调度器状态
- 确保使用相同的配置
- 检查学习率是否正确恢复

## 八、总结

### ✅ 已实现的功能

1. **每个epoch自动保存checkpoint**
2. **最佳模型自动保存**
3. **错误时自动保存checkpoint**
4. **NaN/Inf检测和警告**
5. **梯度异常检测**
6. **内存不足异常处理**
7. **自动清理旧checkpoint**

### 📋 检查清单

训练前：
- [ ] 检查磁盘空间（至少10GB）
- [ ] 检查GPU内存（如果使用GPU）
- [ ] 确认batch_size和num_centers设置合理
- [ ] 确认学习率设置合理

训练中：
- [ ] 监控训练loss是否正常下降
- [ ] 监控验证准确率是否提升
- [ ] 注意NaN/Inf警告
- [ ] 注意内存使用警告

训练后：
- [ ] 检查checkpoint文件是否正常保存
- [ ] 备份最佳模型
- [ ] 清理旧checkpoint（可选）

