# MPS内存不足问题分析与解决方案

## 一、问题原因分析

### 1. 错误信息解读

```
RuntimeError: MPS backend out of memory
- MPS allocated: 8.14 GiB
- other allocations: 836.77 MiB  
- max allowed: 9.07 GiB
- Tried to allocate: 253.95 MiB
```

**关键信息**：
- MPS（Metal Performance Shaders，macOS GPU）已分配 8.14 GiB
- 系统限制为 9.07 GiB
- 尝试再分配 253.95 MiB 时失败

### 2. 根本原因：横截面模式下的Batch合并

#### 2.1 Batch Size放大效应

在横截面局部训练模式下：

```python
# train_epoch_cross_sectional 中的合并逻辑
sequences = torch.cat(all_sequences, dim=0)  # [total_active, max_seq_len, features]
```

**计算过程**：
1. **DataLoader batch_size**: 64（配置值）
2. **每个样本的活跃节点数**: `num_centers` (12) + 邻居节点 (约8-15个) ≈ **20-30个节点**
3. **合并后的总活跃节点数**: 64 × 25 = **1600个节点**
4. **输入张量大小**: `[1600, 80, 7]` = **896,000个元素**

#### 2.2 Transformer注意力机制的内存需求

Transformer的multi-head attention需要计算：

```python
# 注意力矩阵大小
attention_matrix = [seq_len, seq_len] = [80, 80]  # 每个头
# 8个头 × batch_size = 8 × 1600 = 12,800个注意力矩阵
# 每个矩阵: 80 × 80 × 4 bytes (float32) = 25.6 KB
# 总内存: 12,800 × 25.6 KB ≈ 327 MB
```

**实际内存需求**：
- 输入张量: `[1600, 80, 7]` × 4 bytes ≈ **3.6 MB**
- 中间激活值: `[1600, 80, 128]` × 4 bytes ≈ **65.5 MB**（每层）
- 注意力矩阵: `[1600, 80, 80]` × 4 bytes × 8 heads ≈ **327 MB**（每层）
- 梯度: 与激活值相同大小 ≈ **65.5 MB**（每层）
- **单层总需求**: ≈ **460 MB**
- **2层Transformer**: ≈ **920 MB**
- **加上GAT和其他层**: ≈ **1.5-2 GB**

#### 2.3 MPS内存限制

- **MPS限制**: 约9 GB（macOS系统限制）
- **已使用**: 8.14 GB
- **尝试分配**: 253.95 MB
- **结果**: 超出限制

### 3. 为什么传统模式没问题？

在传统训练模式下：
- Batch size = 64
- 每个样本 = 1个节点
- **总节点数 = 64**（而不是1600）
- 内存需求减少 **25倍**

## 二、解决方案

### 方案1：减小Batch Size（推荐）

**修改配置文件**：

```yaml
training:
  batch_size: 8  # 从64减小到8
```

**效果**：
- 合并后节点数: 8 × 25 = 200（而不是1600）
- 内存需求减少 **8倍**
- 训练速度可能稍慢，但更稳定

### 方案2：减小中心节点数

**修改配置文件**：

```yaml
data:
  num_center_nodes: 6  # 从12减小到6
```

**效果**：
- 每个样本的活跃节点数: 6 + 邻居 ≈ 12-15个（而不是20-30）
- 合并后节点数: 64 × 12 = 768（而不是1600）
- 内存需求减少 **2倍**

### 方案3：使用梯度累积

**修改trainer.py**：

```python
# 在train_epoch_cross_sectional中添加梯度累积
accumulation_steps = 4  # 累积4个batch再更新
for batch_idx, batch in enumerate(pbar):
    # ... 前向传播 ...
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
```

**效果**：
- 实际batch size = 8 × 4 = 32
- 内存使用 = batch_size=8的内存
- 训练效果 ≈ batch_size=32

### 方案4：使用CPU而不是MPS

**修改配置文件或环境变量**：

```yaml
training:
  device: "cpu"  # 而不是 "auto" 或 "mps"
```

**或者设置环境变量**：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**效果**：
- CPU内存通常更大（16GB+）
- 训练速度较慢，但不会OOM

### 方案5：调整MPS内存限制（不推荐）

**设置环境变量**：

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**警告**：
- 可能导致系统不稳定
- 可能影响其他应用
- 不推荐使用

### 方案6：优化模型架构

**减小模型维度**：

```yaml
model:
  time_encoder:
    d_model: 64  # 从128减小到64
  gat:
    hidden_features: 64  # 从128减小到64
```

**效果**：
- 内存需求减少 **4倍**
- 模型容量降低，可能影响性能

## 三、推荐配置

### 配置A：平衡性能和内存（推荐）

```yaml
training:
  batch_size: 8  # 减小batch size

data:
  num_center_nodes: 8  # 减小中心节点数
```

**预期效果**：
- 合并后节点数: 8 × 15 = 120
- 内存需求: 约1-1.5 GB
- 训练速度: 适中

### 配置B：最小内存使用

```yaml
training:
  batch_size: 4
  use_grad_accumulation: true
  grad_accumulation_steps: 4

data:
  num_center_nodes: 6
```

**预期效果**：
- 合并后节点数: 4 × 12 = 48
- 内存需求: 约500 MB
- 训练速度: 较慢（但通过梯度累积保持效果）

### 配置C：使用CPU

```yaml
training:
  device: "cpu"
  batch_size: 16  # CPU可以支持稍大的batch
  num_workers: 4
```

## 四、代码修改示例

### 修改train.py添加梯度累积支持

```python
# 在trainer.train()调用中添加参数
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    adj_matrix=adj_matrix_tensor,
    num_epochs=config.training.num_epochs,
    save_path=str(save_path),
    use_cross_sectional=use_cross_sectional,
    grad_accumulation_steps=config.training.get('grad_accumulation_steps', 1)  # 新增
)
```

### 修改trainer.py支持梯度累积

```python
def train_epoch_cross_sectional(self, dataloader: DataLoader, adj_matrix: torch.Tensor,
                                epoch: int = 0, grad_accumulation_steps: int = 1):
    # ...
    for batch_idx, batch in enumerate(pbar):
        # ... 前向传播 ...
        loss = loss / grad_accumulation_steps  # 缩放损失
        loss.backward()
        
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
```

## 五、监控内存使用

### 添加内存监控代码

```python
import torch

def print_memory_usage():
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        print(f"MPS Allocated: {allocated:.2f} GB")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"CUDA Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# 在训练循环中调用
for batch_idx, batch in enumerate(pbar):
    print_memory_usage()
    # ... 训练代码 ...
```

## 六、总结

**问题根源**：
- 横截面模式下batch合并导致实际batch size放大25倍
- Transformer注意力机制内存需求与batch size平方相关
- MPS内存限制较严格（约9GB）

**最佳解决方案**：
1. **立即修复**：减小batch_size到8，减小num_center_nodes到8
2. **长期优化**：实现梯度累积，支持更大的有效batch size
3. **备选方案**：使用CPU训练（如果MPS内存不足）

**预期效果**：
- 内存使用: 从8GB+降低到1-2GB
- 训练稳定性: 显著提升
- 训练速度: 可能稍慢，但可通过梯度累积补偿

