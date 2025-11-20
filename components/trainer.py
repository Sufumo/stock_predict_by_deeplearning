"""
训练器组件
负责模型训练、验证和评估
支持K折验证、金融指标、学习率调度等
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import os
import shutil
from pathlib import Path

try:
    from .metrics import FinancialMetricsCalculator
    from .validator import TimeSeriesKFold
    from .monitor import NaNDetector, GradientMonitor
except ImportError:
    # 如果是直接运行该文件
    from metrics import FinancialMetricsCalculator
    from validator import TimeSeriesKFold
    from monitor import NaNDetector, GradientMonitor


class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                 use_scheduler: bool = False, scheduler_params: Optional[Dict] = None,
                 compute_financial_metrics: bool = True, max_grad_norm: Optional[float] = None):
        """
        Args:
            model: 要训练的模型
            device: 训练设备（CPU/GPU）
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
            use_scheduler: 是否使用学习率调度器
            scheduler_params: 调度器参数
            compute_financial_metrics: 是否计算金融指标
            max_grad_norm: 梯度裁剪阈值
        """
        self.model = model
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if use_scheduler:
            if scheduler_params is None:
                scheduler_params = {'mode': 'min', 'factor': 0.5, 'patience': 5}
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **scheduler_params
            )

        # 梯度裁剪
        self.max_grad_norm = max_grad_norm

        # 损失函数（交叉熵）
        self.criterion = nn.CrossEntropyLoss()

        # 金融指标计算器
        self.compute_financial_metrics = compute_financial_metrics
        if compute_financial_metrics:
            self.metrics_calculator = FinancialMetricsCalculator()

        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': []
        }

        # 金融指标历史
        if compute_financial_metrics:
            self.val_history['IC'] = []
            self.val_history['RankIC'] = []
            self.val_history['long_short_return'] = []

        # ⭐ NaN/Inf检测器（用于调试）
        self.nan_detector = NaNDetector(model, check_frequency=50)
        self.enable_nan_detection = False  # 默认关闭，可在训练时开启

        # ⭐ 梯度监控器（用于调试）
        self.gradient_monitor = GradientMonitor(model)
        self.enable_gradient_monitor = False  # 默认关闭

    def enable_debugging(self, enable_nan_detection: bool = True, enable_gradient_monitor: bool = False):
        """
        启用调试模式

        Args:
            enable_nan_detection: 启用NaN/Inf检测
            enable_gradient_monitor: 启用梯度监控
        """
        self.enable_nan_detection = enable_nan_detection
        self.enable_gradient_monitor = enable_gradient_monitor

        if enable_nan_detection:
            self.nan_detector.enable()
            print("✓ NaN/Inf detection enabled")

        if enable_gradient_monitor:
            self.gradient_monitor.register_hooks()
            print("✓ Gradient monitoring enabled")

    def disable_debugging(self):
        """禁用调试模式"""
        self.enable_nan_detection = False
        self.enable_gradient_monitor = False
        self.nan_detector.disable()
        self.gradient_monitor.remove_hooks()
        print("✓ Debugging disabled")

    def print_monitoring_report(self):
        """打印监控报告"""
        if self.enable_gradient_monitor:
            self.gradient_monitor.print_summary(top_k=10)
        if self.enable_nan_detection:
            self.nan_detector.print_report()

    def train_epoch(self, dataloader: DataLoader, adj_matrix: torch.Tensor) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            adj_matrix: 邻接矩阵
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        adj_matrix = adj_matrix.to(self.device)
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            # 准备数据
            sequences = batch['sequence'].to(self.device)  # [batch_size, max_seq_len, features]
            targets = batch['target'].to(self.device)  # [batch_size]
            masks = batch['mask'].to(self.device)  # [batch_size, max_seq_len]
            industry_indices = batch['industry_idx'].to(self.device)  # [batch_size]
            
            batch_size, max_seq_len, features = sequences.shape
            
            # 提取不同时间窗口的数据
            # 假设max_seq_len=80，则：
            # - x_80: 全部80个时间步
            # - x_40: 最后40个时间步
            # - x_20: 最后20个时间步
            x_80 = sequences  # [batch_size, 80, features]
            x_40 = sequences[:, -40:, :]  # [batch_size, 40, features]
            x_20 = sequences[:, -20:, :]  # [batch_size, 20, features]
            
            # 对应的掩码
            mask_80 = masks  # [batch_size, 80]
            mask_40 = masks[:, -40:]  # [batch_size, 40]
            mask_20 = masks[:, -20:]  # [batch_size, 20]
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions, _ = self.model(
                x_20, x_40, x_80,
                mask_20, mask_40, mask_80,
                adj_matrix, industry_indices
            )
            
            # 计算损失
            loss = self.criterion(predictions, targets)

            # ⭐ NaN/Inf检测（在反向传播前）
            if self.enable_nan_detection:
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ NaN/Inf detected in loss!")
                    print(f"   Loss value: {loss.item()}")
                    print(f"   Predictions stats: mean={predictions.mean().item():.4f}, "
                          f"std={predictions.std().item():.4f}, "
                          f"min={predictions.min().item():.4f}, "
                          f"max={predictions.max().item():.4f}")
                    print(f"   Targets: {targets[:10].cpu().numpy()}")
                    self.nan_detector.print_report()
                    raise ValueError("Training collapsed due to NaN/Inf loss!")

            # 反向传播
            loss.backward()

            # ⭐ NaN检测（梯度）
            if self.enable_nan_detection:
                if not self.nan_detector.step(loss):
                    self.nan_detector.print_report()
                    raise ValueError("Training collapsed due to NaN/Inf in gradients!")

            # 梯度裁剪
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred_classes = predictions.argmax(dim=1)
            correct += (pred_classes == targets).sum().item()
            total += targets.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def train_epoch_cross_sectional(self, dataloader: DataLoader, adj_matrix: torch.Tensor,
                                    epoch: int = 0) -> Dict[str, float]:
        """
        横截面局部训练模式的训练epoch

        特点：
        - 时间步追踪
        - 支持node_mask
        - 记录门控值统计

        Args:
            dataloader: CrossSectionalLocalDataset的DataLoader
            adj_matrix: 完整86节点邻接矩阵
            epoch: 当前epoch编号

        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        adj_matrix = adj_matrix.to(self.device)

        # 时间步追踪
        current_time_step = -1
        time_step_losses = []
        time_step_accs = []
        
        # ⭐ 存储每个时间步的统计信息（用于最后统一输出）
        time_step_stats = {}  # {time_step: {'losses': [...], 'accs': [...]}}

        # 门控值统计
        all_gate_values = []

        pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # ⭐ 处理可变大小的batch（使用自定义collate函数）
            # batch['sequence'] 等是列表，需要逐个处理或合并
            sequences_list = batch['sequence']  # List of [num_active_i, max_seq_len, features]
            targets_list = batch['target']  # List of [num_active_i]
            masks_list = batch['mask']  # List of [num_active_i, max_seq_len]
            industry_indices_list = batch['industry_idx']  # List of [num_active_i]
            node_mask = batch['node_mask'].to(self.device)  # [batch_size, 86]
            time_indices = batch['time_index']  # [batch_size]
            
            # ⭐ 合并所有样本的序列（因为每个样本的num_active可能不同）
            # 将所有样本的序列、目标等合并成一个大的batch
            all_sequences = []
            all_targets = []
            all_masks = []
            all_industry_indices = []
            all_node_masks = []
            
            for i in range(len(sequences_list)):
                all_sequences.append(sequences_list[i].to(self.device))
                all_targets.append(targets_list[i].to(self.device))
                all_masks.append(masks_list[i].to(self.device))
                all_industry_indices.append(industry_indices_list[i].to(self.device))
                all_node_masks.append(node_mask[i])  # [86]
            
            # 合并所有序列
            sequences = torch.cat(all_sequences, dim=0)  # [total_active, max_seq_len, features]
            targets = torch.cat(all_targets, dim=0)  # [total_active]
            masks = torch.cat(all_masks, dim=0)  # [total_active, max_seq_len]
            industry_indices = torch.cat(all_industry_indices, dim=0)  # [total_active]
            
            # 使用第一个样本的time_index（通常batch中所有样本来自同一时间步）
            time_idx = time_indices[0].item() if len(time_indices) > 0 else -1

            # 检查是否进入新的时间步
            if time_idx != current_time_step:
                if current_time_step >= 0:
                    # ⭐ 存储上一个时间步的统计信息（不打印）
                    if len(time_step_losses) > 0:
                        time_step_stats[current_time_step] = {
                            'losses': time_step_losses.copy(),
                            'accs': time_step_accs.copy()
                        }

                current_time_step = time_idx
                time_step_losses = []
                time_step_accs = []

            num_active, max_seq_len, features = sequences.shape

            # 提取不同时间窗口的数据
            x_80 = sequences  # [num_active, 80, features]
            x_40 = sequences[:, -40:, :]  # [num_active, 40, features]
            x_20 = sequences[:, -20:, :]  # [num_active, 20, features]

            # 对应的掩码
            mask_80 = masks
            mask_40 = masks[:, -40:]
            mask_20 = masks[:, -20:]

            # 前向传播（横截面模式）
            # ⭐ 注意：由于合并了多个样本，node_mask需要特殊处理
            # 这里使用第一个样本的node_mask（通常batch中所有样本来自同一时间步）
            # 如果需要更精确的处理，可以分别处理每个样本
            batch_node_mask = all_node_masks[0] if len(all_node_masks) > 0 else node_mask[0]
            
            self.optimizer.zero_grad()
            predictions, _, gates = self.model(
                x_20, x_40, x_80,
                mask_20, mask_40, mask_80,
                adj_matrix, industry_indices,
                node_mask=batch_node_mask  # ⭐ 传递node_mask
            )

            # 计算损失
            loss = self.criterion(predictions, targets)

            # NaN/Inf检测
            if self.enable_nan_detection:
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n❌ NaN/Inf detected in loss!")
                    print(f"   Time step: {time_idx}")
                    print(f"   Batch: {batch_idx}")
                    print(f"   Loss value: {loss.item()}")
                    self.nan_detector.print_report()
                    raise ValueError("Training collapsed!")

            # 反向传播
            loss.backward()

            # NaN检测（梯度）
            if self.enable_nan_detection:
                if not self.nan_detector.step(loss):
                    self.nan_detector.print_report()
                    raise ValueError("Training collapsed!")

            # 梯度裁剪
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # 统计
            total_loss += loss.item() * num_active
            _, predicted = torch.max(predictions.data, 1)
            correct += (predicted == targets).sum().item()
            total += num_active

            # 记录时间步统计
            time_step_losses.append(loss.item())
            batch_acc = 100.0 * (predicted == targets).float().mean().item()
            time_step_accs.append(batch_acc)

            # 收集门控值
            if gates is not None:
                all_gate_values.append(gates.detach().cpu())

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.1f}%',
                'time_step': time_idx
            })

        # ⭐ 存储最后一个时间步的统计信息
        if len(time_step_losses) > 0:
            time_step_stats[current_time_step] = {
                'losses': time_step_losses.copy(),
                'accs': time_step_accs.copy()
            }

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        # ⭐ 统一输出所有时间步的统计信息
        if len(time_step_stats) > 0:
            print(f"\n{'='*60}")
            print(f"Time Step Statistics (Epoch {epoch+1}):")
            print(f"{'='*60}")
            
            # 按时间步排序
            sorted_time_steps = sorted(time_step_stats.keys())
            
            # 每10个时间步打印一次摘要，或全部打印（如果时间步数较少）
            if len(sorted_time_steps) <= 50:
                # 打印所有时间步
                for ts in sorted_time_steps:
                    stats = time_step_stats[ts]
                    avg_ts_loss = np.mean(stats['losses'])
                    avg_ts_acc = np.mean(stats['accs'])
                    print(f"  Time step {ts:4d}: Loss={avg_ts_loss:.4f}, Acc={avg_ts_acc:.2f}%")
            else:
                # 打印摘要：每10个时间步打印一次
                print(f"  Total time steps: {len(sorted_time_steps)}")
                print(f"  Showing summary (every 10th time step):")
                for i, ts in enumerate(sorted_time_steps):
                    if i % 10 == 0 or i == len(sorted_time_steps) - 1:
                        stats = time_step_stats[ts]
                        avg_ts_loss = np.mean(stats['losses'])
                        avg_ts_acc = np.mean(stats['accs'])
                        print(f"  Time step {ts:4d}: Loss={avg_ts_loss:.4f}, Acc={avg_ts_acc:.2f}%")
            
            # 打印总体统计
            all_losses = []
            all_accs = []
            for stats in time_step_stats.values():
                all_losses.extend(stats['losses'])
                all_accs.extend(stats['accs'])
            
            if len(all_losses) > 0:
                print(f"\n  Overall Statistics:")
                print(f"    Mean Loss: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
                print(f"    Mean Acc:  {np.mean(all_accs):.2f}% ± {np.std(all_accs):.2f}%")
                print(f"    Min Acc:   {np.min(all_accs):.2f}%")
                print(f"    Max Acc:   {np.max(all_accs):.2f}%")
            
            print(f"{'='*60}")

        # 计算门控值统计
        gate_stats = {}
        if len(all_gate_values) > 0:
            all_gates_tensor = torch.cat(all_gate_values, dim=0)  # [total_active_nodes, 1]
            gate_stats = {
                'gate_mean': all_gates_tensor.mean().item(),
                'gate_std': all_gates_tensor.std().item(),
                'gate_min': all_gates_tensor.min().item(),
                'gate_max': all_gates_tensor.max().item(),
                'favor_time_ratio': (all_gates_tensor > 0.5).float().mean().item(),
                'favor_embedding_ratio': (all_gates_tensor <= 0.5).float().mean().item()
            }

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            **gate_stats
        }

    def validate(self, dataloader: DataLoader, adj_matrix: torch.Tensor,
                compute_metrics: bool = True) -> Dict[str, float]:
        """
        验证模型

        Args:
            dataloader: 数据加载器
            adj_matrix: 邻接矩阵
            compute_metrics: 是否计算金融指标

        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # 收集预测和真实值用于金融指标计算
        all_predictions_prob = []
        all_targets = []
        all_returns = []

        adj_matrix = adj_matrix.to(self.device)

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validating')
            for batch in pbar:
                # 准备数据
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)
                industry_indices = batch['industry_idx'].to(self.device)
                
                batch_size, max_seq_len, features = sequences.shape
                
                # 提取不同时间窗口的数据
                x_80 = sequences
                x_40 = sequences[:, -40:, :]
                x_20 = sequences[:, -20:, :]
                
                mask_80 = masks
                mask_40 = masks[:, -40:]
                mask_20 = masks[:, -20:]
                
                # 前向传播
                predictions, _ = self.model(
                    x_20, x_40, x_80,
                    mask_20, mask_40, mask_80,
                    adj_matrix, industry_indices
                )
                
                # 计算损失
                loss = self.criterion(predictions, targets)
                
                # 统计
                total_loss += loss.item()
                pred_classes = predictions.argmax(dim=1)
                correct += (pred_classes == targets).sum().item()
                total += targets.size(0)

                # 收集数据用于金融指标
                if compute_metrics and self.compute_financial_metrics:
                    all_predictions_prob.append(predictions.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    # 尝试获取真实收益率
                    if 'return' in batch:
                        all_returns.append(batch['return'].cpu().numpy())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0.0

        results = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

        # 计算金融指标
        if compute_metrics and self.compute_financial_metrics and len(all_predictions_prob) > 0:
            all_predictions_prob = np.concatenate(all_predictions_prob, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # 使用预测概率的最高类别作为分数
            pred_scores = np.max(all_predictions_prob, axis=1)

            # 如果有真实收益率数据,计算IC等指标
            if len(all_returns) > 0:
                all_returns = np.concatenate(all_returns, axis=0)
                financial_metrics = self.metrics_calculator.compute_all_metrics(
                    pred_scores, all_returns
                )
                results.update(financial_metrics)
            else:
                # 如果没有收益率数据,使用目标类别作为替代
                # 将类别转换为连续值(-2, -1, 0, 1, 2)用于相关性计算
                pseudo_returns = all_targets - 2.0  # 假设5类:0,1,2,3,4 -> -2,-1,0,1,2
                financial_metrics = self.metrics_calculator.compute_all_metrics(
                    pred_scores, pseudo_returns
                )
                results.update(financial_metrics)

        return results
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              adj_matrix: torch.Tensor, num_epochs: int = 50,
              save_path: Optional[str] = None,
              use_cross_sectional: bool = False) -> Dict[str, list]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            adj_matrix: 邻接矩阵
            num_epochs: 训练轮数
            save_path: 模型保存路径
            use_cross_sectional: 是否使用横截面训练模式

        Returns:
            训练历史字典
        """
        best_val_acc = 0.0

        # ⭐ 如果使用节点级门控，添加门控统计记录
        if use_cross_sectional:
            self.train_history['gate_mean'] = []
            self.train_history['gate_std'] = []
            self.train_history['favor_time_ratio'] = []

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            # ⭐ 根据模式选择训练方法
            if use_cross_sectional:
                train_metrics = self.train_epoch_cross_sectional(train_loader, adj_matrix, epoch)
            else:
                train_metrics = self.train_epoch(train_loader, adj_matrix)

            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])

            # 记录门控统计
            if use_cross_sectional and 'gate_mean' in train_metrics:
                self.train_history['gate_mean'].append(train_metrics.get('gate_mean', 0.0))
                self.train_history['gate_std'].append(train_metrics.get('gate_std', 0.0))
                self.train_history['favor_time_ratio'].append(train_metrics.get('favor_time_ratio', 0.5))
            
            # 验证
            val_metrics = self.validate(val_loader, adj_matrix)
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])

            # 记录金融指标
            if self.compute_financial_metrics:
                for key in ['IC', 'RankIC', 'long_short_return']:
                    if key in val_metrics:
                        self.val_history[key].append(val_metrics[key])

            # 打印结果
            print(f'Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.2f}%')

            # 打印金融指标
            if self.compute_financial_metrics and 'IC' in val_metrics:
                print(f'Val IC: {val_metrics.get("IC", 0):.4f}, '
                      f'RankIC: {val_metrics.get("RankIC", 0):.4f}, '
                      f'Long-Short: {val_metrics.get("long_short_return", 0):.4f}')

            # 学习率调度
            if self.use_scheduler and self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Learning Rate: {current_lr:.6f}')

            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_accuracy': best_val_acc,
                        'val_metrics': val_metrics,
                    }, save_path)
                    print(f'Model saved to {save_path}')
        
        return {
            'train': self.train_history,
            'val': self.val_history
        }
    
    def predict(self, dataloader: DataLoader, adj_matrix: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            dataloader: 数据加载器
            adj_matrix: 邻接矩阵
            
        Returns:
            - 预测概率，形状为 [样本数, num_classes]
            - 预测类别，形状为 [样本数]
        """
        self.model.eval()
        all_predictions = []
        
        adj_matrix = adj_matrix.to(self.device)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                sequences = batch['sequence'].to(self.device)
                masks = batch['mask'].to(self.device)
                industry_indices = batch['industry_idx'].to(self.device)
                
                batch_size, max_seq_len, features = sequences.shape
                
                x_80 = sequences
                x_40 = sequences[:, -40:, :]
                x_20 = sequences[:, -20:, :]
                
                mask_80 = masks
                mask_40 = masks[:, -40:]
                mask_20 = masks[:, -20:]
                
                predictions, _ = self.model(
                    x_20, x_40, x_80,
                    mask_20, mask_40, mask_80,
                    adj_matrix, industry_indices
                )
                
                all_predictions.append(predictions.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        pred_classes = np.argmax(all_predictions, axis=1)

        return all_predictions, pred_classes

    def k_fold_validate(self, dataset, adj_matrix: torch.Tensor,
                       n_splits: int = 5, min_train_size: float = 0.4,
                       num_epochs: int = 30, batch_size: int = 32,
                       save_dir: str = "./checkpoints",
                       resume_from_checkpoint: bool = True,
                       load_previous_fold: bool = False) -> Dict[str, List]:
        """
        时间序列K折交叉验证

        Args:
            dataset: 完整数据集
            adj_matrix: 邻接矩阵
            n_splits: 折数
            min_train_size: 最小训练集比例
            num_epochs: 每折训练轮数
            batch_size: 批大小
            save_dir: 模型保存目录
            resume_from_checkpoint: 是否从checkpoint恢复（跳过已完成的fold）
            load_previous_fold: 是否从上一个fold的模型继续训练（False则重新初始化）

        Returns:
            K折验证结果字典
        """
        print(f"\n{'='*60}")
        print(f"Starting {n_splits}-Fold Time Series Cross-Validation")
        print(f"{'='*60}\n")

        # 创建K折验证器
        tscv = TimeSeriesKFold(n_splits=n_splits, min_train_size=min_train_size)

        # 准备索引数组
        indices = np.arange(len(dataset))

        # 存储所有折的结果
        fold_results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        if self.compute_financial_metrics:
            fold_results['val_IC'] = []
            fold_results['val_RankIC'] = []
            fold_results['val_long_short'] = []

        # 创建保存目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # ⭐ 检测已完成的fold（用于断点续训）
        completed_folds = []
        if resume_from_checkpoint:
            for fold_num in range(1, n_splits + 1):
                fold_checkpoint = os.path.join(save_dir, f"fold_{fold_num}_best.pth")
                if os.path.exists(fold_checkpoint):
                    try:
                        checkpoint = torch.load(fold_checkpoint, weights_only=False, map_location='cpu')
                        # 检查checkpoint是否完整（包含必要的键）
                        if 'model_state_dict' in checkpoint and 'val_metrics' in checkpoint:
                            completed_folds.append(fold_num)
                            print(f"✓ Found completed fold {fold_num} checkpoint")
                    except Exception as e:
                        print(f"⚠ Warning: Could not load fold {fold_num} checkpoint: {e}")
            
            if completed_folds:
                print(f"\n📋 Resuming training: Found {len(completed_folds)} completed fold(s): {completed_folds}")
                # 计算需要训练的fold
                all_folds = set(range(1, n_splits + 1))
                folds_to_train = sorted(all_folds - set(completed_folds))
                folds_to_skip = sorted(completed_folds)
                
                if folds_to_skip:
                    print(f"   ✓ Will SKIP fold(s): {folds_to_skip} (using checkpoint results)")
                if folds_to_train:
                    print(f"   → Will TRAIN fold(s): {folds_to_train}")
                else:
                    print(f"   ✓ All folds completed! Will only load results.")
            else:
                print(f"\n📋 No completed folds found, starting from scratch")
                print(f"   → Will TRAIN all folds: {list(range(1, n_splits + 1))}")

        # K折验证
        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices), 1):
            # ⭐ 跳过已完成的fold
            if resume_from_checkpoint and fold in completed_folds:
                print(f"\n{'-'*60}")
                print(f"Fold {fold}/{n_splits} - SKIPPED (already completed)")
                print(f"{'-'*60}\n")
                
                # 加载已完成的fold结果
                fold_checkpoint = os.path.join(save_dir, f"fold_{fold}_best.pth")
                checkpoint = torch.load(fold_checkpoint, weights_only=False, map_location='cpu')
                
                # 创建数据加载器用于评估
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=0)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                # 加载模型状态
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # 评估（如果需要重新计算指标）
                final_val_metrics = self.validate(val_loader, adj_matrix)
                final_train_metrics = self.validate(train_loader, adj_matrix, compute_metrics=False)
                
                # 使用checkpoint中的指标，或重新计算的指标
                if 'val_metrics' in checkpoint:
                    final_val_metrics = checkpoint['val_metrics']
                
                # 记录结果
                fold_results['train_loss'].append(final_train_metrics['loss'])
                fold_results['train_acc'].append(final_train_metrics['accuracy'])
                fold_results['val_loss'].append(final_val_metrics['loss'])
                fold_results['val_acc'].append(final_val_metrics['accuracy'])
                
                if self.compute_financial_metrics:
                    fold_results['val_IC'].append(final_val_metrics.get('IC', 0))
                    fold_results['val_RankIC'].append(final_val_metrics.get('RankIC', 0))
                    fold_results['val_long_short'].append(final_val_metrics.get('long_short_return', 0))
                
                print(f"Fold {fold} Results (from checkpoint):")
                print(f"  Val Loss: {final_val_metrics['loss']:.4f}")
                print(f"  Val Acc: {final_val_metrics['accuracy']:.2f}%")
                if self.compute_financial_metrics:
                    print(f"  Val IC: {final_val_metrics.get('IC', 0):.4f}")
                    print(f"  Val RankIC: {final_val_metrics.get('RankIC', 0):.4f}")
                
                continue
            print(f"\n{'-'*60}")
            print(f"Fold {fold}/{n_splits}")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            print(f"{'-'*60}\n")

            # 创建子数据集
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # 创建数据加载器
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=False,  # 保持时间顺序
                num_workers=0
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

            # ⭐ 模型初始化策略
            if load_previous_fold and fold > 1:
                # 从上一个fold的checkpoint加载模型
                prev_fold_checkpoint = os.path.join(save_dir, f"fold_{fold-1}_best.pth")
                if os.path.exists(prev_fold_checkpoint):
                    try:
                        prev_checkpoint = torch.load(prev_fold_checkpoint, weights_only=False, map_location=self.device)
                        self.model.load_state_dict(prev_checkpoint['model_state_dict'])
                        print(f"  ✓ Loaded model from fold {fold-1} checkpoint")
                        
                        # 可选：也加载优化器状态（如果checkpoint中有）
                        if 'optimizer_state_dict' in prev_checkpoint:
                            self.optimizer.load_state_dict(prev_checkpoint['optimizer_state_dict'])
                            print(f"  ✓ Loaded optimizer from fold {fold-1} checkpoint")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load fold {fold-1} checkpoint: {e}")
                        print(f"  → Reinitializing model parameters")
                        # 如果加载失败，重新初始化
                        for layer in self.model.children():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                else:
                    # 上一个fold的checkpoint不存在，重新初始化
                    for layer in self.model.children():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
            else:
                # 重新初始化模型(每折重新训练)
                # 注意:这里需要从外部传入模型构造函数
                # 为简化,我们重置模型参数
                for layer in self.model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

            # 重新初始化优化器（除非从checkpoint加载了）
            if not (load_previous_fold and fold > 1 and os.path.exists(os.path.join(save_dir, f"fold_{fold-1}_best.pth"))):
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    weight_decay=self.optimizer.defaults['weight_decay']
                )

            # 训练当前折
            best_val_acc = 0.0
            fold_save_path = os.path.join(save_dir, f"fold_{fold}_best.pth")

            for epoch in range(num_epochs):
                print(f'Fold {fold}, Epoch {epoch + 1}/{num_epochs}')

                # 训练
                train_metrics = self.train_epoch(train_loader, adj_matrix)

                # 验证
                val_metrics = self.validate(val_loader, adj_matrix)

                print(f'  Train Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%')
                print(f'  Val Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%')

                if self.compute_financial_metrics and 'IC' in val_metrics:
                    print(f'  Val IC: {val_metrics["IC"]:.4f}, RankIC: {val_metrics["RankIC"]:.4f}')

                # 保存最佳模型
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    torch.save({
                        'fold': fold,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),  # ⭐ 保存优化器状态
                        'val_metrics': val_metrics,
                        'best_val_acc': best_val_acc,
                    }, fold_save_path)

            # 加载最佳模型进行最终评估
            checkpoint = torch.load(fold_save_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 最终验证
            final_val_metrics = self.validate(val_loader, adj_matrix)
            final_train_metrics = self.validate(train_loader, adj_matrix, compute_metrics=False)

            # 记录结果
            fold_results['train_loss'].append(final_train_metrics['loss'])
            fold_results['train_acc'].append(final_train_metrics['accuracy'])
            fold_results['val_loss'].append(final_val_metrics['loss'])
            fold_results['val_acc'].append(final_val_metrics['accuracy'])

            if self.compute_financial_metrics:
                fold_results['val_IC'].append(final_val_metrics.get('IC', 0))
                fold_results['val_RankIC'].append(final_val_metrics.get('RankIC', 0))
                fold_results['val_long_short'].append(final_val_metrics.get('long_short_return', 0))

            print(f"\nFold {fold} Final Results:")
            print(f"  Val Loss: {final_val_metrics['loss']:.4f}")
            print(f"  Val Acc: {final_val_metrics['accuracy']:.2f}%")
            if self.compute_financial_metrics:
                print(f"  Val IC: {final_val_metrics.get('IC', 0):.4f}")
                print(f"  Val RankIC: {final_val_metrics.get('RankIC', 0):.4f}")

        # 打印汇总结果
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Summary")
        print(f"{'='*60}\n")

        print(f"Average Train Loss: {np.mean(fold_results['train_loss']):.4f} ± {np.std(fold_results['train_loss']):.4f}")
        print(f"Average Train Acc: {np.mean(fold_results['train_acc']):.2f}% ± {np.std(fold_results['train_acc']):.2f}%")
        print(f"Average Val Loss: {np.mean(fold_results['val_loss']):.4f} ± {np.std(fold_results['val_loss']):.4f}")
        print(f"Average Val Acc: {np.mean(fold_results['val_acc']):.2f}% ± {np.std(fold_results['val_acc']):.2f}%")

        if self.compute_financial_metrics:
            print(f"\nFinancial Metrics:")
            print(f"Average IC: {np.mean(fold_results['val_IC']):.4f} ± {np.std(fold_results['val_IC']):.4f}")
            print(f"Average RankIC: {np.mean(fold_results['val_RankIC']):.4f} ± {np.std(fold_results['val_RankIC']):.4f}")
            print(f"Average Long-Short: {np.mean(fold_results['val_long_short']):.4f} ± {np.std(fold_results['val_long_short']):.4f}")

        # ⭐ 保存最佳模型（选择验证准确率最高的fold）
        best_fold_idx = np.argmax(fold_results['val_acc'])
        best_fold = best_fold_idx + 1  # fold编号从1开始
        best_fold_path = os.path.join(save_dir, f"fold_{best_fold}_best.pth")
        best_model_path = os.path.join(save_dir, "best_model.pth")
        
        # 复制最佳fold的模型为best_model.pth
        if os.path.exists(best_fold_path):
            shutil.copy2(best_fold_path, best_model_path)
            print(f"\n✓ Best model saved: {best_model_path} (from Fold {best_fold}, Val Acc: {fold_results['val_acc'][best_fold_idx]:.2f}%)")
        else:
            print(f"\n⚠ Warning: Could not find {best_fold_path} to save as best_model.pth")

        return fold_results

