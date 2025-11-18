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
              save_path: Optional[str] = None) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            adj_matrix: 邻接矩阵
            num_epochs: 训练轮数
            save_path: 模型保存路径
            
        Returns:
            训练历史字典
        """
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)
            
            # 训练
            train_metrics = self.train_epoch(train_loader, adj_matrix)
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            
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
                       save_dir: str = "./checkpoints") -> Dict[str, List]:
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

        # K折验证
        for fold, (train_idx, val_idx) in enumerate(tscv.split(indices), 1):
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

            # 重新初始化模型(每折重新训练)
            # 注意:这里需要从外部传入模型构造函数
            # 为简化,我们重置模型参数
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            # 重新初始化优化器
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
                        'val_metrics': val_metrics,
                    }, fold_save_path)

            # 加载最佳模型进行最终评估
            checkpoint = torch.load(fold_save_path)
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

        return fold_results

