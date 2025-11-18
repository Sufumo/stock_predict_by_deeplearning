"""
可视化工具
绘制训练曲线、注意力权重、混淆矩阵等
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path


class Visualizer:
    """可视化工具类"""

    def __init__(self, save_dir: str = "./visualizations", dpi: int = 300):
        """
        Args:
            save_dir: 图片保存目录
            dpi: 图片分辨率
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_training_curves(self, history: Dict[str, Dict[str, List]], save_name: str = "training_curves.png"):
        """
        绘制训练和验证曲线

        Args:
            history: 训练历史,格式为 {'train': {...}, 'val': {...}}
            save_name: 保存文件名
        """
        train_hist = history.get('train', {})
        val_hist = history.get('val', {})

        # 确定要绘制的指标
        metrics_to_plot = []
        if 'loss' in train_hist:
            metrics_to_plot.append(('loss', 'Loss'))
        if 'accuracy' in train_hist:
            metrics_to_plot.append(('accuracy', 'Accuracy (%)'))
        if 'IC' in val_hist:
            metrics_to_plot.append(('IC', 'IC'))
        if 'RankIC' in val_hist:
            metrics_to_plot.append(('RankIC', 'RankIC'))

        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            print("No metrics to plot")
            return

        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        epochs = range(1, len(train_hist.get('loss', [])) + 1)

        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx]

            # 训练曲线
            if metric_key in train_hist:
                ax.plot(epochs, train_hist[metric_key], 'b-', label=f'Train {metric_name}', linewidth=2)

            # 验证曲线
            if metric_key in val_hist:
                ax.plot(epochs, val_hist[metric_key], 'r-', label=f'Val {metric_name}', linewidth=2)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} over Epochs', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             save_name: str = "confusion_matrix.png"):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
            save_name: 保存文件名
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        if class_names is None:
            class_names = [f'Q{i+1}' for i in range(n_classes)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})

        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_attention_weights(self, attention_matrix: np.ndarray,
                              industry_names: Optional[List[str]] = None,
                              save_name: str = "attention_weights.png",
                              top_k: int = 20):
        """
        绘制GAT注意力权重热力图

        Args:
            attention_matrix: 注意力矩阵 [num_industries, num_industries]
            industry_names: 行业名称列表
            save_name: 保存文件名
            top_k: 只显示前k个行业(避免过于拥挤)
        """
        n_industries = attention_matrix.shape[0]

        # 如果行业太多,只显示top_k个
        if n_industries > top_k:
            # 计算每个行业的平均注意力强度
            avg_attention = attention_matrix.sum(axis=1)
            top_indices = np.argsort(avg_attention)[-top_k:]
            attention_matrix = attention_matrix[top_indices][:, top_indices]

            if industry_names is not None:
                industry_names = [industry_names[i] for i in top_indices]

        # 如果没有名称,使用索引
        if industry_names is None:
            industry_names = [f'Ind{i}' for i in range(attention_matrix.shape[0])]

        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_matrix, cmap='YlOrRd',
                   xticklabels=industry_names,
                   yticklabels=industry_names,
                   cbar_kws={'label': 'Attention Weight'})

        plt.xlabel('Target Industry', fontsize=12)
        plt.ylabel('Source Industry', fontsize=12)
        plt.title(f'GAT Attention Weights (Top {top_k} Industries)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Attention weights saved to {save_path}")

    def plot_quantile_returns(self, predictions: np.ndarray, returns: np.ndarray,
                             n_quantiles: int = 5,
                             save_name: str = "quantile_returns.png"):
        """
        绘制分位数收益分析

        Args:
            predictions: 预测分数
            returns: 真实收益率
            n_quantiles: 分位数个数
            save_name: 保存文件名
        """
        # 按预测值排序,分成n个分位数
        sorted_indices = np.argsort(predictions)
        quantile_size = len(sorted_indices) // n_quantiles

        quantile_returns = []
        quantile_labels = []

        for i in range(n_quantiles):
            start_idx = i * quantile_size
            if i == n_quantiles - 1:
                end_idx = len(sorted_indices)
            else:
                end_idx = (i + 1) * quantile_size

            quantile_idx = sorted_indices[start_idx:end_idx]
            quantile_ret = returns[quantile_idx]
            quantile_returns.append(quantile_ret)
            quantile_labels.append(f'Q{i+1}')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 箱线图
        bp = ax1.boxplot(quantile_returns, labels=quantile_labels,
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Quantile', fontsize=12)
        ax1.set_ylabel('Returns', fontsize=12)
        ax1.set_title('Return Distribution by Prediction Quantile', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 平均收益柱状图
        mean_returns = [np.mean(q) for q in quantile_returns]
        colors = ['red' if r < 0 else 'green' for r in mean_returns]
        ax2.bar(quantile_labels, mean_returns, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Quantile', fontsize=12)
        ax2.set_ylabel('Mean Return', fontsize=12)
        ax2.set_title('Mean Return by Prediction Quantile', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (label, value) in enumerate(zip(quantile_labels, mean_returns)):
            ax2.text(i, value, f'{value:.4f}', ha='center',
                    va='bottom' if value > 0 else 'top')

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"Quantile returns plot saved to {save_path}")

    def plot_kfold_results(self, fold_results: Dict[str, List],
                          save_name: str = "kfold_results.png"):
        """
        绘制K折验证结果

        Args:
            fold_results: K折结果字典
            save_name: 保存文件名
        """
        metrics = list(fold_results.keys())
        n_metrics = len(metrics)
        n_folds = len(fold_results[metrics[0]])

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        folds = list(range(1, n_folds + 1))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = fold_results[metric]

            ax.plot(folds, values, 'o-', linewidth=2, markersize=8)
            ax.axhline(y=np.mean(values), color='r', linestyle='--',
                      label=f'Mean: {np.mean(values):.4f}')

            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} across Folds', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"K-fold results plot saved to {save_path}")


if __name__ == "__main__":
    # 测试
    print("Testing Visualizer...")

    vis = Visualizer(save_dir="./test_visualizations")

    # 测试训练曲线
    history = {
        'train': {
            'loss': [0.8, 0.6, 0.5, 0.4, 0.3],
            'accuracy': [60, 70, 75, 80, 85]
        },
        'val': {
            'loss': [0.9, 0.7, 0.6, 0.5, 0.4],
            'accuracy': [55, 65, 70, 75, 80],
            'IC': [0.05, 0.10, 0.15, 0.18, 0.20]
        }
    }
    vis.plot_training_curves(history)

    # 测试混淆矩阵
    y_true = np.random.randint(0, 5, 100)
    y_pred = np.random.randint(0, 5, 100)
    vis.plot_confusion_matrix(y_true, y_pred)

    print("Visualization tests completed!")
