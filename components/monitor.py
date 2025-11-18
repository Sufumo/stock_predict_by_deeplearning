"""
训练监控组件
用于监控梯度、激活值和NaN/Inf
"""
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class GradientMonitor:
    """
    梯度监控器
    记录每层的梯度统计信息
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch模型
        """
        self.model = model
        self.gradient_stats = defaultdict(dict)
        self.hooks = []

    def register_hooks(self):
        """注册反向传播钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(name, grad)
                )
                self.hooks.append(hook)

    def _gradient_hook(self, name: str, grad: torch.Tensor):
        """梯度钩子回调函数"""
        if grad is not None:
            grad_cpu = grad.detach().cpu()
            self.gradient_stats[name] = {
                'mean': grad_cpu.mean().item(),
                'std': grad_cpu.std().item(),
                'max': grad_cpu.max().item(),
                'min': grad_cpu.min().item(),
                'norm': grad_cpu.norm().item(),
                'has_nan': torch.isnan(grad_cpu).any().item(),
                'has_inf': torch.isinf(grad_cpu).any().item()
            }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        return dict(self.gradient_stats)

    def print_summary(self, top_k: int = 10):
        """
        打印梯度统计摘要

        Args:
            top_k: 显示前k个梯度范数最大的层
        """
        if not self.gradient_stats:
            print("No gradient statistics available")
            return

        # 按梯度范数排序
        sorted_stats = sorted(
            self.gradient_stats.items(),
            key=lambda x: x[1]['norm'],
            reverse=True
        )

        print(f"\n{'='*80}")
        print(f"Gradient Statistics (Top {top_k} by norm)")
        print(f"{'='*80}")
        print(f"{'Layer Name':<50} {'Norm':>10} {'Mean':>10} {'Std':>10} {'NaN/Inf':>10}")
        print(f"{'-'*80}")

        for i, (name, stats) in enumerate(sorted_stats[:top_k]):
            has_problem = "✓" if stats['has_nan'] or stats['has_inf'] else ""
            print(f"{name:<50} {stats['norm']:>10.4f} {stats['mean']:>10.4f} "
                  f"{stats['std']:>10.4f} {has_problem:>10}")

        # 检查是否有NaN/Inf
        nan_layers = [name for name, stats in self.gradient_stats.items() if stats['has_nan']]
        inf_layers = [name for name, stats in self.gradient_stats.items() if stats['has_inf']]

        if nan_layers:
            print(f"\n⚠️  WARNING: NaN gradients detected in {len(nan_layers)} layers:")
            for layer in nan_layers[:5]:
                print(f"   - {layer}")

        if inf_layers:
            print(f"\n⚠️  WARNING: Inf gradients detected in {len(inf_layers)} layers:")
            for layer in inf_layers[:5]:
                print(f"   - {layer}")

    def clear(self):
        """清除统计信息"""
        self.gradient_stats.clear()

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ActivationMonitor:
    """
    激活值监控器
    记录每层的激活值统计信息
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch模型
        """
        self.model = model
        self.activation_stats = defaultdict(dict)
        self.hooks = []

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        注册前向传播钩子

        Args:
            layer_names: 要监控的层名称列表，None表示监控所有层
        """
        for name, module in self.model.named_modules():
            # 跳过容器模块
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue

            # 如果指定了层名称，只监控这些层
            if layer_names is not None and name not in layer_names:
                continue

            hook = module.register_forward_hook(
                lambda module, input, output, name=name: self._activation_hook(name, output)
            )
            self.hooks.append(hook)

    def _activation_hook(self, name: str, output: torch.Tensor):
        """激活值钩子回调函数"""
        if isinstance(output, torch.Tensor):
            output_cpu = output.detach().cpu()
            self.activation_stats[name] = {
                'mean': output_cpu.mean().item(),
                'std': output_cpu.std().item(),
                'max': output_cpu.max().item(),
                'min': output_cpu.min().item(),
                'has_nan': torch.isnan(output_cpu).any().item(),
                'has_inf': torch.isinf(output_cpu).any().item()
            }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        return dict(self.activation_stats)

    def print_summary(self, top_k: int = 10):
        """
        打印激活值统计摘要

        Args:
            top_k: 显示前k个激活值范围最大的层
        """
        if not self.activation_stats:
            print("No activation statistics available")
            return

        # 按激活值范围(max-min)排序
        sorted_stats = sorted(
            self.activation_stats.items(),
            key=lambda x: x[1]['max'] - x[1]['min'],
            reverse=True
        )

        print(f"\n{'='*80}")
        print(f"Activation Statistics (Top {top_k} by range)")
        print(f"{'='*80}")
        print(f"{'Layer Name':<50} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'-'*80}")

        for i, (name, stats) in enumerate(sorted_stats[:top_k]):
            print(f"{name:<50} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} {stats['max']:>10.4f}")

        # 检查是否有NaN/Inf
        nan_layers = [name for name, stats in self.activation_stats.items() if stats['has_nan']]
        inf_layers = [name for name, stats in self.activation_stats.items() if stats['has_inf']]

        if nan_layers:
            print(f"\n⚠️  WARNING: NaN activations detected in {len(nan_layers)} layers:")
            for layer in nan_layers[:5]:
                print(f"   - {layer}")

        if inf_layers:
            print(f"\n⚠️  WARNING: Inf activations detected in {len(inf_layers)} layers:")
            for layer in inf_layers[:5]:
                print(f"   - {layer}")

    def clear(self):
        """清除统计信息"""
        self.activation_stats.clear()

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class NaNDetector:
    """
    NaN/Inf早期检测器
    在训练过程中检测NaN/Inf并及时报告
    """

    def __init__(self, model: nn.Module, check_frequency: int = 50):
        """
        Args:
            model: PyTorch模型
            check_frequency: 检查频率（每N步检查一次）
        """
        self.model = model
        self.check_frequency = check_frequency
        self.step_count = 0
        self.gradient_monitor = GradientMonitor(model)
        self.activation_monitor = ActivationMonitor(model)

    def enable(self):
        """启用监控"""
        self.gradient_monitor.register_hooks()
        self.activation_monitor.register_hooks()

    def disable(self):
        """禁用监控"""
        self.gradient_monitor.remove_hooks()
        self.activation_monitor.remove_hooks()

    def check_parameters(self) -> bool:
        """
        检查模型参数是否包含NaN/Inf

        Returns:
            True表示正常，False表示检测到问题
        """
        for name, param in self.model.named_parameters():
            if param.data is not None:
                if torch.isnan(param.data).any():
                    print(f"❌ NaN detected in parameter: {name}")
                    return False
                if torch.isinf(param.data).any():
                    print(f"❌ Inf detected in parameter: {name}")
                    return False
        return True

    def check_gradients(self) -> bool:
        """
        检查梯度是否包含NaN/Inf

        Returns:
            True表示正常，False表示检测到问题
        """
        grad_stats = self.gradient_monitor.get_stats()
        for name, stats in grad_stats.items():
            if stats['has_nan']:
                print(f"❌ NaN detected in gradient: {name}")
                print(f"   Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")
                return False
            if stats['has_inf']:
                print(f"❌ Inf detected in gradient: {name}")
                print(f"   Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")
                return False
        return True

    def check_activations(self) -> bool:
        """
        检查激活值是否包含NaN/Inf

        Returns:
            True表示正常，False表示检测到问题
        """
        activation_stats = self.activation_monitor.get_stats()
        for name, stats in activation_stats.items():
            if stats['has_nan']:
                print(f"❌ NaN detected in activation: {name}")
                print(f"   Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
                return False
            if stats['has_inf']:
                print(f"❌ Inf detected in activation: {name}")
                print(f"   Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
                return False
        return True

    def step(self, loss: torch.Tensor = None) -> bool:
        """
        执行一次检查

        Args:
            loss: 当前步的损失值

        Returns:
            True表示正常，False表示检测到问题
        """
        self.step_count += 1

        # 检查loss
        if loss is not None:
            if torch.isnan(loss):
                print(f"❌ NaN detected in loss at step {self.step_count}")
                return False
            if torch.isinf(loss):
                print(f"❌ Inf detected in loss at step {self.step_count}")
                return False

        # 定期检查梯度和激活值
        if self.step_count % self.check_frequency == 0:
            if not self.check_gradients():
                return False
            if not self.check_activations():
                return False

        return True

    def print_report(self):
        """打印监控报告"""
        print(f"\n{'='*80}")
        print(f"NaN Detector Report (Step {self.step_count})")
        print(f"{'='*80}")

        # 参数检查
        params_ok = self.check_parameters()
        print(f"Parameters: {'✓ OK' if params_ok else '✗ FAILED'}")

        # 打印梯度和激活值统计
        self.gradient_monitor.print_summary(top_k=5)
        self.activation_monitor.print_summary(top_k=5)
