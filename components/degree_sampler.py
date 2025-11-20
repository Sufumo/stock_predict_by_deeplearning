"""
基于度数的节点采样器
用于横截面局部训练中选择中心节点
"""
import torch
import numpy as np
from typing import List, Optional, Set


class DegreeBasedSampler:
    """
    基于行业关系图度数的采样器
    度数高的行业（如科技、金融）更容易被选为中心节点
    """

    def __init__(self, adj_matrix: torch.Tensor, temperature: float = 1.0):
        """
        Args:
            adj_matrix: [num_industries, num_industries] 邻接矩阵
            temperature: 温度参数，控制采样平滑度
                        - 1.0: 严格按度数概率采样
                        - >1.0: 更均匀（弱化度数差异）
                        - <1.0: 更陡峭（强化度数差异）
        """
        self.num_nodes = adj_matrix.shape[0]
        self.adj_matrix = adj_matrix
        self.temperature = temperature

        # 计算每个节点的度数（出度+入度，无向图则相同）
        self.degrees = adj_matrix.sum(dim=1).float()  # [num_industries]

        # 转换为采样概率
        if self.degrees.sum() > 0:
            # 应用温度
            scaled_degrees = self.degrees / self.temperature
            # Softmax转换为概率
            self.probs = torch.softmax(scaled_degrees, dim=0)
        else:
            # 如果没有边，均匀分布
            self.probs = torch.ones(self.num_nodes) / self.num_nodes

        # 统计信息
        self.sample_counts = torch.zeros(self.num_nodes, dtype=torch.long)

    def sample(self, k: int, exclude_indices: Optional[Set[int]] = None) -> List[int]:
        """
        采样k个中心节点

        Args:
            k: 采样数量
            exclude_indices: 要排除的节点索引集合（已经被采样过的）

        Returns:
            sampled_indices: 采样的节点索引列表
        """
        if k > self.num_nodes:
            raise ValueError(f"Cannot sample {k} nodes from {self.num_nodes} total nodes")

        # 创建候选节点集合
        if exclude_indices is None:
            candidate_indices = list(range(self.num_nodes))
            candidate_probs = self.probs.clone()
        else:
            # 排除已采样的节点
            candidate_indices = [i for i in range(self.num_nodes) if i not in exclude_indices]
            if len(candidate_indices) < k:
                raise ValueError(f"Not enough candidates: {len(candidate_indices)} < {k}")

            # 重新归一化概率
            mask = torch.ones(self.num_nodes, dtype=torch.bool)
            mask[list(exclude_indices)] = False
            candidate_probs = self.probs.clone()
            candidate_probs[~mask] = 0
            candidate_probs = candidate_probs / candidate_probs.sum()

        # 基于概率采样（无放回）
        sampled_indices = torch.multinomial(
            candidate_probs,
            num_samples=k,
            replacement=False
        ).tolist()

        # 更新统计
        for idx in sampled_indices:
            self.sample_counts[idx] += 1

        return sampled_indices

    def sample_uniform(self, k: int, exclude_indices: Optional[Set[int]] = None) -> List[int]:
        """
        均匀采样（作为对比baseline）

        Args:
            k: 采样数量
            exclude_indices: 要排除的节点索引集合

        Returns:
            sampled_indices: 采样的节点索引列表
        """
        if exclude_indices is None:
            candidates = list(range(self.num_nodes))
        else:
            candidates = [i for i in range(self.num_nodes) if i not in exclude_indices]

        if len(candidates) < k:
            raise ValueError(f"Not enough candidates: {len(candidates)} < {k}")

        # 均匀随机采样
        sampled_indices = np.random.choice(candidates, size=k, replace=False).tolist()

        # 更新统计
        for idx in sampled_indices:
            self.sample_counts[idx] += 1

        return sampled_indices

    def get_neighbors(self, center_indices: List[int]) -> List[int]:
        """
        获取给定中心节点的所有1-hop邻居

        Args:
            center_indices: 中心节点索引列表

        Returns:
            neighbor_indices: 邻居节点索引列表（不包含中心节点本身）
        """
        neighbor_set = set()

        for idx in center_indices:
            # 找到与idx相连的所有节点
            neighbors = torch.where(self.adj_matrix[idx] > 0)[0]
            neighbor_set.update(neighbors.tolist())

        # 移除中心节点本身
        neighbor_set -= set(center_indices)

        return sorted(list(neighbor_set))

    def reset_statistics(self):
        """重置采样统计"""
        self.sample_counts = torch.zeros(self.num_nodes, dtype=torch.long)

    def get_sampling_statistics(self) -> dict:
        """
        获取采样统计信息

        Returns:
            stats: 包含采样统计的字典
        """
        return {
            'total_samples': self.sample_counts.sum().item(),
            'mean_samples_per_node': self.sample_counts.float().mean().item(),
            'std_samples_per_node': self.sample_counts.float().std().item(),
            'min_samples': self.sample_counts.min().item(),
            'max_samples': self.sample_counts.max().item(),
            'sample_counts': self.sample_counts.tolist(),
            'degrees': self.degrees.tolist(),
            'sampling_probs': self.probs.tolist()
        }


class SequentialSampler:
    """
    顺序采样器（确保每个时间步每个行业被采样恰好一次）
    作为对比baseline
    """

    def __init__(self, num_nodes: int, adj_matrix: torch.Tensor, shuffle: bool = True):
        """
        Args:
            num_nodes: 节点总数
            adj_matrix: 邻接矩阵（用于获取邻居）
            shuffle: 是否在每个epoch开始时打乱顺序
        """
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.shuffle = shuffle

        # 初始化采样顺序
        self.reset()

    def reset(self):
        """重置采样器（新的epoch或时间步）"""
        if self.shuffle:
            self.order = torch.randperm(self.num_nodes).tolist()
        else:
            self.order = list(range(self.num_nodes))
        self.current_idx = 0

    def sample(self, k: int) -> List[int]:
        """
        按顺序采样k个节点

        Args:
            k: 采样数量

        Returns:
            sampled_indices: 采样的节点索引列表
        """
        if self.current_idx + k > self.num_nodes:
            # 如果剩余节点不足，从头开始循环
            remaining = self.num_nodes - self.current_idx
            sampled = self.order[self.current_idx:] + self.order[:k-remaining]
            self.current_idx = k - remaining
        else:
            sampled = self.order[self.current_idx:self.current_idx + k]
            self.current_idx += k

        return sampled

    def get_neighbors(self, center_indices: List[int]) -> List[int]:
        """获取邻居节点（与DegreeBasedSampler保持一致）"""
        neighbor_set = set()
        for idx in center_indices:
            neighbors = torch.where(self.adj_matrix[idx] > 0)[0]
            neighbor_set.update(neighbors.tolist())
        neighbor_set -= set(center_indices)
        return sorted(list(neighbor_set))

    def has_remaining(self) -> bool:
        """检查当前时间步是否还有未采样的节点"""
        return self.current_idx < self.num_nodes
