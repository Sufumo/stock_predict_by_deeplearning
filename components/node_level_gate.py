"""
节点级门控层
为每个节点独立学习时间特征和行业嵌入的融合权重
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class NodeLevelGate(nn.Module):
    """
    节点级自适应门控机制

    为每个节点学习一个门控值 g ∈ [0, 1]，用于融合时间特征和行业嵌入：
    output = g * time_features + (1 - g) * embeddings

    门控值由节点的时间特征和嵌入通过MLP动态计算，
    让模型自主决定每个节点应该更依赖哪种特征。
    """

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Args:
            feature_dim: 特征维度（时间特征和嵌入的维度）
            hidden_dim: MLP隐藏层维度，默认为feature_dim
            dropout: Dropout率
        """
        super(NodeLevelGate, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim

        # 门控MLP: 输入为拼接的[time_features; embeddings]
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

        # 可选的特征变换层（在融合前对特征进行非线性变换）
        self.use_transform = False
        if self.use_transform:
            self.time_transform = nn.Linear(feature_dim, feature_dim)
            self.emb_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, time_features: torch.Tensor,
                embeddings: torch.Tensor,
                return_gates: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            time_features: [num_nodes, feature_dim] 时间序列特征
            embeddings: [num_nodes, feature_dim] 行业嵌入
            return_gates: 是否返回门控值（用于可视化分析）

        Returns:
            fused_features: [num_nodes, feature_dim] 融合后的特征
            gates: [num_nodes, 1] 门控值，如果return_gates=False则为None
        """
        # 1. 拼接时间特征和嵌入
        concat = torch.cat([time_features, embeddings], dim=-1)  # [num_nodes, feature_dim*2]

        # 2. 通过MLP计算门控值
        gates = self.gate_mlp(concat)  # [num_nodes, 1]

        # 3. 可选的特征变换
        if self.use_transform:
            time_features = self.time_transform(time_features)
            embeddings = self.emb_transform(embeddings)

        # 4. 门控融合
        fused_features = gates * time_features + (1 - gates) * embeddings

        if return_gates:
            return fused_features, gates
        else:
            return fused_features, None

    def get_gate_statistics(self, gates: torch.Tensor) -> dict:
        """
        获取门控值的统计信息（用于分析）

        Args:
            gates: [num_nodes, 1] 门控值

        Returns:
            stats: 统计信息字典
        """
        gates_flat = gates.squeeze(-1)  # [num_nodes]

        return {
            'mean': gates_flat.mean().item(),
            'std': gates_flat.std().item(),
            'min': gates_flat.min().item(),
            'max': gates_flat.max().item(),
            'median': gates_flat.median().item(),
            # 分位数
            'q25': gates_flat.quantile(0.25).item(),
            'q75': gates_flat.quantile(0.75).item(),
            # 倾向性统计
            'favor_time_ratio': (gates_flat > 0.5).float().mean().item(),  # >0.5倾向时间特征
            'favor_embedding_ratio': (gates_flat <= 0.5).float().mean().item(),  # <=0.5倾向嵌入
            'strong_time_ratio': (gates_flat > 0.7).float().mean().item(),  # >0.7强烈倾向时间
            'strong_embedding_ratio': (gates_flat < 0.3).float().mean().item()  # <0.3强烈倾向嵌入
        }


class GlobalGate(nn.Module):
    """
    全局门控机制（所有节点共享一个可学习的融合权重）
    作为对比baseline，参数量更少
    """

    def __init__(self, initial_alpha: float = 0.5):
        """
        Args:
            initial_alpha: 初始的融合权重（0-1之间）
        """
        super(GlobalGate, self).__init__()

        # 可学习的全局融合权重
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self, time_features: torch.Tensor,
                embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            time_features: [num_nodes, feature_dim]
            embeddings: [num_nodes, feature_dim]

        Returns:
            fused_features: [num_nodes, feature_dim]
            alpha_value: scalar tensor（全局alpha值）
        """
        # 使用sigmoid确保alpha在[0,1]范围内
        alpha = torch.sigmoid(self.alpha)

        # 融合
        fused_features = alpha * time_features + (1 - alpha) * embeddings

        # 返回alpha值用于监控
        return fused_features, alpha.unsqueeze(0).unsqueeze(0)


class AdaptiveGate(nn.Module):
    """
    自适应门控（基于节点特征的条件门控）

    与NodeLevelGate的区别：
    - NodeLevelGate: 每个节点看到自己的时间特征和嵌入后决定如何融合
    - AdaptiveGate: 基于节点特征学习一个通用的融合策略
    """

    def __init__(self, feature_dim: int, num_heads: int = 1):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 多头门控数量
        """
        super(AdaptiveGate, self).__init__()

        self.num_heads = num_heads
        self.feature_dim = feature_dim

        # 多头注意力风格的门控
        if num_heads > 1:
            self.query = nn.Linear(feature_dim, feature_dim)
            self.key = nn.Linear(feature_dim, feature_dim)
            self.gate_linear = nn.Linear(feature_dim, num_heads)
        else:
            self.gate_linear = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim // 2),
                nn.Tanh(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            )

    def forward(self, time_features: torch.Tensor,
                embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        if self.num_heads == 1:
            concat = torch.cat([time_features, embeddings], dim=-1)
            gates = self.gate_linear(concat)  # [num_nodes, 1]
            fused = gates * time_features + (1 - gates) * embeddings
        else:
            # 多头版本
            q = self.query(time_features)
            k = self.key(embeddings)
            gates = torch.softmax(self.gate_linear(q + k), dim=-1)  # [num_nodes, num_heads]
            # 简化：使用平均
            gates = gates.mean(dim=-1, keepdim=True)  # [num_nodes, 1]
            fused = gates * time_features + (1 - gates) * embeddings

        return fused, gates
