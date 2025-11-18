"""
图注意力网络（GAT）组件
用于建模行业间的动态关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """
    图注意力层（GAT Layer）
    实现多头注意力机制
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 8,
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
            alpha: LeakyReLU的负斜率
            concat: 是否拼接多头输出
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # 每个头的输出维度
        self.head_dim = out_features // num_heads if concat else out_features
        
        # 线性变换层（每个头）
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力权重参数
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: 节点特征矩阵，形状为 [num_nodes, in_features]
            adj: 邻接矩阵，形状为 [num_nodes, num_nodes]
            
        Returns:
            更新后的节点特征，形状为 [num_nodes, out_features]
        """
        num_nodes, in_features = h.shape
        
        # 对每个头进行处理
        head_outputs = []
        
        for head in range(self.num_heads):
            # 线性变换
            Wh = torch.mm(h, self.W[head])  # [num_nodes, head_dim]
            
            # 计算注意力分数 e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            # 使用广播机制计算所有节点对的注意力分数
            Wh1 = torch.mm(Wh, self.a[head][:self.head_dim, :])  # [num_nodes, 1]
            Wh2 = torch.mm(Wh, self.a[head][self.head_dim:, :])  # [num_nodes, 1]
            
            # 广播计算 e_ij = Wh1_i + Wh2_j
            e = Wh1 + Wh2.T  # [num_nodes, num_nodes]
            e = self.leakyrelu(e)
            
            # 应用邻接矩阵掩码（只保留有连接的节点对）
            # 将无连接的节点对的注意力分数设为负无穷
            attention_mask = (adj > 0).float()
            e = e.masked_fill(attention_mask == 0, float('-inf'))
            
            # Softmax归一化
            attention = F.softmax(e, dim=1)
            attention = self.dropout_layer(attention)
            
            # 应用注意力权重
            h_prime = torch.mm(attention, Wh)  # [num_nodes, head_dim]
            head_outputs.append(h_prime)
        
        # 拼接或平均多头输出
        if self.concat:
            h_out = torch.cat(head_outputs, dim=1)  # [num_nodes, out_features]
        else:
            h_out = torch.mean(torch.stack(head_outputs), dim=0)  # [num_nodes, out_features]
        
        return h_out


class GAT(nn.Module):
    """
    图注意力网络（GAT）
    多层GAT堆叠，用于学习行业间的复杂关系
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            num_layers: GAT层数
            dropout: Dropout比率
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(
            GraphAttentionLayer(in_features, hidden_features, num_heads, dropout, concat=True)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                GraphAttentionLayer(hidden_features, hidden_features, num_heads, dropout, concat=True)
            )
        
        # 最后一层（不拼接，直接输出）
        if num_layers > 1:
            self.layers.append(
                GraphAttentionLayer(hidden_features, out_features, num_heads, dropout, concat=False)
            )
        else:
            # 如果只有一层，直接输出
            self.layers[0] = GraphAttentionLayer(
                in_features, out_features, num_heads, dropout, concat=False
            )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵，形状为 [num_nodes, in_features] 或 [batch_size, num_nodes, in_features]
            adj: 邻接矩阵，形状为 [num_nodes, num_nodes] 或 [batch_size, num_nodes, num_nodes]
            
        Returns:
            更新后的节点特征，形状为 [num_nodes, out_features] 或 [batch_size, num_nodes, out_features]
        """
        h = x
        
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            
            # 除了最后一层，都应用激活函数和dropout
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        
        return h


class LearningCompressionLayer(nn.Module):
    """
    学习压缩层（LCL）
    对高维特征进行降维，创建紧凑的节点特征
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度（压缩后的维度）
            dropout: Dropout比率
        """
        super(LearningCompressionLayer, self).__init__()
        
        self.compression = nn.Sequential(
            nn.Linear(in_features, (in_features + out_features) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((in_features + out_features) // 2, out_features),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征，形状为 [..., in_features]
            
        Returns:
            压缩后的特征，形状为 [..., out_features]
        """
        return self.compression(x)

