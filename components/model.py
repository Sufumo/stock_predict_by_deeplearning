"""
主模型组件
整合DWT增强、多尺度Transformer编码器、动态门控和GAT
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dwt_enhancement import DWTEnhancement
from .time_encoder import MultiScaleTimeEncoder
from .dynamic_gate import DynamicAttentionGate
from .gat_layer import GAT, LearningCompressionLayer


class IndustryStockModel(nn.Module):
    """
    行业股票预测模型
    整合时间特征提取和行业关系建模
    """
    
    def __init__(self, 
                 input_features: int = 7,
                 time_encoder_dim: int = 128,
                 compression_dim: int = 64,
                 gat_hidden_dim: int = 128,
                 gat_output_dim: int = 64,
                 num_classes: int = 5,
                 num_heads: int = 8,
                 num_gat_layers: int = 2,
                 dropout: float = 0.1,
                 use_dwt: bool = True):
        """
        Args:
            input_features: 输入特征数（K线数据的特征维度）
            time_encoder_dim: 时间编码器输出维度
            compression_dim: LCL压缩后的维度
            gat_hidden_dim: GAT隐藏层维度
            gat_output_dim: GAT输出维度
            num_classes: 分类类别数（5分位数）
            num_heads: 注意力头数
            num_gat_layers: GAT层数
            dropout: Dropout比率
            use_dwt: 是否使用DWT增强
        """
        super(IndustryStockModel, self).__init__()
        
        self.input_features = input_features
        self.time_encoder_dim = time_encoder_dim
        self.compression_dim = compression_dim
        self.use_dwt = use_dwt
        
        # DWT增强模块
        if use_dwt:
            self.dwt_enhancement = DWTEnhancement()
        else:
            self.dwt_enhancement = None
        
        # 多尺度时间编码器（共享参数）
        self.time_encoder = MultiScaleTimeEncoder(
            d_model=time_encoder_dim,
            nhead=num_heads,
            num_layers=2,
            dim_feedforward=time_encoder_dim * 4,
            dropout=dropout
        )
        
        # 动态注意力门控机制
        self.dynamic_gate = DynamicAttentionGate(
            d_model=time_encoder_dim,
            hidden_dim=time_encoder_dim // 2,
            dropout=dropout
        )
        
        # 学习压缩层（LCL）
        self.compression_layer = LearningCompressionLayer(
            in_features=time_encoder_dim,
            out_features=compression_dim,
            dropout=dropout
        )
        
        # GAT图注意力网络
        self.gat = GAT(
            in_features=compression_dim,
            hidden_features=gat_hidden_dim,
            out_features=gat_output_dim,
            num_heads=num_heads,
            num_layers=num_gat_layers,
            dropout=dropout
        )
        
        # 备用MLP（当没有GAT时使用）
        self.fallback_mlp = nn.Sequential(
            nn.Linear(compression_dim, gat_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 预测头（5分位数分类）
        # 注意:移除Softmax,因为CrossEntropyLoss已包含LogSoftmax
        self.predictor = nn.Sequential(
            nn.Linear(gat_output_dim, gat_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_output_dim // 2, num_classes)
        )
    
    def forward(self, x_20: torch.Tensor, x_40: torch.Tensor, x_80: torch.Tensor,
                mask_20: Optional[torch.Tensor] = None,
                mask_40: Optional[torch.Tensor] = None,
                mask_80: Optional[torch.Tensor] = None,
                adj_matrix: Optional[torch.Tensor] = None,
                industry_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_20: 20日窗口数据，形状为 [batch_size, 20, input_features]
            x_40: 40日窗口数据，形状为 [batch_size, 40, input_features]
            x_80: 80日窗口数据，形状为 [batch_size, 80, input_features]
            mask_20, mask_40, mask_80: 对应的掩码
            adj_matrix: 邻接矩阵，形状为 [num_industries, num_industries]
            industry_indices: 行业索引，形状为 [batch_size]
            
        Returns:
            - 预测概率，形状为 [batch_size, num_classes]
            - 行业特征，形状为 [batch_size, gat_output_dim]
        """
        batch_size = x_20.shape[0]
        
        # 第一阶段：时间特征提取
        # 1. DWT增强（如果启用）
        if self.dwt_enhancement is not None:
            x_20_enhanced = self.dwt_enhancement(x_20)
            x_40_enhanced = self.dwt_enhancement(x_40)
            x_80_enhanced = self.dwt_enhancement(x_80)
        else:
            x_20_enhanced = x_20
            x_40_enhanced = x_40
            x_80_enhanced = x_80
        
        # 2. 多尺度Transformer编码
        feat_20, feat_40, feat_80 = self.time_encoder(
            x_20_enhanced, x_40_enhanced, x_80_enhanced,
            mask_20, mask_40, mask_80
        )
        
        # 3. 动态门控融合
        time_features = self.dynamic_gate(feat_20, feat_40, feat_80)  # [batch_size, time_encoder_dim]
        
        # 第二阶段：行业关系建模
        # 4. 信息压缩（LCL）
        compressed_features = self.compression_layer(time_features)  # [batch_size, compression_dim]
        
        # 5. GAT处理 - 子图采样模式
        # 只处理batch内的行业及其邻居,避免大量零特征问题
        if adj_matrix is not None and industry_indices is not None:
            # 提取子图: batch中的行业 + 它们的1跳邻居
            batch_gat_features = self._process_subgraph(
                compressed_features,
                industry_indices,
                adj_matrix
            )
        else:
            # 如果没有提供邻接矩阵，使用备用MLP
            batch_gat_features = self.fallback_mlp(compressed_features)
        
        # 6. 预测
        predictions = self.predictor(batch_gat_features)  # [batch_size, num_classes]
        
        return predictions, batch_gat_features

    def _process_subgraph(self, compressed_features: torch.Tensor,
                         industry_indices: torch.Tensor,
                         adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        子图采样模式的GAT处理
        只处理batch中的行业及其1跳邻居,提高效率并避免零特征问题

        Args:
            compressed_features: [batch_size, compression_dim]
            industry_indices: [batch_size] batch中各样本对应的行业索引
            adj_matrix: [num_industries, num_industries] 完整邻接矩阵

        Returns:
            batch_gat_features: [batch_size, gat_output_dim]
        """
        batch_size = compressed_features.shape[0]
        device = compressed_features.device

        # 获取batch中的唯一行业索引
        unique_indices = torch.unique(industry_indices)

        # 找到这些行业的1跳邻居
        # 对于每个batch中的行业,找到所有与它相连的行业
        batch_and_neighbors = set(unique_indices.cpu().tolist())
        for idx in unique_indices:
            # 找到与idx相连的行业(邻接矩阵中非零元素)
            neighbors = torch.where(adj_matrix[idx] > 0)[0]
            batch_and_neighbors.update(neighbors.cpu().tolist())

        # 转换为排序的列表和张量
        subgraph_nodes = sorted(list(batch_and_neighbors))
        subgraph_nodes_tensor = torch.tensor(subgraph_nodes, device=device)
        num_subgraph_nodes = len(subgraph_nodes)

        # 创建子图特征矩阵
        subgraph_features = torch.zeros(
            num_subgraph_nodes, self.compression_dim, device=device
        )

        # 创建从原始索引到子图索引的映射
        index_mapping = {orig_idx: sub_idx for sub_idx, orig_idx in enumerate(subgraph_nodes)}

        # 填充batch中行业的特征
        for i, orig_idx in enumerate(industry_indices):
            sub_idx = index_mapping[orig_idx.item()]
            subgraph_features[sub_idx] = compressed_features[i]

        # 对于邻居节点,使用零向量(或可选:使用历史平均特征)
        # 这里保持零向量,因为邻居只是提供结构信息

        # 提取子图的邻接矩阵
        subgraph_adj = adj_matrix[subgraph_nodes_tensor][:, subgraph_nodes_tensor]

        # 在子图上运行GAT
        gat_output = self.gat(subgraph_features, subgraph_adj)  # [num_subgraph_nodes, gat_output_dim]

        # 提取batch中行业的输出特征
        batch_gat_features = torch.zeros(
            batch_size, gat_output.shape[1], device=device
        )
        for i, orig_idx in enumerate(industry_indices):
            sub_idx = index_mapping[orig_idx.item()]
            batch_gat_features[i] = gat_output[sub_idx]

        return batch_gat_features

    def extract_time_features(self, x_20: torch.Tensor, x_40: torch.Tensor, 
                             x_80: torch.Tensor,
                             mask_20: Optional[torch.Tensor] = None,
                             mask_40: Optional[torch.Tensor] = None,
                             mask_80: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅提取时间特征（用于分析或可视化）
        
        Returns:
            时间特征向量，形状为 [batch_size, time_encoder_dim]
        """
        # DWT增强
        if self.dwt_enhancement is not None:
            x_20_enhanced = self.dwt_enhancement(x_20)
            x_40_enhanced = self.dwt_enhancement(x_40)
            x_80_enhanced = self.dwt_enhancement(x_80)
        else:
            x_20_enhanced = x_20
            x_40_enhanced = x_40
            x_80_enhanced = x_80
        
        # 编码
        feat_20, feat_40, feat_80 = self.time_encoder(
            x_20_enhanced, x_40_enhanced, x_80_enhanced,
            mask_20, mask_40, mask_80
        )
        
        # 融合
        time_features = self.dynamic_gate(feat_20, feat_40, feat_80)
        
        return time_features

