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
from .node_level_gate import NodeLevelGate, GlobalGate


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
                 use_dwt: bool = True,
                 num_industries: int = 86,
                 use_industry_embedding: bool = True,
                 embedding_fusion_alpha: float = 1.0,
                 use_node_gate: bool = False,
                 gate_hidden_dim: int = 64,
                 cross_sectional_mode: bool = False):
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
            num_industries: 行业总数
            use_industry_embedding: 是否使用可学习的行业嵌入
            embedding_fusion_alpha: 时间特征融合权重(1.0=完全使用时间特征,0.0=完全使用嵌入)
            use_node_gate: 是否使用节点级门控（自适应融合）
            gate_hidden_dim: 门控MLP隐藏层维度
            cross_sectional_mode: 是否使用横截面局部训练模式
        """
        super(IndustryStockModel, self).__init__()

        self.input_features = input_features
        self.time_encoder_dim = time_encoder_dim
        self.compression_dim = compression_dim
        self.use_dwt = use_dwt
        self.num_industries = num_industries
        self.use_industry_embedding = use_industry_embedding
        self.embedding_fusion_alpha = embedding_fusion_alpha
        self.use_node_gate = use_node_gate
        self.cross_sectional_mode = cross_sectional_mode
        
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

        # ⭐ 可学习的行业嵌入层
        # 为每个行业学习一个固定的嵌入向量,用于邻居节点特征
        if use_industry_embedding:
            self.industry_embeddings = nn.Embedding(num_industries, compression_dim)
            # 使用小随机值初始化
            nn.init.normal_(self.industry_embeddings.weight, mean=0.0, std=0.01)
        else:
            self.industry_embeddings = None

        # ⭐ 节点级门控机制（横截面局部训练模式）
        if use_node_gate and use_industry_embedding:
            self.node_gate = NodeLevelGate(
                feature_dim=compression_dim,
                hidden_dim=gate_hidden_dim,
                dropout=dropout
            )
        else:
            self.node_gate = None

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
                industry_indices: Optional[torch.Tensor] = None,
                node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            x_20: 20日窗口数据，形状为 [batch_size, 20, input_features]
            x_40: 40日窗口数据，形状为 [batch_size, 40, input_features]
            x_80: 80日窗口数据，形状为 [batch_size, 80, input_features]
            mask_20, mask_40, mask_80: 对应的掩码
            adj_matrix: 邻接矩阵，形状为 [num_industries, num_industries]
            industry_indices: 行业索引，形状为 [batch_size]
            node_mask: [num_industries] bool tensor, True=有时间特征输入, False=掩码节点
                      仅在cross_sectional_mode=True时使用

        Returns:
            - 预测概率，形状为 [batch_size, num_classes]
            - 行业特征，形状为 [batch_size, gat_output_dim]
            - 门控值，形状为 [num_active_nodes, 1]，如果不使用门控则为None
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
        
        # 5. GAT处理 - 根据模式选择处理方式
        gates = None  # 门控值（用于分析）

        if adj_matrix is not None and industry_indices is not None:
            if self.cross_sectional_mode and node_mask is not None:
                # 横截面局部训练模式：完整86节点图 + 部分节点有输入
                batch_gat_features, gates = self._process_cross_sectional_subgraph(
                    compressed_features,
                    industry_indices,
                    adj_matrix,
                    node_mask
                )
            else:
                # 传统子图采样模式
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

        return predictions, batch_gat_features, gates

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

        # ⭐ 创建子图特征矩阵 - 使用行业嵌入初始化
        if self.use_industry_embedding and self.industry_embeddings is not None:
            # 使用行业嵌入初始化所有节点(包括batch节点和邻居节点)
            subgraph_embeddings = self.industry_embeddings(subgraph_nodes_tensor)
            subgraph_features = subgraph_embeddings.clone()
        else:
            # 如果不使用嵌入,初始化为零向量
            subgraph_features = torch.zeros(
                num_subgraph_nodes, self.compression_dim, device=device
            )

        # 创建从原始索引到子图索引的映射
        index_mapping = {orig_idx: sub_idx for sub_idx, orig_idx in enumerate(subgraph_nodes)}

        # ⭐ 填充batch中行业的特征 - 可选融合或替换
        for i, orig_idx in enumerate(industry_indices):
            sub_idx = index_mapping[orig_idx.item()]

            if self.use_industry_embedding and self.embedding_fusion_alpha < 1.0:
                # 融合模式: 时间特征 + 行业嵌入
                alpha = self.embedding_fusion_alpha
                subgraph_features[sub_idx] = (
                    alpha * compressed_features[i] +
                    (1 - alpha) * subgraph_features[sub_idx]
                )
            else:
                # 替换模式: 完全使用时间特征 (默认)
                subgraph_features[sub_idx] = compressed_features[i]

        # 邻居节点保持行业嵌入特征(或零向量)

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

    def _process_cross_sectional_subgraph(self, compressed_features: torch.Tensor,
                                          industry_indices: torch.Tensor,
                                          adj_matrix: torch.Tensor,
                                          node_mask: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        横截面局部训练模式的GAT处理
        构建完整的86节点图，部分节点有时间特征输入，其余节点使用纯嵌入

        Args:
            compressed_features: [batch_size, compression_dim] 有输入节点的时间特征
            industry_indices: [batch_size] 有输入节点的行业索引
            adj_matrix: [86, 86] 完整邻接矩阵
            node_mask: [86] bool tensor, True=有时间特征输入, False=掩码节点

        Returns:
            batch_gat_features: [batch_size, gat_output_dim]
            gates: [num_active_nodes, 1] 门控值（如果使用节点级门控）
        """
        batch_size = compressed_features.shape[0]
        device = compressed_features.device
        num_industries = self.num_industries

        # 1. 初始化所有86个节点的嵌入
        all_node_indices = torch.arange(num_industries, device=device)
        all_embeddings = self.industry_embeddings(all_node_indices)  # [86, compression_dim]

        # 2. 创建时间特征矩阵（初始化为零）
        all_time_features = torch.zeros(num_industries, self.compression_dim, device=device)

        # 3. 填充有输入节点的时间特征
        for i, idx in enumerate(industry_indices):
            all_time_features[idx] = compressed_features[i]

        # 4. 根据是否使用节点级门控，决定融合方式
        gates = None

        if self.use_node_gate and self.node_gate is not None:
            # 使用节点级门控融合
            # 准备输入：有输入的节点和掩码节点分别处理
            fused_features = all_embeddings.clone()  # 先用嵌入初始化

            # 找到有输入的节点索引
            active_node_indices = torch.where(node_mask)[0]  # [num_active]

            if len(active_node_indices) > 0:
                # 对有输入的节点应用门控融合
                active_time_features = all_time_features[active_node_indices]  # [num_active, dim]
                active_embeddings = all_embeddings[active_node_indices]  # [num_active, dim]

                # 门控融合
                fused_active, gates = self.node_gate(
                    active_time_features,
                    active_embeddings,
                    return_gates=True
                )  # [num_active, dim], [num_active, 1]

                # 更新有输入节点的特征
                fused_features[active_node_indices] = fused_active

            # 掩码节点保持纯嵌入（已经在clone时设置）
            final_features = fused_features

        else:
            # 使用固定alpha融合
            # 有输入的节点：alpha * 时间特征 + (1-alpha) * 嵌入
            # 掩码节点：纯嵌入
            alpha = self.embedding_fusion_alpha

            # 创建alpha mask
            alpha_mask = node_mask.float().unsqueeze(1)  # [86, 1]

            # 融合
            final_features = (
                alpha_mask * alpha * all_time_features +
                alpha_mask * (1 - alpha) * all_embeddings +
                (1 - alpha_mask) * all_embeddings
            )

        # 5. 在完整的86节点图上运行GAT
        gat_output = self.gat(final_features, adj_matrix)  # [86, gat_output_dim]

        # 6. 提取batch中节点的输出
        batch_gat_features = gat_output[industry_indices]  # [batch_size, gat_output_dim]

        return batch_gat_features, gates

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

