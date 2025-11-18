"""
动态注意力门控机制（DAGM）组件
用于融合多尺度时间特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAttentionGate(nn.Module):
    """
    动态注意力门控机制（DAGM）
    根据当前特征输入动态计算不同时间尺度的权重
    """
    
    def __init__(self, d_model: int = 128, hidden_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            d_model: 输入特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(DynamicAttentionGate, self).__init__()
        
        self.d_model = d_model
        
        # 门控网络：根据三个特征向量计算权重
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # 输出3个权重（对应20D、40D、80D）
            nn.Softmax(dim=-1)
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, feat_20: torch.Tensor, feat_40: torch.Tensor, 
                feat_80: torch.Tensor) -> torch.Tensor:
        """
        动态融合三个时间尺度的特征
        
        Args:
            feat_20: 20日窗口特征，形状为 [batch_size, d_model]
            feat_40: 40日窗口特征，形状为 [batch_size, d_model]
            feat_80: 80日窗口特征，形状为 [batch_size, d_model]
            
        Returns:
            融合后的特征向量，形状为 [batch_size, d_model]
        """
        batch_size = feat_20.shape[0]
        
        # 拼接三个特征向量
        concat_feat = torch.cat([feat_20, feat_40, feat_80], dim=1)  # [batch_size, d_model*3]
        
        # 计算动态权重
        weights = self.gate_network(concat_feat)  # [batch_size, 3]
        
        # 应用权重
        weighted_feat_20 = feat_20 * weights[:, 0:1]  # [batch_size, d_model]
        weighted_feat_40 = feat_40 * weights[:, 1:2]
        weighted_feat_80 = feat_80 * weights[:, 2:3]
        
        # 加权求和
        weighted_sum = weighted_feat_20 + weighted_feat_40 + weighted_feat_80
        
        # 进一步融合（使用融合层）
        fused_feat = self.fusion_layer(concat_feat)  # [batch_size, d_model]
        
        # 残差连接
        output = weighted_sum + fused_feat
        
        return output

