"""
时间编码器组件
使用共享参数的Transformer编码器处理不同时间窗口的数据
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return x


class SharedTransformerEncoder(nn.Module):
    """
    共享参数的Transformer编码器
    用于处理不同时间窗口的数据（20日、40日、80日）
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1, max_seq_len: int = 100):
        """
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super(SharedTransformerEncoder, self).__init__()
        
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(7, d_model)  # 7个特征维度

        # ⭐ 添加 LayerNorm 稳定输入，防止数值不稳定
        self.input_norm = nn.LayerNorm(d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # 使用 (seq_len, batch, features) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, features]
            mask: 掩码张量，形状为 [batch_size, seq_len]，1表示有效，0表示无效
            
        Returns:
            编码后的特征向量，形状为 [batch_size, d_model]
        """
        batch_size, seq_len, features = x.shape

        # 投影到模型维度
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]

        # ⭐ 应用 LayerNorm 归一化
        x = self.input_norm(x)

        # 转换为 (seq_len, batch_size, d_model) 格式
        x = x.transpose(0, 1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 创建注意力掩码（如果提供了mask）
        src_mask = None
        if mask is not None:
            # 将mask转换为注意力掩码格式
            # mask: [batch_size, seq_len] -> [seq_len, batch_size]
            mask_t = mask.transpose(0, 1)  # [seq_len, batch_size]
            # 创建padding mask: False表示有效位置，True表示需要mask的位置
            src_key_padding_mask = (mask_t == 0)  # [seq_len, batch_size]
            src_key_padding_mask = src_key_padding_mask.transpose(0, 1)  # [batch_size, seq_len]
        else:
            src_key_padding_mask = None
        
        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 全局平均池化（考虑mask）
        if mask is not None:
            # 扩展mask维度以匹配encoded
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            mask_expanded = mask_expanded.transpose(0, 1)  # [seq_len, batch_size, 1]
            
            # 掩码平均池化
            encoded_masked = encoded * mask_expanded  # [seq_len, batch_size, d_model]
            sum_features = encoded_masked.sum(dim=0)  # [batch_size, d_model]
            sum_mask = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
            pooled = sum_features / (sum_mask + 1e-8)  # [batch_size, d_model]
        else:
            # 简单平均池化
            pooled = encoded.mean(dim=0)  # [batch_size, d_model]
        
        return pooled


class MultiScaleTimeEncoder(nn.Module):
    """
    多尺度时间编码器
    使用共享的Transformer编码器处理20日、40日、80日三个时间窗口
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
        """
        super(MultiScaleTimeEncoder, self).__init__()
        
        # 共享的Transformer编码器
        self.shared_encoder = SharedTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=100
        )
    
    def forward(self, x_20: torch.Tensor, x_40: torch.Tensor, x_80: torch.Tensor,
                mask_20: Optional[torch.Tensor] = None,
                mask_40: Optional[torch.Tensor] = None,
                mask_80: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对三个不同时间窗口的数据进行编码
        
        Args:
            x_20: 20日窗口数据，形状为 [batch_size, 20, features]
            x_40: 40日窗口数据，形状为 [batch_size, 40, features]
            x_80: 80日窗口数据，形状为 [batch_size, 80, features]
            mask_20, mask_40, mask_80: 对应的掩码
            
        Returns:
            三个编码后的特征向量，每个形状为 [batch_size, d_model]
        """
        # 使用共享编码器处理三个时间窗口
        encoded_20 = self.shared_encoder(x_20, mask_20)
        encoded_40 = self.shared_encoder(x_40, mask_40)
        encoded_80 = self.shared_encoder(x_80, mask_80)
        
        return encoded_20, encoded_40, encoded_80

