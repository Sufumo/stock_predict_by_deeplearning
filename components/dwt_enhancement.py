"""
离散小波变换（DWT）信号增强组件
用于对时间序列数据进行去噪和特征提取
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: pywt not available. DWT enhancement will use simplified version.")


class DWTEnhancement(nn.Module):
    """
    离散小波变换增强模块
    对输入时间序列进行小波分解，过滤高频噪声
    """
    
    def __init__(self, wavelet: str = 'db4', mode: str = 'symmetric', 
                 use_pywt: bool = True):
        """
        Args:
            wavelet: 小波基函数，默认'db4'（Daubechies 4）
            mode: 边界处理模式
            use_pywt: 是否使用pywt库（如果可用）
        """
        super(DWTEnhancement, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.use_pywt = use_pywt and PYWT_AVAILABLE
        
        if not self.use_pywt:
            # 如果pywt不可用，使用简化的移动平均滤波
            print("Using simplified moving average filter instead of DWT")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入序列进行DWT增强
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, features] 或 [seq_len, features]
            
        Returns:
            增强后的张量，形状与输入相同
        """
        if self.use_pywt:
            return self._dwt_enhance(x)
        else:
            return self._simple_filter(x)
    
    def _dwt_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """使用pywt进行DWT增强"""
        original_shape = x.shape
        is_2d = len(original_shape) == 2
        
        if is_2d:
            x = x.unsqueeze(0)  # [1, seq_len, features]
        
        batch_size, seq_len, features = x.shape
        enhanced = []
        
        # 转换为numpy进行处理
        x_np = x.detach().cpu().numpy()
        
        for b in range(batch_size):
            batch_enhanced = []
            for f in range(features):
                signal = x_np[b, :, f]
                
                try:
                    # 进行小波分解
                    coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode)
                    
                    # 阈值去噪（使用软阈值）
                    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(signal)))
                    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') 
                                    for c in coeffs]
                    
                    # 重构信号
                    enhanced_signal = pywt.waverec(coeffs_thresh, self.wavelet, mode=self.mode)
                    
                    # 确保长度一致
                    if len(enhanced_signal) > seq_len:
                        enhanced_signal = enhanced_signal[:seq_len]
                    elif len(enhanced_signal) < seq_len:
                        enhanced_signal = np.pad(enhanced_signal, 
                                               (0, seq_len - len(enhanced_signal)), 
                                               mode='edge')
                    
                    batch_enhanced.append(enhanced_signal)
                except Exception as e:
                    # 如果DWT失败，使用原始信号
                    batch_enhanced.append(signal)
            
            enhanced.append(np.stack(batch_enhanced, axis=1))
        
        enhanced_tensor = torch.FloatTensor(np.array(enhanced)).to(x.device)
        
        if is_2d:
            enhanced_tensor = enhanced_tensor.squeeze(0)
        
        return enhanced_tensor
    
    def _simple_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        简化的移动平均滤波器（当pywt不可用时使用）
        使用一维卷积实现移动平均
        """
        original_shape = x.shape
        is_2d = len(original_shape) == 2
        
        if is_2d:
            x = x.unsqueeze(0)  # [1, seq_len, features]
        
        batch_size, seq_len, features = x.shape
        
        # 使用移动平均窗口大小为3
        kernel_size = 3
        padding = kernel_size // 2
        
        # 对每个特征维度进行滤波
        enhanced_list = []
        for f in range(features):
            feature_data = x[:, :, f].unsqueeze(1)  # [batch, 1, seq_len]
            
            # 创建移动平均核
            kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
            
            # 应用卷积（移动平均）
            filtered = torch.nn.functional.conv1d(
                feature_data, kernel, padding=padding
            )
            
            # 裁剪到原始长度
            filtered = filtered[:, :, :seq_len]
            enhanced_list.append(filtered.squeeze(1))
        
        enhanced = torch.stack(enhanced_list, dim=2)  # [batch, seq_len, features]
        
        if is_2d:
            enhanced = enhanced.squeeze(0)
        
        return enhanced

