"""
数据加载和预处理组件
负责从JSON文件加载行业K线数据，并进行预处理
"""
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import torch


class IndustryDataLoader:
    """行业数据加载器"""
    
    def __init__(self, data_dir: str = None, data_path: str = None, relation_path: str = None,
                 window_sizes: List[int] = None, future_days: int = None, num_classes: int = None):
        """
        初始化数据加载器
        
        支持两种初始化方式：
        1. 新方式（推荐）：data_dir + 配置文件中的参数
        2. 旧方式：data_path + relation_path
        
        Args:
            data_dir: 数据目录（包含 kline_file 和 relation_file）
            data_path: industry_kline_data.json 文件路径（旧方式）
            relation_path: industry_relation.csv 文件路径（旧方式）
            window_sizes: 时间窗口大小列表 [20, 40, 80]（新方式）
            future_days: 预测未来天数（新方式）
            num_classes: 分类类别数（新方式，用于兼容）
        """
        # 新方式：使用 data_dir
        if data_dir is not None:
            from pathlib import Path
            data_dir = Path(data_dir)
            self.data_path = str(data_dir / "industry_kline_data.json")
            self.relation_path = str(data_dir / "industry_relation.csv")
        # 旧方式：直接指定路径
        elif data_path is not None and relation_path is not None:
            self.data_path = data_path
            self.relation_path = relation_path
        else:
            raise ValueError("必须提供 data_dir 或 (data_path, relation_path)")
        
        self.window_sizes = window_sizes if window_sizes is not None else [20, 40, 80]
        self.future_days = future_days if future_days is not None else 30
        self.num_classes = num_classes if num_classes is not None else 5
        
        self.raw_data = None
        self.relation_df = None
        self.industry_list = None
        self.industry_to_idx = {}
        self.adj_matrix = None
        
    def load_data(self):
        """加载原始数据"""
        # 加载K线数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 加载行业关系数据
        self.relation_df = pd.read_csv(self.relation_path)
        
        # 获取行业列表（按CSV中的顺序）
        self.industry_list = self.relation_df['industry'].tolist()
        
        # 创建行业到索引的映射
        self.industry_to_idx = {industry: idx for idx, industry in enumerate(self.industry_list)}
        
        print(f"加载了 {len(self.industry_list)} 个行业的数据")
        
    def parse_kline_data(self, industry_name: str) -> np.ndarray:
        """
        解析单个行业的K线数据
        
        Args:
            industry_name: 行业名称
            
        Returns:
            numpy数组，形状为 [时间步, 特征数]
            特征包括：开盘、收盘、最高、最低、成交量、成交额等
        """
        if industry_name not in self.raw_data:
            return None
        
        klines = self.raw_data[industry_name]
        if not klines:
            return None
        
        # 解析K线数据
        # 格式: ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", ...]
        data_list = []
        for kline in klines:
            try:
                # 提取数值特征（跳过日期字符串）
                date_str = kline[0]
                open_price = float(kline[1])
                close_price = float(kline[2])
                high_price = float(kline[3])
                low_price = float(kline[4])
                volume = float(kline[5])
                amount = float(kline[6])
                
                # 计算收益率
                if len(data_list) > 0:
                    prev_close = data_list[-1][1]  # 前一天的收盘价
                    return_rate = (close_price - prev_close) / prev_close if prev_close > 0 else 0.0
                else:
                    return_rate = 0.0
                
                # 特征向量：[开盘, 收盘, 最高, 最低, 成交量, 成交额, 收益率]
                features = [open_price, close_price, high_price, low_price, 
                           volume, amount, return_rate]
                data_list.append(features)
            except (ValueError, IndexError) as e:
                continue
        
        if not data_list:
            return None
        
        return np.array(data_list, dtype=np.float32)
    
    def build_adjacency_matrix(self) -> np.ndarray:
        """
        构建行业关系邻接矩阵
        
        Returns:
            邻接矩阵，形状为 [行业数, 行业数]
            如果两个行业属于同一个sw_industry，则为1，否则为0
        """
        n_industries = len(self.industry_list)
        adj_matrix = np.zeros((n_industries, n_industries), dtype=np.float32)
        
        # 根据sw_industry分组
        sw_groups = {}
        for idx, row in self.relation_df.iterrows():
            sw_industry = row['sw_industry']
            if sw_industry not in sw_groups:
                sw_groups[sw_industry] = []
            sw_groups[sw_industry].append(idx)
        
        # 同一组内的行业之间建立连接
        for group_indices in sw_groups.values():
            for i in group_indices:
                for j in group_indices:
                    if i != j:
                        adj_matrix[i, j] = 1.0
        
        # 添加自连接
        np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix
    
    def prepare_sequences(self, window_sizes: List[int] = [20, 40, 80], 
                          future_days: int = 30) -> Dict[str, np.ndarray]:
        """
        准备时间序列数据
        
        Args:
            window_sizes: 时间窗口大小列表 [20, 40, 80]
            future_days: 预测未来天数
            
        Returns:
            字典，包含：
            - 'sequences': [样本数, 最大窗口, 特征数] 的数组
            - 'targets': [样本数] 的目标值（未来30天收益率分位数）
            - 'masks': [样本数, 最大窗口] 的掩码，标记有效时间步
            - 'industry_indices': [样本数] 的行业索引
        """
        max_window = max(window_sizes)
        all_sequences = []
        all_targets = []
        all_masks = []
        all_industry_indices = []
        
        # 先收集所有行业的未来收益率，用于全局分位数计算
        all_future_returns = []
        
        for industry_idx, industry_name in enumerate(self.industry_list):
            data = self.parse_kline_data(industry_name)
            if data is None or len(data) < max_window + future_days:
                continue
            
            # 计算未来30天的收益率
            current_prices = data[:, 1]  # 收盘价
            
            for i in range(len(data) - max_window - future_days + 1):
                # 计算未来收益率
                start_price = current_prices[i+max_window-1]
                end_price = current_prices[i+max_window+future_days-1]
                future_return = (end_price - start_price) / start_price if start_price > 0 else 0.0
                all_future_returns.append(future_return)
        
        # 计算全局分位数阈值（5分位数）
        if len(all_future_returns) == 0:
            print("Warning: No valid sequences found!")
            return {
                'sequences': np.array([]),
                'targets': np.array([]),
                'masks': np.array([]),
                'industry_indices': np.array([])
            }
        
        all_future_returns_array = np.array(all_future_returns)
        quantiles = np.percentile(all_future_returns_array, [20, 40, 60, 80])
        
        # 再次遍历，创建序列和标签
        for industry_idx, industry_name in enumerate(self.industry_list):
            data = self.parse_kline_data(industry_name)
            if data is None or len(data) < max_window + future_days:
                continue
            
            current_prices = data[:, 1]  # 收盘价
            
            for i in range(len(data) - max_window - future_days + 1):
                # 提取序列
                seq = data[i:i+max_window]
                
                # 计算未来收益率
                start_price = current_prices[i+max_window-1]
                end_price = current_prices[i+max_window+future_days-1]
                future_return = (end_price - start_price) / start_price if start_price > 0 else 0.0
                
                # 分配分位数标签
                if future_return <= quantiles[0]:
                    target = 0  # 最低20%
                elif future_return <= quantiles[1]:
                    target = 1  # 20-40%
                elif future_return <= quantiles[2]:
                    target = 2  # 40-60%
                elif future_return <= quantiles[3]:
                    target = 3  # 60-80%
                else:
                    target = 4  # 最高20%
                
                all_sequences.append(seq)
                all_targets.append(target)
                all_industry_indices.append(industry_idx)
                
                # 创建掩码（全部有效，因为都是max_window长度）
                mask = np.ones(max_window, dtype=np.float32)
                all_masks.append(mask)
        
        return {
            'sequences': np.array(all_sequences),
            'targets': np.array(all_targets),
            'masks': np.array(all_masks),
            'industry_indices': np.array(all_industry_indices)
        }
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备数据（兼容 example_train.py 的接口）
        
        Returns:
            samples: [样本数, 最大窗口, 特征数] 的序列数组
            targets: [样本数] 的目标值数组
            adj_matrix: [行业数, 行业数] 的邻接矩阵
        """
        # 加载数据
        if self.raw_data is None:
            self.load_data()
        
        # 构建邻接矩阵
        if self.adj_matrix is None:
            self.adj_matrix = self.build_adjacency_matrix()
        
        # 准备序列（如果还没有准备过）
        if not hasattr(self, '_data_dict') or self._data_dict is None:
            self._data_dict = self.prepare_sequences(
                window_sizes=self.window_sizes,
                future_days=self.future_days
            )
        
        samples = self._data_dict['sequences']
        targets = self._data_dict['targets']
        
        return samples, targets, self.adj_matrix
    
    def get_data_dict(self) -> Dict[str, np.ndarray]:
        """
        获取完整的数据字典（包含sequences, targets, masks, industry_indices）
        
        Returns:
            包含所有数据的字典
        """
        # 确保数据已准备
        if not hasattr(self, '_data_dict') or self._data_dict is None:
            self.prepare_data()
        return self._data_dict


class IndustryDataset(Dataset):
    """PyTorch数据集类"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 masks: np.ndarray = None, industry_indices: np.ndarray = None):
        """
        Args:
            sequences: [样本数, 时间步, 特征数]
            targets: [样本数] 标签
            masks: [样本数, 时间步] 掩码（可选，如果为None则创建全1掩码）
            industry_indices: [样本数] 行业索引（可选，如果为None则创建零索引）
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
        
        # 如果没有提供掩码，创建全1掩码
        if masks is None:
            masks = np.ones((len(sequences), sequences.shape[1]), dtype=np.float32)
        self.masks = torch.FloatTensor(masks)
        
        # 如果没有提供行业索引，创建零索引（用于兼容旧接口）
        if industry_indices is None:
            industry_indices = np.zeros(len(sequences), dtype=np.int64)
        self.industry_indices = torch.LongTensor(industry_indices)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target': self.targets[idx],
            'mask': self.masks[idx],
            'industry_idx': self.industry_indices[idx]
        }

