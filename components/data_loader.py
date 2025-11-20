"""
数据加载和预处理组件
负责从JSON文件加载行业K线数据，并进行预处理
"""
import json
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import torch


class StandardScaler:
    """
    标准化器 - 将特征标准化为均值0，标准差1
    支持分组归一化（不同特征组使用不同的均值和标准差）
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data: np.ndarray):
        """
        拟合标准化器

        Args:
            data: shape [N, D] 的数组，N是样本数，D是特征数
        """
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        # 防止除零错误，对于std=0的特征，设置为1
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        self.fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        应用标准化

        Args:
            data: shape [N, D] 或 [..., D] 的数组

        Returns:
            标准化后的数据
        """
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """逆变换，恢复原始尺度"""
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted.")

        return data * self.std + self.mean

    def get_params(self) -> dict:
        """获取参数"""
        return {
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }

    def set_params(self, params: dict):
        """设置参数"""
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = params['fitted']


class IndustryDataLoader:
    """行业数据加载器"""
    
    def __init__(self, data_dir: str = None, data_path: str = None, relation_path: str = None,
                 industry_list_path: str = None, window_sizes: List[int] = None,
                 future_days: int = None, num_classes: int = None):
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
        self.data_path = None
        self.relation_path = None
        self.industry_list_path = industry_list_path
        
        # 新方式：使用 data_dir
        if data_dir is not None:
            base_dir = Path(data_dir)
            self.data_path = self._resolve_data_file(
                base_dir,
                ["industry_kline_data_cleaned.json", "industry_kline_data.json"],
                "industry_kline_data.json"
            )
            self.relation_path = self._resolve_data_file(
                base_dir,
                ["industry_relation_cleaned.csv", "industry_relation.csv"],
                "industry_relation.csv"
            )
            if self.industry_list_path is None:
                industry_list_file = base_dir / "industry_list.json"
                if industry_list_file.exists():
                    self.industry_list_path = str(industry_list_file)
        # 旧方式：直接指定路径
        elif data_path is not None and relation_path is not None:
            self.data_path = data_path
            self.relation_path = relation_path
        else:
            raise ValueError("必须提供 data_dir 或 (data_path, relation_path)")
        
        if self.industry_list_path is None:
            raise ValueError("必须提供 industry_list.json，用于构建样本的行业列表")
        
        self.window_sizes = window_sizes if window_sizes is not None else [20, 40, 80]
        self.future_days = future_days if future_days is not None else 30
        self.num_classes = num_classes if num_classes is not None else 5

        self.raw_data = None
        self.relation_df = None
        self.industry_list = None
        self.industry_to_idx = {}
        self.adj_matrix = None

        # 特征标准化器（分组归一化）
        # 特征索引：0-开盘, 1-收盘, 2-最高, 3-最低, 4-成交量, 5-成交额, 6-收益率
        self.scaler_price = StandardScaler()     # 价格特征 [0, 1, 2, 3]
        self.scaler_volume = StandardScaler()    # 成交量 [4]
        self.scaler_amount = StandardScaler()    # 成交额 [5]
        # 收益率 [6] 保持原始值，不归一化（已经在合理范围）
        self.scalers_fitted = False
        
    @staticmethod
    def _resolve_data_file(base_dir: Path, candidates: List[str], default_name: str) -> str:
        """
        在候选列表中查找存在的文件，若都不存在，则返回默认名称对应的路径
        """
        for name in candidates:
            if not name:
                continue
            candidate_path = base_dir / name
            if candidate_path.exists():
                return str(candidate_path)
        return str(base_dir / default_name)

    def load_data(self):
        """加载原始数据"""
        # 加载K线数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 加载行业列表（用于确定GAT节点和样本行业）
        with open(self.industry_list_path, 'r', encoding='utf-8') as f:
            industry_list_raw = json.load(f)
        if not isinstance(industry_list_raw, list):
            raise ValueError("industry_list.json 格式必须为字符串列表")
        
        seen = set()
        self.industry_list = []
        for name in industry_list_raw:
            if not isinstance(name, str):
                continue
            if name not in seen:
                self.industry_list.append(name)
                seen.add(name)
        
        # 创建行业到索引的映射
        self.industry_to_idx = {industry: idx for idx, industry in enumerate(self.industry_list)}
        
        # 加载行业关系数据（用于构建静态GAT图）
        self.relation_df = pd.read_csv(self.relation_path)
        
        print(f"加载了 {len(self.industry_list)} 个行业的数据（来自 industry_list.json）")
        print(f"读取到 {len(self.relation_df)} 条行业关系（来自 industry_relation_cleaned.csv）")
        
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

    def fit_scalers(self):
        """
        拟合所有特征的标准化器
        需要先遍历所有行业数据，收集所有特征值
        """
        print("正在拟合特征标准化器...")

        all_price_features = []    # [开盘, 收盘, 最高, 最低]
        all_volume_features = []   # [成交量]
        all_amount_features = []   # [成交额]

        for industry_name in self.industry_list:
            data = self.parse_kline_data(industry_name)
            if data is None or len(data) == 0:
                continue

            # data shape: [时间步, 7特征]
            # 特征索引：0-开盘, 1-收盘, 2-最高, 3-最低, 4-成交量, 5-成交额, 6-收益率
            all_price_features.append(data[:, :4])     # 价格 [0:4]
            all_volume_features.append(data[:, 4:5])   # 成交量 [4:5]
            all_amount_features.append(data[:, 5:6])   # 成交额 [5:6]

        # 合并所有行业的数据
        all_price_features = np.concatenate(all_price_features, axis=0)
        all_volume_features = np.concatenate(all_volume_features, axis=0)
        all_amount_features = np.concatenate(all_amount_features, axis=0)

        # 拟合标准化器
        self.scaler_price.fit(all_price_features)
        self.scaler_volume.fit(all_volume_features)
        self.scaler_amount.fit(all_amount_features)
        self.scalers_fitted = True

        # 打印归一化前的统计信息
        print(f"\n归一化前的特征统计:")
        print(f"  价格特征 - Mean: {self.scaler_price.mean.flatten()[:4]}")
        print(f"  价格特征 - Std:  {self.scaler_price.std.flatten()[:4]}")
        print(f"  成交量   - Mean: {self.scaler_volume.mean.item():.2e}, Std: {self.scaler_volume.std.item():.2e}")
        print(f"  成交额   - Mean: {self.scaler_amount.mean.item():.2e}, Std: {self.scaler_amount.std.item():.2e}")

        # 验证归一化后的范围
        normalized_prices = self.scaler_price.transform(all_price_features)
        normalized_volume = self.scaler_volume.transform(all_volume_features)
        normalized_amount = self.scaler_amount.transform(all_amount_features)

        print(f"\n归一化后的特征范围:")
        print(f"  价格特征 - Min: {normalized_prices.min(axis=0)}, Max: {normalized_prices.max(axis=0)}")
        print(f"  成交量   - Min: {normalized_volume.min():.2f}, Max: {normalized_volume.max():.2f}")
        print(f"  成交额   - Min: {normalized_amount.min():.2f}, Max: {normalized_amount.max():.2f}")
        print()

    def normalize_features(self, data: np.ndarray) -> np.ndarray:
        """
        对单个序列的特征进行归一化

        Args:
            data: shape [时间步, 7特征] 的原始数据

        Returns:
            归一化后的数据，shape不变
        """
        if not self.scalers_fitted:
            raise RuntimeError("Scalers have not been fitted. Call fit_scalers() first.")

        # 分别归一化不同的特征组
        normalized_data = data.copy()
        normalized_data[:, :4] = self.scaler_price.transform(data[:, :4])      # 价格
        normalized_data[:, 4:5] = self.scaler_volume.transform(data[:, 4:5])   # 成交量
        normalized_data[:, 5:6] = self.scaler_amount.transform(data[:, 5:6])   # 成交额
        # 收益率 data[:, 6:7] 保持不变

        return normalized_data

    def build_adjacency_matrix(self) -> np.ndarray:
        """
        构建行业关系邻接矩阵（静态GAT图）
        
        Returns:
            邻接矩阵，形状为 [行业数, 行业数]
            仅使用 industry_relation_cleaned.csv 中的边信息构图
        """
        # ⭐ 确保数据已加载
        if self.industry_list is None:
            self.load_data()
        
        n_industries = len(self.industry_list)
        adj_matrix = np.eye(n_industries, dtype=np.float32)
        
        if self.relation_df is None or n_industries == 0:
            return adj_matrix
        
        required_cols = {'industry', 'sw_industry'}
        if not required_cols.issubset(self.relation_df.columns):
            raise ValueError("industry_relation_cleaned.csv 必须包含 'industry' 和 'sw_industry' 两列")
        
        seen_edges = set()
        valid_edges = 0
        
        for _, row in self.relation_df.iterrows():
            src = row['industry']
            dst = row['sw_industry']
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src not in self.industry_to_idx or dst not in self.industry_to_idx:
                continue
            i = self.industry_to_idx[src]
            j = self.industry_to_idx[dst]
            if i == j:
                continue
            edge = tuple(sorted((i, j)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
            valid_edges += 1
        
        if valid_edges == 0:
            print("Warning: 未在 industry_relation_cleaned.csv 中找到有效的行业关系，将仅使用对角自连接。")
        
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
        # 如果标准化器还未拟合，先拟合
        if not self.scalers_fitted:
            self.fit_scalers()

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

                # ⭐ 应用特征归一化
                seq_normalized = self.normalize_features(seq)

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

                all_sequences.append(seq_normalized)  # 使用归一化后的序列
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
        准备数据（兼容 train.py 的接口）
        
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
    
    def prepare_cross_sectional_data(self, window_sizes: List[int] = [20, 40, 80],
                                     future_days: int = 30) -> List[Dict[str, np.ndarray]]:
        """
        准备横截面数据：每个时间步包含所有86个行业的数据
        
        用于回测：每个时间步生成一个batch，包含所有行业在该时间步的数据
        
        Args:
            window_sizes: 时间窗口大小列表 [20, 40, 80]
            future_days: 预测未来天数
        
        Returns:
            时间步列表，每个元素是一个字典，包含：
            - 'sequences': [86, 最大窗口, 特征数] 的数组（86个行业）
            - 'targets': [86] 的目标值
            - 'masks': [86, 最大窗口] 的掩码
            - 'industry_indices': [86] 的行业索引（0-85）
            - 'time_index': 时间步索引（用于计算未来收益率）
        """
        # ⭐ 确保数据已加载（fit_scalers需要industry_list）
        if self.raw_data is None or self.industry_list is None:
            self.load_data()
        
        # 如果标准化器还未拟合，先拟合
        if not self.scalers_fitted:
            self.fit_scalers()
        
        max_window = max(window_sizes)
        
        # 先收集所有行业的数据，找到最小公共时间范围
        industry_data_dict = {}
        min_length = float('inf')
        
        for industry_idx, industry_name in enumerate(self.industry_list):
            data = self.parse_kline_data(industry_name)
            if data is None or len(data) < max_window + future_days:
                continue
            industry_data_dict[industry_idx] = data
            min_length = min(min_length, len(data))
        
        if min_length == float('inf'):
            print("Warning: No valid industry data found!")
            return []
        
        # 计算全局分位数阈值（用于标签分配）
        all_future_returns = []
        for industry_idx, data in industry_data_dict.items():
            current_prices = data[:, 1]  # 收盘价
            for i in range(len(data) - max_window - future_days + 1):
                start_price = current_prices[i+max_window-1]
                end_price = current_prices[i+max_window+future_days-1]
                future_return = (end_price - start_price) / start_price if start_price > 0 else 0.0
                all_future_returns.append(future_return)
        
        if len(all_future_returns) == 0:
            print("Warning: No valid sequences found!")
            return []
        
        all_future_returns_array = np.array(all_future_returns)
        quantiles = np.percentile(all_future_returns_array, [20, 40, 60, 80])
        
        # 生成横截面数据：每个时间步一个batch
        cross_sectional_batches = []
        num_industries = len(self.industry_list)
        
        # 计算可用的时间步数
        max_time_steps = min_length - max_window - future_days + 1
        
        for time_idx in range(max_time_steps):
            batch_sequences = []
            batch_targets = []
            batch_masks = []
            batch_industry_indices = []
            
            # 为每个行业提取该时间步的数据
            for industry_idx in range(num_industries):
                if industry_idx not in industry_data_dict:
                    # 如果该行业没有数据，跳过
                    continue
                
                data = industry_data_dict[industry_idx]
                if time_idx + max_window + future_days > len(data):
                    # 如果该行业在该时间步没有足够的数据，跳过
                    continue
                
                # 提取序列
                seq = data[time_idx:time_idx+max_window]
                
                # 应用特征归一化
                seq_normalized = self.normalize_features(seq)
                
                # 计算未来收益率
                current_prices = data[:, 1]  # 收盘价
                start_price = current_prices[time_idx+max_window-1]
                end_price = current_prices[time_idx+max_window+future_days-1]
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
                
                batch_sequences.append(seq_normalized)
                batch_targets.append(target)
                batch_industry_indices.append(industry_idx)
                
                # 创建掩码（全部有效）
                mask = np.ones(max_window, dtype=np.float32)
                batch_masks.append(mask)
            
            # 如果该时间步有数据，添加到列表
            if len(batch_sequences) > 0:
                cross_sectional_batches.append({
                    'sequences': np.array(batch_sequences),  # [num_valid_industries, max_window, features]
                    'targets': np.array(batch_targets),  # [num_valid_industries]
                    'masks': np.array(batch_masks),  # [num_valid_industries, max_window]
                    'industry_indices': np.array(batch_industry_indices),  # [num_valid_industries]
                    'time_index': time_idx  # 时间步索引
                })
        
        print(f"生成了 {len(cross_sectional_batches)} 个横截面时间步")
        print(f"每个时间步平均包含 {np.mean([len(b['sequences']) for b in cross_sectional_batches]):.1f} 个行业")
        
        return cross_sectional_batches

    def save_scalers(self, save_path: str = "checkpoints/scalers.pkl"):
        """
        保存特征标准化器到文件

        Args:
            save_path: 保存路径
        """
        if not self.scalers_fitted:
            print("Warning: Scalers have not been fitted yet.")
            return

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        scaler_params = {
            'price': self.scaler_price.get_params(),
            'volume': self.scaler_volume.get_params(),
            'amount': self.scaler_amount.get_params(),
            'fitted': self.scalers_fitted
        }

        with open(save_path, 'wb') as f:
            pickle.dump(scaler_params, f)

        print(f"✓ 标准化器已保存到: {save_path}")

    def load_scalers(self, load_path: str = "checkpoints/scalers.pkl"):
        """
        从文件加载特征标准化器

        Args:
            load_path: 加载路径
        """
        if not os.path.exists(load_path):
            print(f"Warning: Scaler file not found at {load_path}")
            return False

        with open(load_path, 'rb') as f:
            scaler_params = pickle.load(f)

        self.scaler_price.set_params(scaler_params['price'])
        self.scaler_volume.set_params(scaler_params['volume'])
        self.scaler_amount.set_params(scaler_params['amount'])
        self.scalers_fitted = scaler_params['fitted']

        print(f"✓ 标准化器已从 {load_path} 加载")
        print(f"  价格特征 - Mean: {self.scaler_price.mean.flatten()[:4]}")
        print(f"  价格特征 - Std:  {self.scaler_price.std.flatten()[:4]}")
        print(f"  成交量   - Mean: {self.scaler_volume.mean.item():.2e}, Std: {self.scaler_volume.std.item():.2e}")
        print(f"  成交额   - Mean: {self.scaler_amount.mean.item():.2e}, Std: {self.scaler_amount.std.item():.2e}")

        return True


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

