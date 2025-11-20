"""
横截面局部采样数据集
用于支持横截面+局部训练的新架构
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import math

from .degree_sampler import DegreeBasedSampler, SequentialSampler


class CrossSectionalLocalDataset(Dataset):
    """
    横截面局部采样数据集

    特点：
    1. 每个时间步包含所有86个行业的同步数据（横截面）
    2. 每次采样只选择部分中心节点+邻居（局部采样）
    3. 其他节点被掩码，只使用行业嵌入

    数据组织：
    - 外层循环：时间步（~900个）
    - 内层循环：每个时间步的多次局部采样
    """

    def __init__(self,
                 cross_sectional_data: List[Dict[str, np.ndarray]],
                 adj_matrix: np.ndarray,
                 num_centers: int = 12,
                 sampler_type: str = 'degree',
                 sampler_temperature: float = 1.0,
                 samples_per_timestep: Optional[int] = None):
        """
        Args:
            cross_sectional_data: prepare_cross_sectional_data()的输出
                List of dicts, each containing:
                - 'sequences': [num_industries, max_window, features]
                - 'targets': [num_industries]
                - 'masks': [num_industries, max_window]
                - 'industry_indices': [num_industries]
                - 'time_index': int
            adj_matrix: [86, 86] 行业关系邻接矩阵
            num_centers: 每次采样的中心节点数
            sampler_type: 采样器类型 ('degree', 'sequential', 'random')
            sampler_temperature: 度数采样的温度参数
            samples_per_timestep: 每个时间步采样多少批（None则自动计算）
        """
        self.cross_sectional_data = cross_sectional_data
        self.adj_matrix = torch.FloatTensor(adj_matrix)
        self.num_centers = num_centers
        self.sampler_type = sampler_type
        self.num_industries = 86

        # 每个时间步的采样次数（确保覆盖所有86个行业）
        if samples_per_timestep is None:
            self.samples_per_timestep = math.ceil(self.num_industries / num_centers)
        else:
            self.samples_per_timestep = samples_per_timestep

        # 初始化采样器
        if sampler_type == 'degree':
            self.sampler = DegreeBasedSampler(self.adj_matrix, temperature=sampler_temperature)
        elif sampler_type == 'sequential':
            self.sampler = SequentialSampler(self.num_industries, self.adj_matrix, shuffle=True)
        else:  # random
            self.sampler = DegreeBasedSampler(self.adj_matrix, temperature=100.0)  # 高温 = 接近均匀

        # 时间步计数
        self.num_time_steps = len(cross_sectional_data)

        print(f"CrossSectionalLocalDataset initialized:")
        print(f"  Time steps: {self.num_time_steps}")
        print(f"  Samples per time step: {self.samples_per_timestep}")
        print(f"  Total samples per epoch: {len(self)}")
        print(f"  Sampler type: {sampler_type}")
        print(f"  Centers per sample: {num_centers}")

    def __len__(self):
        """总样本数 = 时间步数 × 每步采样次数"""
        return self.num_time_steps * self.samples_per_timestep

    def __getitem__(self, idx):
        """
        获取一个训练样本

        Returns:
            dict containing:
            - 'sequence': [num_active, max_window, features] 中心+邻居的时间序列
            - 'target': [num_active] 对应的标签
            - 'mask': [num_active, max_window] 时间序列掩码
            - 'industry_idx': [num_active] 中心+邻居的行业索引
            - 'node_mask': [86] bool tensor, True=有输入, False=掩码节点
            - 'center_mask': [num_active] bool tensor, True=中心节点
            - 'time_index': int 时间步索引
        """
        # 计算时间步索引和子样本索引
        time_idx = idx // self.samples_per_timestep
        sub_idx = idx % self.samples_per_timestep

        # 如果是新时间步的第一个样本，重置采样器
        if sub_idx == 0 and hasattr(self.sampler, 'reset'):
            self.sampler.reset()

        # 获取该时间步的横截面数据
        time_data = self.cross_sectional_data[time_idx]
        all_sequences = time_data['sequences']  # [num_valid_industries, max_window, features]
        all_targets = time_data['targets']
        all_masks = time_data['masks']
        all_industry_indices = time_data['industry_indices']

        # 采样中心节点
        if self.sampler_type == 'sequential' and hasattr(self.sampler, 'sample'):
            center_nodes = self.sampler.sample(self.num_centers)
        else:
            # 度数采样或随机采样
            center_nodes = self.sampler.sample(self.num_centers)

        # 获取1-hop邻居
        neighbor_nodes = self.sampler.get_neighbors(center_nodes)

        # 合并中心+邻居
        active_nodes = center_nodes + neighbor_nodes
        active_nodes_set = set(active_nodes)

        # 创建node_mask (True=有输入, False=掩码)
        node_mask = torch.zeros(self.num_industries, dtype=torch.bool)
        node_mask[active_nodes] = True

        # 提取active节点在该时间步的数据
        # 注意：time_data中的industry_indices可能不包含所有86个行业
        active_sequences = []
        active_targets = []
        active_masks = []
        active_industry_idx = []
        center_mask_list = []

        for node_idx in active_nodes:
            # 查找该行业在time_data中的位置
            try:
                pos = np.where(all_industry_indices == node_idx)[0]
                if len(pos) > 0:
                    pos = pos[0]
                    active_sequences.append(all_sequences[pos])
                    active_targets.append(all_targets[pos])
                    active_masks.append(all_masks[pos])
                    active_industry_idx.append(node_idx)
                    center_mask_list.append(node_idx in center_nodes)
                else:
                    # 该行业在这个时间步没有数据，跳过
                    # 这种情况应该很少见
                    continue
            except Exception as e:
                print(f"Warning: Error processing node {node_idx} at time {time_idx}: {e}")
                continue

        if len(active_sequences) == 0:
            # 如果没有有效数据，返回空样本（应该极少发生）
            print(f"Warning: No valid active nodes at time {time_idx}, sub {sub_idx}")
            # 返回一个dummy样本
            return {
                'sequence': torch.zeros((1, all_sequences.shape[1], all_sequences.shape[2])),
                'target': torch.zeros(1, dtype=torch.long),
                'mask': torch.ones((1, all_sequences.shape[1])),
                'industry_idx': torch.zeros(1, dtype=torch.long),
                'node_mask': node_mask,
                'center_mask': torch.tensor([True]),
                'time_index': torch.tensor(time_idx)
            }

        return {
            'sequence': torch.FloatTensor(np.array(active_sequences)),  # [num_active, max_window, features]
            'target': torch.LongTensor(np.array(active_targets)),  # [num_active]
            'mask': torch.FloatTensor(np.array(active_masks)),  # [num_active, max_window]
            'industry_idx': torch.LongTensor(np.array(active_industry_idx)),  # [num_active]
            'node_mask': node_mask,  # [86]
            'center_mask': torch.BoolTensor(center_mask_list),  # [num_active]
            'time_index': torch.tensor(time_idx)  # scalar
        }

    def get_time_step_info(self, time_idx: int) -> Dict:
        """
        获取某个时间步的统计信息

        Args:
            time_idx: 时间步索引

        Returns:
            info dict
        """
        if time_idx >= self.num_time_steps:
            return {}

        time_data = self.cross_sectional_data[time_idx]
        return {
            'time_index': time_idx,
            'num_valid_industries': len(time_data['industry_indices']),
            'valid_industries': time_data['industry_indices'].tolist(),
            'target_distribution': np.bincount(time_data['targets'], minlength=5).tolist()
        }

    def get_sampling_statistics(self) -> Dict:
        """
        获取采样器的统计信息

        Returns:
            sampling stats
        """
        if hasattr(self.sampler, 'get_sampling_statistics'):
            return self.sampler.get_sampling_statistics()
        return {}

    def reset_sampling_statistics(self):
        """重置采样统计"""
        if hasattr(self.sampler, 'reset_statistics'):
            self.sampler.reset_statistics()


def cross_sectional_collate_fn(batch):
    """
    自定义collate函数，用于处理CrossSectionalLocalDataset的可变大小序列
    
    由于每个样本的num_active可能不同，我们需要：
    1. 将序列、目标、掩码等按样本组织（不堆叠）
    2. 固定大小的张量（node_mask, time_index）可以正常堆叠
    
    Args:
        batch: List of dicts from __getitem__
    
    Returns:
        dict with batched tensors
    """
    # 固定大小的字段可以直接堆叠
    node_masks = torch.stack([item['node_mask'] for item in batch])  # [batch_size, 86]
    time_indices = torch.stack([item['time_index'] for item in batch])  # [batch_size]
    
    # 可变大小的字段需要保持为列表
    sequences = [item['sequence'] for item in batch]  # List of [num_active_i, max_window, features]
    targets = [item['target'] for item in batch]  # List of [num_active_i]
    masks = [item['mask'] for item in batch]  # List of [num_active_i, max_window]
    industry_indices = [item['industry_idx'] for item in batch]  # List of [num_active_i]
    center_masks = [item['center_mask'] for item in batch]  # List of [num_active_i]
    
    return {
        'sequence': sequences,  # List of tensors
        'target': targets,  # List of tensors
        'mask': masks,  # List of tensors
        'industry_idx': industry_indices,  # List of tensors
        'node_mask': node_masks,  # [batch_size, 86]
        'center_mask': center_masks,  # List of tensors
        'time_index': time_indices  # [batch_size]
    }
