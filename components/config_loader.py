"""
配置加载器
支持YAML配置文件和命令行参数覆盖
"""
import yaml
import os
import torch
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """配置类,支持字典式和属性式访问"""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Args:
            config_dict: 配置字典
        """
        self._config = config_dict

        # 将嵌套字典转换为Config对象
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """支持字典式赋值"""
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置值"""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in self._config and isinstance(self._config[key], dict):
                # 递归更新嵌套字典
                if hasattr(self, key) and isinstance(getattr(self, key), Config):
                    getattr(self, key).update(value)
                else:
                    self._config[key].update(value)
            else:
                self[key] = value

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Config对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    保存配置到YAML文件

    Args:
        config: Config对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_device(device_str: str = "auto") -> torch.device:
    """
    获取计算设备

    Args:
        device_str: 设备字符串 ("auto", "cuda", "cpu", "mps")

    Returns:
        torch.device对象
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def setup_directories(config: Config):
    """
    创建必要的目录

    Args:
        config: 配置对象
    """
    dirs_to_create = [
        config.training.save_dir,
        config.inference.output_dir,
        config.visualization.save_dir,
        config.logging.log_dir
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def setup_seed(seed: int, deterministic: bool = True):
    """
    设置随机种子以保证可复现性

    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        参数命名空间
    """
    parser = argparse.ArgumentParser(description="MMF-GAT Industry Prediction")

    # 配置文件
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Path to config file")

    # 训练参数覆盖
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to use")

    # 模型参数覆盖
    parser.add_argument("--d_model", type=int, help="Time encoder dimension")
    parser.add_argument("--nhead", type=int, help="Number of attention heads")
    parser.add_argument("--use_dwt", action="store_true", help="Use DWT enhancement")
    parser.add_argument("--no_dwt", action="store_true", help="Disable DWT enhancement")

    # 数据参数
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--use_kfold", action="store_true", help="Use K-fold validation")
    parser.add_argument("--n_splits", type=int, help="Number of K-fold splits")

    # 实验参数
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--seed", type=int, help="Random seed")

    # 模式
    parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"],
                       default="train", help="Running mode")

    # 推理参数
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for inference")

    return parser.parse_args()


def merge_args_to_config(config: Config, args: argparse.Namespace) -> Config:
    """
    将命令行参数合并到配置中

    Args:
        config: 配置对象
        args: 命令行参数

    Returns:
        更新后的配置对象
    """
    # 训练参数
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.device is not None:
        config.training.device = args.device

    # 模型参数
    if args.d_model is not None:
        config.model.time_encoder.d_model = args.d_model
        config.model.compression.in_features = args.d_model
    if args.nhead is not None:
        config.model.time_encoder.nhead = args.nhead
        config.model.gat.num_heads = args.nhead
    if args.use_dwt:
        config.model.use_dwt = True
    if args.no_dwt:
        config.model.use_dwt = False

    # 数据参数
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.use_kfold:
        config.data.use_kfold = True
    if args.n_splits is not None:
        config.data.n_splits = args.n_splits

    # 实验参数
    if args.exp_name is not None:
        config.experiment.name = args.exp_name
    if args.seed is not None:
        config.experiment.seed = args.seed

    # 推理参数
    if args.checkpoint is not None:
        config.inference.checkpoint_path = args.checkpoint

    return config


def load_config_with_cli(default_config_path: str = "config/default_config.yaml") -> tuple:
    """
    加载配置并合并命令行参数

    Args:
        default_config_path: 默认配置文件路径

    Returns:
        (config, args) 元组
    """
    args = parse_args()

    # 使用指定的配置文件或默认配置
    config_path = args.config if args.config else default_config_path
    config = load_config(config_path)

    # 合并命令行参数
    config = merge_args_to_config(config, args)

    return config, args


if __name__ == "__main__":
    # 测试配置加载
    print("=== 测试配置加载 ===")

    # 加载默认配置
    config = load_config("../config/default_config.yaml")

    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Model d_model: {config.model.time_encoder.d_model}")
    print(f"Use DWT: {config.model.use_dwt}")

    # 测试更新
    config.training.batch_size = 64
    print(f"Updated batch size: {config.training.batch_size}")

    # 测试设备选择
    device = get_device(config.training.device)
    print(f"Device: {device}")

    # 测试字典转换
    config_dict = config.to_dict()
    print(f"Config dict keys: {list(config_dict.keys())}")
