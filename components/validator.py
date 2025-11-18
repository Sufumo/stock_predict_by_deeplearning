"""
时间序列K折验证器
实现严格按时间顺序的交叉验证,避免未来信息泄露
"""
import numpy as np
from typing import List, Tuple, Iterator


class TimeSeriesKFold:
    """
    时间序列K折验证

    与传统K折不同,该验证器严格保持时间顺序:
    - 训练集总是在验证集之前
    - 每个fold的验证集在时间上都晚于训练集
    - 不进行随机打乱

    示例(n_splits=3):
    Fold 1: Train [0:40%], Val [40%:60%]
    Fold 2: Train [0:60%], Val [60%:80%]
    Fold 3: Train [0:80%], Val [80%:100%]
    """

    def __init__(self, n_splits: int = 5, min_train_size: float = 0.3):
        """
        Args:
            n_splits: 折数
            min_train_size: 最小训练集比例(0-1之间)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 < min_train_size < 1:
            raise ValueError("min_train_size must be between 0 and 1")

        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练集和验证集的索引

        Args:
            X: 数据数组(只需要知道长度)

        Yields:
            (train_indices, val_indices) 元组
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # 计算每个fold的验证集大小
        # 总共有(1 - min_train_size)的数据用于验证
        # 分成n_splits个fold
        val_size = int(n_samples * (1 - self.min_train_size) / self.n_splits)

        if val_size < 1:
            raise ValueError(f"Validation size too small. Consider reducing n_splits or min_train_size")

        # 第一个fold的训练集结束位置
        first_train_end = int(n_samples * self.min_train_size)

        for fold in range(self.n_splits):
            # 计算当前fold的验证集范围
            val_start = first_train_end + fold * val_size
            val_end = val_start + val_size

            # 最后一个fold的验证集延伸到数据末尾
            if fold == self.n_splits - 1:
                val_end = n_samples

            # 训练集从开始到验证集开始
            train_indices = indices[:val_start]
            val_indices = indices[val_start:val_end]

            if len(val_indices) == 0:
                break

            yield train_indices, val_indices

    def get_n_splits(self) -> int:
        """返回折数"""
        return self.n_splits


class WalkForwardValidator:
    """
    滚动窗口验证器

    使用固定大小的训练窗口和验证窗口向前滚动
    更严格地模拟真实交易环境

    示例(train_window=100, val_window=20, step=20):
    Fold 1: Train [0:100],   Val [100:120]
    Fold 2: Train [20:120],  Val [120:140]
    Fold 3: Train [40:140],  Val [140:160]
    """

    def __init__(self, train_window: int, val_window: int, step: int = None):
        """
        Args:
            train_window: 训练窗口大小(样本数)
            val_window: 验证窗口大小(样本数)
            step: 每次向前移动的步长,默认为val_window
        """
        if train_window < 1 or val_window < 1:
            raise ValueError("Windows must be at least 1")

        self.train_window = train_window
        self.val_window = val_window
        self.step = step if step is not None else val_window

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练集和验证集的索引

        Args:
            X: 数据数组(只需要知道长度)

        Yields:
            (train_indices, val_indices) 元组
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # 从第一个完整的训练窗口开始
        train_start = 0

        while True:
            train_end = train_start + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window

            # 如果验证集超出数据范围,停止
            if val_end > n_samples:
                # 最后一折可以使用剩余的所有数据作为验证集
                if val_start < n_samples:
                    train_indices = indices[train_start:train_end]
                    val_indices = indices[val_start:n_samples]
                    yield train_indices, val_indices
                break

            train_indices = indices[train_start:train_end]
            val_indices = indices[val_start:val_end]

            yield train_indices, val_indices

            # 向前移动
            train_start += self.step

    def get_n_splits(self, X: np.ndarray) -> int:
        """返回总折数"""
        n_samples = len(X)
        n_splits = 0
        train_start = 0

        while True:
            val_start = train_start + self.train_window
            val_end = val_start + self.val_window

            if val_start >= n_samples:
                break

            n_splits += 1
            train_start += self.step

            if val_end >= n_samples:
                break

        return n_splits


def split_by_date(dates: np.ndarray, train_end_date: str, val_end_date: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据日期划分训练集和验证集

    Args:
        dates: 日期数组(格式如'2024-01-01')
        train_end_date: 训练集截止日期(不包含)
        val_end_date: 验证集截止日期(不包含),如果为None则使用所有后续数据

    Returns:
        (train_indices, val_indices)
    """
    dates = np.array(dates)

    # 训练集: dates < train_end_date
    train_mask = dates < train_end_date
    train_indices = np.where(train_mask)[0]

    # 验证集: train_end_date <= dates < val_end_date
    if val_end_date is not None:
        val_mask = (dates >= train_end_date) & (dates < val_end_date)
    else:
        val_mask = dates >= train_end_date

    val_indices = np.where(val_mask)[0]

    return train_indices, val_indices


if __name__ == "__main__":
    # 测试TimeSeriesKFold
    print("=== TimeSeriesKFold测试 ===")
    X = np.arange(100)
    tscv = TimeSeriesKFold(n_splits=3, min_train_size=0.4)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"Fold {fold}:")
        print(f"  训练集: {len(train_idx)} 样本, 范围 [{train_idx[0]}, {train_idx[-1]}]")
        print(f"  验证集: {len(val_idx)} 样本, 范围 [{val_idx[0]}, {val_idx[-1]}]")

    # 测试WalkForwardValidator
    print("\n=== WalkForwardValidator测试 ===")
    wfv = WalkForwardValidator(train_window=40, val_window=10, step=10)

    for fold, (train_idx, val_idx) in enumerate(wfv.split(X), 1):
        print(f"Fold {fold}:")
        print(f"  训练集: {len(train_idx)} 样本, 范围 [{train_idx[0]}, {train_idx[-1]}]")
        print(f"  验证集: {len(val_idx)} 样本, 范围 [{val_idx[0]}, {val_idx[-1]}]")
