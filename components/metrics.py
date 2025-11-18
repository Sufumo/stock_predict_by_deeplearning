"""
金融评估指标
包括IC、RankIC、Sharpe Ratio、分层收益分析等量化指标
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy import stats


def information_coefficient(predictions: np.ndarray, returns: np.ndarray) -> float:
    """
    计算信息系数(IC - Information Coefficient)
    IC是预测值与真实收益率的Pearson相关系数

    Args:
        predictions: 预测值数组,形状为 [n_samples]
        returns: 真实收益率数组,形状为 [n_samples]

    Returns:
        IC值,范围[-1, 1],越接近1表示预测能力越强
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions and returns must have the same length")

    # 移除nan值
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[mask]
    ret_clean = returns[mask]

    if len(pred_clean) < 2:
        return 0.0

    # Pearson相关系数
    ic, _ = stats.pearsonr(pred_clean, ret_clean)

    return ic


def rank_information_coefficient(predictions: np.ndarray, returns: np.ndarray) -> float:
    """
    计算排序信息系数(RankIC)
    RankIC是预测值排序与真实收益率排序的Spearman秩相关系数
    相比IC更稳健,对异常值不敏感

    Args:
        predictions: 预测值数组
        returns: 真实收益率数组

    Returns:
        RankIC值,范围[-1, 1]
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions and returns must have the same length")

    # 移除nan值
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[mask]
    ret_clean = returns[mask]

    if len(pred_clean) < 2:
        return 0.0

    # Spearman秩相关系数
    rank_ic, _ = stats.spearmanr(pred_clean, ret_clean)

    return rank_ic


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    计算夏普比率(Sharpe Ratio)
    衡量每单位风险的超额收益

    Args:
        returns: 收益率序列(时间序列)
        risk_free_rate: 无风险利率(年化)
        periods_per_year: 每年的周期数(日度=252,周度=52,月度=12)

    Returns:
        年化夏普比率
    """
    if len(returns) < 2:
        return 0.0

    # 移除nan
    returns_clean = returns[~np.isnan(returns)]

    if len(returns_clean) < 2:
        return 0.0

    # 平均收益
    mean_return = np.mean(returns_clean)
    # 收益标准差
    std_return = np.std(returns_clean, ddof=1)

    if std_return == 0:
        return 0.0

    # 年化
    period_risk_free = risk_free_rate / periods_per_year
    sharpe = (mean_return - period_risk_free) / std_return * np.sqrt(periods_per_year)

    return sharpe


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    计算最大回撤(Maximum Drawdown)

    Args:
        cumulative_returns: 累计收益率序列

    Returns:
        最大回撤(正数表示回撤幅度)
    """
    if len(cumulative_returns) < 2:
        return 0.0

    # 累计财富曲线
    wealth = 1 + cumulative_returns

    # 历史最高点
    running_max = np.maximum.accumulate(wealth)

    # 回撤
    drawdown = (wealth - running_max) / running_max

    # 最大回撤
    max_dd = -np.min(drawdown)

    return max_dd


def quantile_analysis(predictions: np.ndarray, returns: np.ndarray,
                     n_quantiles: int = 5) -> Dict[str, float]:
    """
    分位数分析
    将预测值分成n个分位数组,计算每组的平均收益率

    Args:
        predictions: 预测值
        returns: 真实收益率
        n_quantiles: 分位数个数

    Returns:
        包含各分位数收益统计的字典
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions and returns must have the same length")

    # 移除nan
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[mask]
    ret_clean = returns[mask]

    if len(pred_clean) < n_quantiles:
        return {}

    # 按预测值排序,分成n个分位数
    quantile_indices = np.argsort(pred_clean)
    quantile_size = len(quantile_indices) // n_quantiles

    results = {}

    for i in range(n_quantiles):
        start_idx = i * quantile_size
        if i == n_quantiles - 1:
            end_idx = len(quantile_indices)  # 最后一组包含所有剩余样本
        else:
            end_idx = (i + 1) * quantile_size

        quantile_idx = quantile_indices[start_idx:end_idx]
        quantile_returns = ret_clean[quantile_idx]

        results[f"Q{i+1}_mean_return"] = np.mean(quantile_returns)
        results[f"Q{i+1}_std_return"] = np.std(quantile_returns)
        results[f"Q{i+1}_count"] = len(quantile_returns)

    # Long-Short收益(最高分位数 - 最低分位数)
    top_quantile_idx = quantile_indices[-quantile_size:]
    bottom_quantile_idx = quantile_indices[:quantile_size]

    long_short_return = (np.mean(ret_clean[top_quantile_idx]) -
                        np.mean(ret_clean[bottom_quantile_idx]))
    results["long_short_return"] = long_short_return

    return results


def win_rate(predictions: np.ndarray, returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    计算胜率
    预测正收益且实际正收益,或预测负收益且实际负收益的比例

    Args:
        predictions: 预测值
        returns: 真实收益率
        threshold: 判断正负的阈值

    Returns:
        胜率,范围[0, 1]
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions and returns must have the same length")

    # 移除nan
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[mask]
    ret_clean = returns[mask]

    if len(pred_clean) == 0:
        return 0.0

    # 预测和实际的方向一致
    pred_direction = pred_clean > threshold
    ret_direction = ret_clean > threshold
    correct = pred_direction == ret_direction

    win_rate_value = np.mean(correct)

    return win_rate_value


def portfolio_performance(predictions: np.ndarray, returns: np.ndarray,
                         top_k: int = 10, bottom_k: int = 10,
                         long_only: bool = False) -> Dict[str, float]:
    """
    投资组合表现分析
    基于预测构建Top K多头 + Bottom K空头的投资组合

    Args:
        predictions: 预测值(可以是概率或分数)
        returns: 真实收益率
        top_k: 选择前k个做多
        bottom_k: 选择后k个做空(如果long_only=True则忽略)
        long_only: 是否只做多

    Returns:
        投资组合收益统计
    """
    if len(predictions) != len(returns):
        raise ValueError("predictions and returns must have the same length")

    # 移除nan
    mask = ~(np.isnan(predictions) | np.isnan(returns))
    pred_clean = predictions[mask]
    ret_clean = returns[mask]

    if len(pred_clean) < max(top_k, bottom_k):
        return {}

    # 按预测值排序
    sorted_indices = np.argsort(pred_clean)

    # Top K (做多)
    top_indices = sorted_indices[-top_k:]
    top_returns = ret_clean[top_indices]
    long_return = np.mean(top_returns)

    results = {
        "long_return": long_return,
        "long_count": len(top_returns)
    }

    if not long_only:
        # Bottom K (做空)
        bottom_indices = sorted_indices[:bottom_k]
        bottom_returns = ret_clean[bottom_indices]
        short_return = -np.mean(bottom_returns)  # 做空收益

        results["short_return"] = short_return
        results["short_count"] = len(bottom_returns)

        # 多空组合收益
        portfolio_return = (long_return + short_return) / 2
        results["portfolio_return"] = portfolio_return

    return results


class FinancialMetricsCalculator:
    """
    金融指标计算器
    用于在训练和验证过程中计算各种金融指标
    """

    def __init__(self, n_quantiles: int = 5, risk_free_rate: float = 0.03):
        """
        Args:
            n_quantiles: 分位数个数
            risk_free_rate: 无风险利率(年化)
        """
        self.n_quantiles = n_quantiles
        self.risk_free_rate = risk_free_rate

    def compute_all_metrics(self, predictions: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            predictions: 预测值
            returns: 真实收益率

        Returns:
            包含所有指标的字典
        """
        metrics = {}

        # IC和RankIC
        metrics["IC"] = information_coefficient(predictions, returns)
        metrics["RankIC"] = rank_information_coefficient(predictions, returns)

        # 胜率
        metrics["win_rate"] = win_rate(predictions, returns)

        # 分位数分析
        quantile_results = quantile_analysis(predictions, returns, self.n_quantiles)
        metrics.update(quantile_results)

        # 投资组合表现
        portfolio_results = portfolio_performance(predictions, returns, top_k=int(len(predictions)*0.2))
        metrics.update(portfolio_results)

        return metrics

    def compute_from_torch(self, predictions: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """
        从PyTorch张量计算指标

        Args:
            predictions: 预测值张量
            returns: 真实收益率张量

        Returns:
            指标字典
        """
        # 转换为numpy
        pred_np = predictions.detach().cpu().numpy()
        ret_np = returns.detach().cpu().numpy()

        return self.compute_all_metrics(pred_np, ret_np)


if __name__ == "__main__":
    # 测试
    np.random.seed(42)

    # 模拟数据
    n_samples = 100
    true_returns = np.random.randn(n_samples) * 0.02  # 2%波动率
    predictions = true_returns + np.random.randn(n_samples) * 0.01  # 加噪声

    print("=== 金融指标测试 ===")
    print(f"IC: {information_coefficient(predictions, true_returns):.4f}")
    print(f"RankIC: {rank_information_coefficient(predictions, true_returns):.4f}")
    print(f"Win Rate: {win_rate(predictions, true_returns):.4f}")

    print("\n=== 分位数分析 ===")
    quantile_res = quantile_analysis(predictions, true_returns, n_quantiles=5)
    for key, value in quantile_res.items():
        print(f"{key}: {value:.6f}")

    print("\n=== 投资组合表现 ===")
    portfolio_res = portfolio_performance(predictions, true_returns, top_k=20, bottom_k=20)
    for key, value in portfolio_res.items():
        print(f"{key}: {value:.6f}")

    # 测试时间序列指标
    cumulative_returns = np.cumsum(true_returns)
    print(f"\nSharpe Ratio: {sharpe_ratio(true_returns):.4f}")
    print(f"Max Drawdown: {max_drawdown(cumulative_returns):.4f}")
