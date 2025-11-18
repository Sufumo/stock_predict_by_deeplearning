"""
数据预处理组件
负责清洗数据源、处理NaN值、日期范围过滤和统计信息输出
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import warnings


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, 
                 start_date: str = '2021-12-01',
                 end_date: str = '2025-11-17',
                 nan_strategy: str = 'forward_fill',
                 min_valid_samples: int = 100,
                 verbose: bool = True):
        """
        初始化数据预处理器
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
            nan_strategy: NaN值处理策略（默认：forward_fill）
            min_valid_samples: 每个行业最少有效样本数
            verbose: 是否输出详细信息
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.nan_strategy = nan_strategy
        self.min_valid_samples = min_valid_samples
        self.verbose = verbose
        
        # 统计信息
        self.stats = {
            'total_industries': 0,
            'valid_industries': 0,
            'removed_industries': [],
            'industry_stats': {},
            'nan_counts': {},
            'date_filtered_counts': {},
            'total_samples_before': 0,
            'total_samples_after': 0
        }
        # 记录每个行业对应的日期序列
        self.industry_dates: Dict[str, List[str]] = {}
    
    def clean_kline_data(self, 
                        raw_data: Dict[str, List[List[str]]],
                        relation_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        清洗K线数据
        
        Args:
            raw_data: 原始K线数据字典 {行业名称: [[日期, 开盘, ...], ...]}
            relation_df: 行业关系DataFrame
            
        Returns:
            cleaned_data: 清洗后的数据字典 {行业名称: numpy数组 [时间步, 特征数]}
            valid_relation_df: 有效行业的关系DataFrame
        """
        if self.verbose:
            print("=" * 60)
            print("开始数据清洗")
            print(f"日期范围: {self.start_date.strftime('%Y-%m-%d')} 到 {self.end_date.strftime('%Y-%m-%d')}")
            print("=" * 60)
        
        cleaned_data = {}
        valid_industries = []
        
        # 获取行业列表
        industry_list = relation_df['industry'].tolist()
        self.stats['total_industries'] = len(industry_list)
        
        for industry_name in industry_list:
            if industry_name not in raw_data:
                if self.verbose:
                    print(f"⚠️  行业 '{industry_name}' 不在原始数据中，跳过")
                self.stats['removed_industries'].append({
                    'industry': industry_name,
                    'reason': '数据不存在'
                })
                continue
            
            klines = raw_data[industry_name]
            if not klines:
                if self.verbose:
                    print(f"⚠️  行业 '{industry_name}' 数据为空，跳过")
                self.stats['removed_industries'].append({
                    'industry': industry_name,
                    'reason': '数据为空'
                })
                continue
            
            # 解析和清洗单个行业的数据
            cleaned_result = self._clean_single_industry(industry_name, klines)
            
            if cleaned_result is None:
                self.stats['removed_industries'].append({
                    'industry': industry_name,
                    'reason': '清洗后数据无效'
                })
                continue
            
            cleaned_array, sorted_dates = cleaned_result
            
            # 检查样本数
            if len(cleaned_array) < self.min_valid_samples:
                if self.verbose:
                    print(f"⚠️  行业 '{industry_name}' 样本数不足 ({len(cleaned_array)} < {self.min_valid_samples})，跳过")
                self.stats['removed_industries'].append({
                    'industry': industry_name,
                    'reason': f'样本数不足 ({len(cleaned_array)} < {self.min_valid_samples})'
                })
                continue
            
            cleaned_data[industry_name] = cleaned_array
            self.industry_dates[industry_name] = sorted_dates
            valid_industries.append(industry_name)
            
            # 统计信息
            nan_stats = self.stats['nan_counts'].get(industry_name, {})
            self.stats['industry_stats'][industry_name] = {
                'samples': len(cleaned_array),
                'features': cleaned_array.shape[1] if len(cleaned_array) > 0 else 0,
                'nan_count': nan_stats.get('after', 0),
                'date_filtered': self.stats['date_filtered_counts'].get(industry_name, 0)
            }
        
        # 过滤有效行业的relation_df
        valid_relation_df = relation_df[relation_df['industry'].isin(valid_industries)].copy()
        valid_relation_df = valid_relation_df.reset_index(drop=True)
        
        self.stats['valid_industries'] = len(valid_industries)
        self.stats['total_samples_after'] = sum(
            self.stats['industry_stats'][ind]['samples'] 
            for ind in valid_industries
        )
        
        if self.verbose:
            self._print_cleaning_summary()
        
        return cleaned_data, valid_relation_df
    
    def _clean_single_industry(self, 
                               industry_name: str,
                               klines: List[List[str]]) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        清洗单个行业的数据
        
        Args:
            industry_name: 行业名称
            klines: K线数据列表
            
        Returns:
            清洗后的numpy数组 [时间步, 特征数]，如果无效则返回None
        """
        data_list = []
        dates = []
        nan_count = 0
        invalid_count = 0
        date_filtered_count = 0
        
        # 第一步：解析数据，应用日期过滤
        for kline in klines:
            try:
                if len(kline) < 7:
                    invalid_count += 1
                    continue
                
                date_str = kline[0]
                
                # 解析日期
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    # 尝试其他日期格式
                    try:
                        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                    except ValueError:
                        invalid_count += 1
                        continue
                
                # 日期范围过滤
                if date_obj < self.start_date or date_obj > self.end_date:
                    date_filtered_count += 1
                    continue
                
                dates.append(date_str)
                
                # 提取数值特征
                features = []
                has_nan = False
                
                for i in range(1, 7):  # 开盘、收盘、最高、最低、成交量、成交额
                    try:
                        value = float(kline[i])
                        # 检查是否为NaN或Inf
                        if np.isnan(value) or np.isinf(value):
                            has_nan = True
                            nan_count += 1
                            features.append(np.nan)
                        elif value < 0 and i < 5:  # 价格不能为负
                            has_nan = True
                            features.append(np.nan)
                        else:
                            features.append(value)
                    except (ValueError, IndexError):
                        has_nan = True
                        nan_count += 1
                        features.append(np.nan)
                
                # 计算收益率（暂时设为NaN，后续处理）
                features.append(np.nan)
                data_list.append(features)
                
            except Exception as e:
                invalid_count += 1
                continue
        
        if len(data_list) == 0:
            return None
        
        # 转换为numpy数组
        data_array = np.array(data_list, dtype=np.float32)
        
        # 按日期排序（确保时间顺序）
        sorted_dates: List[str] = dates
        if dates:
            # 创建日期索引用于排序
            date_indices = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            sorted_indices = np.argsort(date_indices)
            data_array = data_array[sorted_indices]
            sorted_dates = [dates[idx] for idx in sorted_indices]
        
        # 计算收益率
        close_prices = data_array[:, 1]  # 收盘价
        for i in range(len(data_array)):
            if i > 0 and not np.isnan(close_prices[i-1]) and close_prices[i-1] > 0:
                if not np.isnan(close_prices[i]):
                    data_array[i, 6] = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                else:
                    data_array[i, 6] = np.nan
            else:
                data_array[i, 6] = 0.0
        
        # 记录统计
        self.stats['nan_counts'][industry_name] = {
            'before': nan_count + np.isnan(data_array).sum(),
            'after': 0  # 将在处理后更新
        }
        self.stats['date_filtered_counts'][industry_name] = date_filtered_count
        
        # 第二步：处理NaN值（使用forward_fill）
        data_array = self._handle_nan_values(data_array, industry_name)
        
        # 更新NaN统计
        nan_after = np.isnan(data_array).sum()
        self.stats['nan_counts'][industry_name]['after'] = nan_after
        
        # 如果仍有NaN，使用更激进的方法
        if nan_after > 0:
            if self.verbose:
                print(f"  ⚠️  行业 '{industry_name}' 仍有 {nan_after} 个NaN值，使用零填充")
            data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 第三步：数据验证
        if not self._validate_data(data_array):
            if self.verbose:
                print(f"  ❌ 行业 '{industry_name}' 数据验证失败")
            return None
        
        return data_array, sorted_dates
    
    def _handle_nan_values(self, data: np.ndarray, industry_name: str) -> np.ndarray:
        """
        处理NaN值（使用forward_fill前向填充）
        
        Args:
            data: 数据数组 [时间步, 特征数]
            industry_name: 行业名称
            
        Returns:
            处理后的数据数组
        """
        if self.nan_strategy == 'forward_fill':
            # 前向填充（适合时间序列）
            df = pd.DataFrame(data)
            df = df.ffill()  # 前向填充
            # 如果第一行仍有NaN，使用后向填充
            df = df.bfill()  # 后向填充
            return df.values.astype(np.float32)
        
        elif self.nan_strategy == 'backward_fill':
            # 后向填充
            df = pd.DataFrame(data)
            df = df.bfill()  # 后向填充
            df = df.ffill()  # 前向填充
            return df.values.astype(np.float32)
        
        elif self.nan_strategy == 'interpolate':
            # 线性插值
            df = pd.DataFrame(data)
            df = df.interpolate(method='linear', limit_direction='both')
            df = df.ffill().bfill()  # 填充剩余的NaN
            return df.values.astype(np.float32)
        
        elif self.nan_strategy == 'zero':
            # 填充0
            return np.nan_to_num(data, nan=0.0)
        
        elif self.nan_strategy == 'mean':
            # 填充均值（按列）
            df = pd.DataFrame(data)
            df = df.fillna(df.mean())
            return df.values.astype(np.float32)
        
        elif self.nan_strategy == 'drop':
            # 删除包含NaN的行
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]
        
        else:
            warnings.warn(f"未知的NaN处理策略: {self.nan_strategy}，使用前向填充")
            df = pd.DataFrame(data)
            df = df.ffill().bfill()
            return df.values.astype(np.float32)
    
    def _validate_data(self, data: np.ndarray) -> bool:
        """
        验证数据有效性
        
        Args:
            data: 数据数组
            
        Returns:
            是否有效
        """
        if data is None or len(data) == 0:
            return False
        
        # 检查是否有NaN或Inf
        if np.isnan(data).any() or np.isinf(data).any():
            return False
        
        # 检查价格数据是否合理（前4列：开盘、收盘、最高、最低）
        prices = data[:, :4]
        if (prices < 0).any():
            return False
        
        # 检查最高价 >= 最低价
        if (data[:, 2] < data[:, 3]).any():  # 最高 < 最低
            return False
        
        # 检查成交量、成交额是否合理（应该 >= 0）
        if (data[:, 4] < 0).any() or (data[:, 5] < 0).any():
            return False
        
        return True
    
    def _print_cleaning_summary(self):
        """打印清洗摘要"""
        print(f"\n数据清洗完成:")
        print(f"  总行业数: {self.stats['total_industries']}")
        print(f"  有效行业数: {self.stats['valid_industries']}")
        print(f"  移除行业数: {len(self.stats['removed_industries'])}")
        print(f"  总样本数: {self.stats['total_samples_after']:,}")
    
    def print_industry_stats(self, 
                            sort_by: str = 'samples',
                            top_n: Optional[int] = None):
        """
        打印每个行业的统计信息
        
        Args:
            sort_by: 排序字段 ('samples', 'nan_count', 'date_filtered')
            top_n: 只显示前N个行业，None表示显示全部
        """
        if not self.stats['industry_stats']:
            print("没有行业统计数据")
            return
        
        print("\n" + "=" * 80)
        print("各行业样本统计")
        print("=" * 80)
        print(f"{'行业名称':<20} {'样本数':<12} {'NaN数':<12} {'日期过滤':<12} {'特征数':<10}")
        print("-" * 80)
        
        # 排序
        sorted_stats = sorted(
            self.stats['industry_stats'].items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        
        # 限制显示数量
        if top_n is not None:
            sorted_stats = sorted_stats[:top_n]
        
        for industry_name, stats in sorted_stats:
            print(f"{industry_name:<20} "
                  f"{stats['samples']:<12,} "
                  f"{stats['nan_count']:<12} "
                  f"{stats['date_filtered']:<12} "
                  f"{stats['features']:<10}")
        
        print("=" * 80)
        
        # 统计摘要
        samples_list = [s['samples'] for s in self.stats['industry_stats'].values()]
        print(f"\n样本数统计:")
        print(f"  最小值: {min(samples_list):,}")
        print(f"  最大值: {max(samples_list):,}")
        print(f"  平均值: {np.mean(samples_list):.0f}")
        print(f"  中位数: {np.median(samples_list):.0f}")
        print(f"  标准差: {np.std(samples_list):.0f}")
    
    def save_cleaning_report(self, save_path: str):
        """
        保存清洗报告到文件
        
        Args:
            save_path: 保存路径
        """
        report = {
            'summary': {
                'total_industries': self.stats['total_industries'],
                'valid_industries': self.stats['valid_industries'],
                'removed_industries_count': len(self.stats['removed_industries']),
                'total_samples': self.stats['total_samples_after'],
                'date_range': {
                    'start': self.start_date.strftime('%Y-%m-%d'),
                    'end': self.end_date.strftime('%Y-%m-%d')
                }
            },
            'removed_industries': self.stats['removed_industries'],
            'industry_stats': self.stats['industry_stats'],
            'nan_counts': self.stats['nan_counts'],
            'date_filtered_counts': self.stats['date_filtered_counts']
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n清洗报告已保存到: {save_path}")
    
    def get_industry_sample_counts(self) -> Dict[str, int]:
        """
        获取每个行业的样本数
        
        Returns:
            字典 {行业名称: 样本数}
        """
        return {
            industry: stats['samples']
            for industry, stats in self.stats['industry_stats'].items()
        }


def preprocess_data(data_path: str,
                   relation_path: str,
                   output_data_path: Optional[str] = None,
                   output_relation_path: Optional[str] = None,
                   start_date: str = '2021-12-01',
                   end_date: str = '2025-11-17',
                   nan_strategy: str = 'forward_fill',
                   min_valid_samples: int = 100,
                   save_report: bool = True,
                   report_path: Optional[str] = None,
                   verbose: bool = True) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, DataPreprocessor]:
    """
    预处理数据的便捷函数
    
    Args:
        data_path: 原始K线数据JSON文件路径
        relation_path: 行业关系CSV文件路径
        output_data_path: 清洗后数据保存路径（可选）
        output_relation_path: 清洗后关系文件保存路径（可选）
        start_date: 开始日期（格式：YYYY-MM-DD）
        end_date: 结束日期（格式：YYYY-MM-DD）
        nan_strategy: NaN处理策略（默认：forward_fill）
        min_valid_samples: 最少有效样本数
        save_report: 是否保存报告
        report_path: 报告保存路径
        verbose: 是否输出详细信息
        
    Returns:
        cleaned_data: 清洗后的数据字典
        valid_relation_df: 有效行业的关系DataFrame
        preprocessor: 预处理器对象
    """
    # 加载原始数据
    if verbose:
        print("加载原始数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    relation_df = pd.read_csv(relation_path)
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        start_date=start_date,
        end_date=end_date,
        nan_strategy=nan_strategy,
        min_valid_samples=min_valid_samples,
        verbose=verbose
    )
    
    # 清洗数据
    cleaned_data, valid_relation_df = preprocessor.clean_kline_data(raw_data, relation_df)
    
    # 打印统计信息
    if verbose:
        preprocessor.print_industry_stats()
    
    # 保存清洗后的数据
    if output_data_path:
        if verbose:
            print(f"\n保存清洗后的数据到: {output_data_path}")
        # 将numpy数组和日期转换回列表格式（用于JSON保存）
        cleaned_data_json = {}
        for industry, array in cleaned_data.items():
            rows = array.tolist()
            dates = preprocessor.industry_dates.get(industry, [])
            if dates and len(dates) == len(rows):
                cleaned_data_json[industry] = [
                    [dates[i]] + rows[i]
                    for i in range(len(rows))
                ]
            else:
                cleaned_data_json[industry] = rows
        
        Path(output_data_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_data_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data_json, f, ensure_ascii=False, indent=2)
    
    if output_relation_path:
        if verbose:
            print(f"保存清洗后的关系文件到: {output_relation_path}")
        Path(output_relation_path).parent.mkdir(parents=True, exist_ok=True)
        valid_relation_df.to_csv(output_relation_path, index=False, encoding='utf-8')

    # 保存报告
    # if save_report:
    #     if report_path is None:
    #         report_path = Path(data_path).parent / "cleaning_report.json"
    #     preprocessor.save_cleaning_report(report_path)
    
    return cleaned_data, valid_relation_df, preprocessor

