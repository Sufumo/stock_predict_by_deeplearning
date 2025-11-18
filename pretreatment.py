"""
数据预处理流程脚本
完成数据清洗、日期范围过滤、NaN值处理和统计信息输出
"""
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict

from components.data_preprocessor import DataPreprocessor, preprocess_data


def check_files_exist(data_path: str, relation_path: str) -> bool:
    """检查输入文件是否存在"""
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return False
    
    if not os.path.exists(relation_path):
        print(f"❌ 错误: 关系文件不存在: {relation_path}")
        return False
    
    return True


def validate_data_format(data_path: str) -> bool:
    """验证数据格式"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print("❌ 错误: 数据格式不正确，应该是字典格式")
            return False
        
        # 检查至少一个行业的数据格式
        sample_industry = list(data.keys())[0] if data else None
        if sample_industry and data[sample_industry]:
            sample_kline = data[sample_industry][0]
            if not isinstance(sample_kline, list) or len(sample_kline) < 7:
                print("❌ 错误: K线数据格式不正确，每个K线应该是包含至少7个元素的列表")
                print(f"   示例: [日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, ...]")
                return False
        
        return True
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON解析失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 错误: 验证数据格式时出错: {e}")
        return False


def print_preprocessing_summary(preprocessor: DataPreprocessor, 
                                cleaned_data: Dict,
                                valid_relation_df: pd.DataFrame):
    """打印预处理摘要"""
    print("\n" + "=" * 80)
    print("预处理摘要")
    print("=" * 80)
    
    stats = preprocessor.stats
    print(f"\n📊 数据统计:")
    print(f"  总行业数: {stats['total_industries']}")
    print(f"  有效行业数: {stats['valid_industries']}")
    print(f"  移除行业数: {len(stats['removed_industries'])}")
    print(f"  总样本数: {stats['total_samples_after']:,}")
    print(f"  日期范围: {preprocessor.start_date.strftime('%Y-%m-%d')} 到 {preprocessor.end_date.strftime('%Y-%m-%d')}")
    
    if stats['removed_industries']:
        print(f"\n⚠️  移除的行业 ({len(stats['removed_industries'])}个):")
        for removed in stats['removed_industries'][:10]:  # 只显示前10个
            print(f"    - {removed['industry']}: {removed['reason']}")
        if len(stats['removed_industries']) > 10:
            print(f"    ... 还有 {len(stats['removed_industries']) - 10} 个行业被移除")
    
    # 样本数统计
    if cleaned_data:
        samples_list = [len(arr) for arr in cleaned_data.values()]
        print(f"\n📈 样本数分布:")
        print(f"  最小值: {min(samples_list):,}")
        print(f"  最大值: {max(samples_list):,}")
        print(f"  平均值: {np.mean(samples_list):.0f}")
        print(f"  中位数: {np.median(samples_list):.0f}")
        print(f"  标准差: {np.std(samples_list):.0f}")
    
    # NaN统计
    total_nan_before = sum(s.get('before', 0) for s in stats['nan_counts'].values())
    total_nan_after = sum(s.get('after', 0) for s in stats['nan_counts'].values())
    if total_nan_before > 0:
        print(f"\n🔧 NaN值处理:")
        print(f"  处理前NaN总数: {total_nan_before:,}")
        print(f"  处理后NaN总数: {total_nan_after:,}")
        if total_nan_before > 0:
            print(f"  处理率: {(1 - total_nan_after / total_nan_before) * 100:.2f}%")
    
    # 日期过滤统计
    total_date_filtered = sum(stats['date_filtered_counts'].values())
    if total_date_filtered > 0:
        print(f"\n📅 日期过滤:")
        print(f"  过滤掉的样本总数: {total_date_filtered:,}")


def save_sample_counts(sample_counts: Dict[str, int], output_path: str):
    """保存每个行业的样本数到文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为DataFrame便于查看
    df = pd.DataFrame([
        {'industry': industry, 'samples': count}
        for industry, count in sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
    ])
    
    # 保存为CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n💾 样本数统计已保存到: {csv_path}")
    
    # 保存为JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_counts, f, ensure_ascii=False, indent=2)
    print(f"💾 样本数统计已保存到: {json_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("数据预处理流程")
    print("=" * 80)
    
    # ========== 配置参数 ==========
    # 输入文件路径
    data_path = "./data/industry_kline_data.json"
    relation_path = "./data/industry_relation.csv"
    
    # 输出文件路径
    output_data_path = "./data/industry_kline_data_cleaned.json"
    output_relation_path = "./data/industry_relation_cleaned.csv"
    report_path = "./data/cleaning_report.json"
    sample_counts_path = "./data/industry_sample_counts"
    
    # 预处理参数
    start_date = '2021-12-01'  # 开始日期
    end_date = '2025-11-17'    # 结束日期
    nan_strategy = 'forward_fill'  # NaN处理策略：forward_fill（前向填充）
    min_valid_samples = 100  # 每个行业最少有效样本数
    
    # 显示参数
    top_n_industries = 20  # 显示前N个行业的详细统计
    
    # ========== 文件检查 ==========
    print("\n📁 检查输入文件...")
    if not check_files_exist(data_path, relation_path):
        sys.exit(1)
    
    print("✅ 输入文件检查通过")
    
    # ========== 数据格式验证 ==========
    print("\n🔍 验证数据格式...")
    if not validate_data_format(data_path):
        sys.exit(1)
    
    print("✅ 数据格式验证通过")
    
    # ========== 执行预处理 ==========
    print("\n🚀 开始数据预处理...")
    print(f"   日期范围: {start_date} 到 {end_date}")
    print(f"   NaN处理策略: {nan_strategy}")
    print(f"   最少样本数: {min_valid_samples}")
    print("-" * 80)
    
    try:
        cleaned_data, valid_relation_df, preprocessor = preprocess_data(
            data_path=data_path,
            relation_path=relation_path,
            output_data_path=output_data_path,
            output_relation_path=output_relation_path,
            start_date=start_date,
            end_date=end_date,
            nan_strategy=nan_strategy,
            min_valid_samples=min_valid_samples,
            save_report=True,
            report_path=report_path,
            verbose=True
        )
        
        print("\n✅ 数据预处理完成！")
        
    except Exception as e:
        print(f"\n❌ 预处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 打印统计信息 ==========
    print("\n📊 打印详细统计信息...")
    preprocessor.print_industry_stats(sort_by='samples', top_n=top_n_industries)
    
    # ========== 打印预处理摘要 ==========
    print_preprocessing_summary(preprocessor, cleaned_data, valid_relation_df)
    
    # ========== 获取并保存样本数统计 ==========
    print("\n📈 获取各行业样本数...")
    sample_counts = preprocessor.get_industry_sample_counts()
    
    # 打印样本数统计（前20个）
    print("\n样本数最多的前20个行业:")
    sorted_counts = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (industry, count) in enumerate(sorted_counts[:20], 1):
        print(f"  {i:2d}. {industry:<20} {count:>8,} 样本")
    
    # 保存样本数统计
    save_sample_counts(sample_counts, sample_counts_path)
    
    # ========== 验证输出文件 ==========
    print("\n🔍 验证输出文件...")
    if os.path.exists(output_data_path):
        file_size = os.path.getsize(output_data_path) / (1024 * 1024)  # MB
        print(f"✅ 清洗后的数据文件已保存: {output_data_path} ({file_size:.2f} MB)")
    else:
        print(f"⚠️  警告: 输出数据文件不存在: {output_data_path}")
    
    if os.path.exists(output_relation_path):
        print(f"✅ 清洗后的关系文件已保存: {output_relation_path}")
    else:
        print(f"⚠️  警告: 输出关系文件不存在: {output_relation_path}")
    
    if os.path.exists(report_path):
        print(f"✅ 清洗报告已保存: {report_path}")
    
    # ========== 完成 ==========
    print("\n" + "=" * 80)
    print("✅ 数据预处理流程完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  📄 清洗后的数据: {output_data_path}")
    print(f"  📄 清洗后的关系: {output_relation_path}")
    print(f"  📄 清洗报告: {report_path}")
    print(f"  📄 样本数统计: {sample_counts_path}.csv / {sample_counts_path}.json")
    print("\n可以开始使用清洗后的数据进行训练了！")


if __name__ == "__main__":
    main()

