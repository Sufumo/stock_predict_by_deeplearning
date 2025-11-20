"""
推理预测脚本
加载训练好的模型,对新数据进行预测并生成行业排名
支持横截面模式：每个时间步包含所有86个行业的数据
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from components.model import IndustryStockModel
from components.data_loader import IndustryDataLoader, IndustryDataset
from components.config_loader import load_config, get_device


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> IndustryStockModel:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 模型检查点路径
        config: 配置对象
        device: 计算设备

    Returns:
        加载好的模型
    """
    # 创建模型
    model = IndustryStockModel(
        input_features=config.model.input_features,
        time_encoder_dim=config.model.time_encoder.d_model,
        compression_dim=config.model.compression.out_features,
        gat_hidden_dim=config.model.gat.hidden_features,
        gat_output_dim=config.model.gat.out_features,
        num_classes=config.data.num_classes,
        num_heads=config.model.gat.num_heads,
        num_gat_layers=config.model.gat.num_layers,
        dropout=config.model.gat.dropout,
        use_dwt=config.model.use_dwt
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_accuracy' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_accuracy']:.2f}%")

    return model


def predict_industries(model: IndustryStockModel, dataloader: DataLoader,
                      adj_matrix: torch.Tensor, device: torch.device,
                      industry_names: list) -> pd.DataFrame:
    """
    对所有行业进行预测

    Args:
        model: 模型
        dataloader: 数据加载器
        adj_matrix: 邻接矩阵
        device: 设备
        industry_names: 行业名称列表

    Returns:
        包含预测结果的DataFrame
    """
    model.eval()
    # ⭐ 确保adj_matrix是PyTorch tensor
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)
    adj_matrix = adj_matrix.to(device)

    all_predictions = []
    all_industry_indices = []

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            masks = batch['mask'].to(device)
            industry_indices = batch['industry_idx'].to(device)

            batch_size, max_seq_len, features = sequences.shape

            # 提取不同时间窗口的数据
            x_80 = sequences
            x_40 = sequences[:, -40:, :]
            x_20 = sequences[:, -20:, :]

            mask_80 = masks
            mask_40 = masks[:, -40:]
            mask_20 = masks[:, -20:]

            # 预测
            predictions, _ = model(
                x_20, x_40, x_80,
                mask_20, mask_40, mask_80,
                adj_matrix, industry_indices
            )

            all_predictions.append(predictions.cpu().numpy())
            all_industry_indices.append(industry_indices.cpu().numpy())

    # 合并结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_industry_indices = np.concatenate(all_industry_indices, axis=0)

    # 创建结果DataFrame
    results = []
    for idx, pred_probs in zip(all_industry_indices, all_predictions):
        pred_class = np.argmax(pred_probs)
        pred_score = np.max(pred_probs)  # 使用最大概率作为信心分数

        results.append({
            'industry_idx': idx,
            'industry_name': industry_names[idx] if idx < len(industry_names) else f'Industry_{idx}',
            'predicted_class': pred_class,
            'confidence': pred_score,
            'Q1_prob': pred_probs[0],
            'Q2_prob': pred_probs[1],
            'Q3_prob': pred_probs[2],
            'Q4_prob': pred_probs[3],
            'Q5_prob': pred_probs[4],
        })

    df = pd.DataFrame(results)

    # 按置信度排序
    df = df.sort_values('confidence', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


def predict_cross_sectional(model: IndustryStockModel, cross_sectional_batches: list,
                           adj_matrix: torch.Tensor, device: torch.device,
                           industry_names: list) -> pd.DataFrame:
    """
    横截面预测：对每个时间步的所有行业进行预测
    
    Args:
        model: 模型
        cross_sectional_batches: 横截面批次列表（来自prepare_cross_sectional_data）
        adj_matrix: 邻接矩阵
        device: 设备
        industry_names: 行业名称列表
    
    Returns:
        包含所有时间步预测结果的DataFrame
    """
    model.eval()
    # ⭐ 确保adj_matrix是PyTorch tensor
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.FloatTensor(adj_matrix)
    adj_matrix = adj_matrix.to(device)
    
    all_results = []
    
    with torch.no_grad():
        for batch_data in cross_sectional_batches:
            time_idx = batch_data['time_index']
            sequences = torch.FloatTensor(batch_data['sequences']).to(device)
            masks = torch.FloatTensor(batch_data['masks']).to(device)
            industry_indices = torch.LongTensor(batch_data['industry_indices']).to(device)
            targets = batch_data['targets']  # 真实标签
            
            batch_size, max_seq_len, features = sequences.shape
            
            # 提取不同时间窗口的数据
            x_80 = sequences
            x_40 = sequences[:, -40:, :]
            x_20 = sequences[:, -20:, :]
            
            mask_80 = masks
            mask_40 = masks[:, -40:]
            mask_20 = masks[:, -20:]
            
            # 预测
            predictions, _ = model(
                x_20, x_40, x_80,
                mask_20, mask_40, mask_80,
                adj_matrix, industry_indices
            )
            
            pred_probs = predictions.cpu().numpy()  # [batch_size, 5]
            pred_classes = np.argmax(pred_probs, axis=1)  # [batch_size]
            pred_confidences = np.max(pred_probs, axis=1)  # [batch_size]
            
            # 为每个行业创建结果
            for i, (idx, pred_class, confidence, target) in enumerate(
                zip(industry_indices.cpu().numpy(), pred_classes, pred_confidences, targets)
            ):
                all_results.append({
                    'time_index': time_idx,
                    'industry_idx': idx,
                    'industry_name': industry_names[idx] if idx < len(industry_names) else f'Industry_{idx}',
                    'predicted_class': pred_class,
                    'true_class': target,
                    'confidence': confidence,
                    'Q1_prob': pred_probs[i][0],
                    'Q2_prob': pred_probs[i][1],
                    'Q3_prob': pred_probs[i][2],
                    'Q4_prob': pred_probs[i][3],
                    'Q5_prob': pred_probs[i][4],
                })
    
    df = pd.DataFrame(all_results)
    return df


def backtest_strategy(predictions_df: pd.DataFrame, data_loader_obj: IndustryDataLoader,
                     industry_names: list, top_percentile: int = 4) -> pd.DataFrame:
    """
    回测策略：选择预测为前20%（类别4）的行业持有，计算收益率
    
    Args:
        predictions_df: 预测结果DataFrame（来自predict_cross_sectional，需包含real_return列）
        data_loader_obj: 数据加载器对象（用于获取价格数据）
        industry_names: 行业名称列表
        top_percentile: 选择的分位数类别（4表示前20%）
    
    Returns:
        包含每个时间步收益率和累计收益率的DataFrame
    """
    # 确保predictions_df包含real_return列
    if 'real_return' not in predictions_df.columns:
        predictions_df = calculate_real_returns(predictions_df, data_loader_obj, industry_names)
    
    # 按时间步分组
    time_steps = sorted(predictions_df['time_index'].unique())
    
    backtest_results = []
    cumulative_return = 1.0  # 初始资金为1
    
    for time_idx in time_steps:
        # 获取该时间步的所有预测
        time_predictions = predictions_df[predictions_df['time_index'] == time_idx].copy()
        
        # 选择预测为top_percentile（类别4）的行业
        selected_industries = time_predictions[
            time_predictions['predicted_class'] == top_percentile
        ].copy()
        
        if len(selected_industries) == 0:
            # 如果没有选中的行业，收益率为0
            period_return = 0.0
            cumulative_return *= (1 + period_return)
            backtest_results.append({
                'time_index': time_idx,
                'num_selected': 0,
                'period_return': period_return,
                'cumulative_return': cumulative_return,
                'selected_industries': ''
            })
            continue
        
        # 计算该时间步的平均收益率（等权重）
        # 使用真实收益率（real_return）
        period_returns = selected_industries['real_return'].dropna().tolist()
        
        if len(period_returns) == 0:
            # 如果没有有效的收益率数据，收益率为0
            period_return = 0.0
        else:
            # 等权重平均收益率
            period_return = np.mean(period_returns)
        
        cumulative_return *= (1 + period_return)
        
        backtest_results.append({
            'time_index': time_idx,
            'num_selected': len(selected_industries),
            'period_return': period_return,
            'cumulative_return': cumulative_return,
            'selected_industries': ', '.join(selected_industries['industry_name'].tolist()[:5])  # 前5个
        })
    
    backtest_df = pd.DataFrame(backtest_results)
    return backtest_df


def calculate_real_returns(predictions_df: pd.DataFrame, data_loader_obj: IndustryDataLoader,
                           industry_names: list, future_days: int = 30) -> pd.DataFrame:
    """
    计算真实的未来收益率（基于实际价格数据）
    
    Args:
        predictions_df: 预测结果DataFrame
        data_loader_obj: 数据加载器对象
        industry_names: 行业名称列表
        future_days: 未来天数
    
    Returns:
        添加了真实收益率的DataFrame
    """
    # 加载原始数据
    if data_loader_obj.raw_data is None:
        data_loader_obj.load_data()
    
    # 为每个预测计算真实收益率
    real_returns = []
    
    for _, row in predictions_df.iterrows():
        time_idx = row['time_index']
        industry_idx = int(row['industry_idx'])
        industry_name = industry_names[industry_idx]
        
        # 解析该行业的数据
        data = data_loader_obj.parse_kline_data(industry_name)
        if data is None or time_idx + 80 + future_days > len(data):
            real_returns.append(np.nan)
            continue
        
        # 计算真实收益率
        current_prices = data[:, 1]  # 收盘价
        start_price = current_prices[time_idx + 80 - 1]  # 当前时间步的最后一天
        end_price = current_prices[time_idx + 80 + future_days - 1]  # 未来30天后的价格
        
        if start_price > 0:
            real_return = (end_price - start_price) / start_price
        else:
            real_return = 0.0
        
        real_returns.append(real_return)
    
    predictions_df['real_return'] = real_returns
    return predictions_df


def plot_backtest_results(backtest_df: pd.DataFrame, save_path: str = "predictions/backtest_results.png"):
    """
    绘制回测结果：累计收益率折线图
    
    Args:
        backtest_df: 回测结果DataFrame
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制累计收益率
    plt.plot(backtest_df['time_index'], backtest_df['cumulative_return'], 
             label='Cumulative Return', linewidth=2)
    
    # 添加基准线（收益率为0）
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.title('Backtest Results: Cumulative Return Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    final_return = backtest_df['cumulative_return'].iloc[-1]
    total_return = (final_return - 1.0) * 100
    avg_period_return = backtest_df['period_return'].mean() * 100
    
    stats_text = f'Final Return: {total_return:.2f}%\n'
    stats_text += f'Avg Period Return: {avg_period_return:.4f}%\n'
    stats_text += f'Total Periods: {len(backtest_df)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n回测结果图表已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="MMF-GAT Industry Prediction")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="predictions/predictions.csv",
                       help="Output CSV file path")
    parser.add_argument("--mode", type=str, default="cross_sectional", 
                       choices=["standard", "cross_sectional"],
                       help="Prediction mode: standard or cross_sectional")
    parser.add_argument("--backtest", action="store_true",
                       help="Run backtest strategy")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = get_device(config.training.device)

    print(f"Using device: {device}")
    print(f"Loading data from {config.data.data_dir}...")

    # 加载数据
    data_loader_obj = IndustryDataLoader(
        data_dir=config.data.data_dir,
        window_sizes=[config.data.window_20, config.data.window_40, config.data.window_80],
        future_days=config.data.future_days,
        num_classes=config.data.num_classes
    )

    # ⭐ 确保数据已加载（构建邻接矩阵需要industry_list）
    if data_loader_obj.raw_data is None:
        data_loader_obj.load_data()
    
    # 获取行业列表
    try:
        import json
        industry_list_path = Path(config.data.data_dir) / config.data.industry_list_file
        with open(industry_list_path, 'r', encoding='utf-8') as f:
            industry_names = json.load(f)
    except:
        # 如果无法加载，使用data_loader中的行业列表
        if data_loader_obj.industry_list is not None:
            industry_names = data_loader_obj.industry_list
        else:
            industry_names = [f'Industry_{i}' for i in range(86)]

    # ⭐ 确保adj_matrix已构建
    if data_loader_obj.adj_matrix is None:
        data_loader_obj.adj_matrix = data_loader_obj.build_adjacency_matrix()
    adj_matrix = data_loader_obj.adj_matrix

    # 加载模型
    model = load_model(args.checkpoint, config, device)

    if args.mode == "cross_sectional":
        # ⭐ 横截面模式：每个时间步包含所有86个行业
        print("\n" + "="*60)
        print("横截面预测模式")
        print("="*60)
        
        # 准备横截面数据
        print("\n准备横截面数据...")
        cross_sectional_batches = data_loader_obj.prepare_cross_sectional_data(
            window_sizes=[config.data.window_20, config.data.window_40, config.data.window_80],
            future_days=config.data.future_days
        )
        
        # 预测
        print("\n进行预测...")
        predictions_df = predict_cross_sectional(
            model, cross_sectional_batches, adj_matrix, device, industry_names
        )
        
        # 计算真实收益率
        print("\n计算真实收益率...")
        predictions_df = calculate_real_returns(
            predictions_df, data_loader_obj, industry_names, config.data.future_days
        )
        
        # 保存预测结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {output_path}")
        
        # 显示每个时间步的排名（前10）
        print("\n" + "="*60)
        print("各时间步行业排名（Top 10，预测类别为4的行业）")
        print("="*60)
        time_steps = sorted(predictions_df['time_index'].unique())[:10]  # 显示前10个时间步
        for time_idx in time_steps:
            time_pred = predictions_df[predictions_df['time_index'] == time_idx].copy()
            time_pred = time_pred.sort_values('confidence', ascending=False)
            top_industries = time_pred[time_pred['predicted_class'] == 4].head(10)
            
            print(f"\n时间步 {time_idx}:")
            print(f"  选中行业数: {len(top_industries)}")
            if len(top_industries) > 0:
                print(f"  Top 5: {', '.join(top_industries['industry_name'].head(5).tolist())}")
        
        # 回测策略
        if args.backtest:
            print("\n" + "="*60)
            print("回测策略：选择预测为前20%（类别4）的行业")
            print("="*60)
            
            backtest_df = backtest_strategy(
                predictions_df, data_loader_obj, industry_names, top_percentile=4
            )
            
            # 保存回测结果
            backtest_output = output_path.parent / "backtest_results.csv"
            backtest_df.to_csv(backtest_output, index=False, encoding='utf-8-sig')
            print(f"\n回测结果已保存到: {backtest_output}")
            
            # 绘制收益率折线图
            plot_path = output_path.parent / "backtest_results.png"
            plot_backtest_results(backtest_df, str(plot_path))
            
            # 打印回测统计
            print("\n" + "="*60)
            print("回测统计")
            print("="*60)
            final_return = backtest_df['cumulative_return'].iloc[-1]
            total_return = (final_return - 1.0) * 100
            avg_period_return = backtest_df['period_return'].mean() * 100
            std_period_return = backtest_df['period_return'].std() * 100
            
            print(f"总时间步数: {len(backtest_df)}")
            print(f"最终累计收益率: {total_return:.2f}%")
            print(f"平均每期收益率: {avg_period_return:.4f}%")
            print(f"收益率标准差: {std_period_return:.4f}%")
            print(f"平均选中行业数: {backtest_df['num_selected'].mean():.1f}")
    
    else:
        # 标准模式（原有逻辑）
        print("\n" + "="*60)
        print("标准预测模式")
        print("="*60)
        
        # 创建数据集(使用全部数据进行预测)
        data_dict = data_loader_obj.get_data_dict()
        dataset = IndustryDataset(
            sequences=data_dict['sequences'],
            targets=data_dict['targets'],
            masks=data_dict['masks'],
            industry_indices=data_dict['industry_indices']
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=config.inference.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 预测
        print("\n进行预测...")
        results_df = predict_industries(model, dataloader, adj_matrix, device, industry_names)
        
        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n预测结果已保存到: {output_path}")
        print(f"\nTop 10 Industries by Confidence:")
        print(results_df[['rank', 'industry_name', 'predicted_class', 'confidence']].head(10))
        
        # 输出分位数统计
        print(f"\n预测分布:")
        class_dist = results_df['predicted_class'].value_counts().sort_index()
        for class_idx, count in class_dist.items():
            print(f"  Q{class_idx + 1}: {count} industries ({100 * count / len(results_df):.1f}%)")


if __name__ == "__main__":
    main()
