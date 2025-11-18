"""
推理预测脚本
加载训练好的模型,对新数据进行预测并生成行业排名
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def main():
    parser = argparse.ArgumentParser(description="MMF-GAT Industry Prediction")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="predictions/predictions.csv",
                       help="Output CSV file path")
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

    # 获取行业列表
    try:
        import json
        industry_list_path = Path(config.data.data_dir) / config.data.industry_list_file
        with open(industry_list_path, 'r', encoding='utf-8') as f:
            industry_names = json.load(f)
    except:
        industry_names = [f'Industry_{i}' for i in range(86)]

    # 创建数据集(使用全部数据进行预测)
    samples, targets, adj_matrix = data_loader_obj.prepare_data()
    dataset = IndustryDataset(samples, targets)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")

    # 加载模型
    model = load_model(args.checkpoint, config, device)

    # 预测
    print("Predicting...")
    results_df = predict_industries(model, dataloader, adj_matrix, device, industry_names)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nPredictions saved to {output_path}")
    print(f"\nTop 10 Industries by Confidence:")
    print(results_df[['rank', 'industry_name', 'predicted_class', 'confidence']].head(10))

    # 输出分位数统计
    print(f"\nPrediction Distribution:")
    class_dist = results_df['predicted_class'].value_counts().sort_index()
    for class_idx, count in class_dist.items():
        print(f"  Q{class_idx + 1}: {count} industries ({100 * count / len(results_df):.1f}%)")


if __name__ == "__main__":
    main()
