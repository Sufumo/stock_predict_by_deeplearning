"""
示例训练脚本
使用配置系统进行训练,支持K折验证、金融指标等
"""
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from components.data_loader import IndustryDataLoader, IndustryDataset
from components.model import IndustryStockModel
from components.trainer import Trainer
from components.config_loader import (
    load_config_with_cli, get_device, setup_directories, setup_seed
)
from components.visualizer import Visualizer


def main():
    """主训练流程"""
    # 加载配置和命令行参数
    config, args = load_config_with_cli()

    print("=" * 60)
    print("MMF-GAT Industry Stock Prediction Training")
    print("=" * 60)

    # 设置随机种子
    if config.experiment.seed is not None:
        setup_seed(config.experiment.seed, config.experiment.deterministic)
        print(f"Random seed set to: {config.experiment.seed}")

    # 创建必要目录
    setup_directories(config)

    # 选择设备
    device = get_device(config.training.device)
    print(f"Using device: {device}")

    # 1. 数据加载和预处理
    print(f"\n{'='*60}")
    print("Step 1: Loading Data")
    print(f"{'='*60}")

    data_loader = IndustryDataLoader(
        data_dir=config.data.data_dir,
        window_sizes=[config.data.window_20, config.data.window_40, config.data.window_80],
        future_days=config.data.future_days,
        num_classes=config.data.num_classes
    )

    # 准备数据
    samples, targets, adj_matrix = data_loader.prepare_data()

    print(f"Total samples: {len(samples)}")
    print(f"Number of industries: {adj_matrix.shape[0]}")
    print(f"Label distribution: {np.bincount(targets)}")

    # 获取完整的序列数据（包含masks和industry_indices）
    data_dict = data_loader.get_data_dict()

    # 创建数据集（包含所有必要信息）
    dataset = IndustryDataset(
        sequences=data_dict['sequences'],
        targets=data_dict['targets'],
        masks=data_dict['masks'],
        industry_indices=data_dict['industry_indices']
    )

    # 2. 创建模型
    print(f"\n{'='*60}")
    print("Step 2: Creating Model")
    print(f"{'='*60}")

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 3. 创建训练器
    print(f"\n{'='*60}")
    print("Step 3: Creating Trainer")
    print(f"{'='*60}")

    scheduler_params = None
    if config.training.use_scheduler:
        # 将Config对象转换为字典
        if hasattr(config.training.scheduler_params, 'to_dict'):
            scheduler_params = config.training.scheduler_params.to_dict()
        elif isinstance(config.training.scheduler_params, dict):
            scheduler_params = config.training.scheduler_params
        else:
            # 如果是Config对象但没有to_dict方法，手动转换为字典
            scheduler_params = {
                'mode': getattr(config.training.scheduler_params, 'mode', 'min'),
                'factor': getattr(config.training.scheduler_params, 'factor', 0.5),
                'patience': getattr(config.training.scheduler_params, 'patience', 5),
                'min_lr': getattr(config.training.scheduler_params, 'min_lr', 0.00001)
            }
            # 移除None值
            scheduler_params = {k: v for k, v in scheduler_params.items() if v is not None}

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_scheduler=config.training.use_scheduler,
        scheduler_params=scheduler_params,
        compute_financial_metrics=config.validation.compute_financial_metrics,
        max_grad_norm=config.training.max_grad_norm if config.training.use_grad_clip else None
    )

    # 4. 训练模式选择
    print(f"\n{'='*60}")
    print("Step 4: Training")
    print(f"{'='*60}")

    adj_matrix_tensor = torch.FloatTensor(adj_matrix)
    save_path = Path(config.training.save_dir) / "best_model.pth"

    if config.data.use_kfold:
        # K折交叉验证
        print(f"Using {config.data.n_splits}-Fold Cross-Validation")

        fold_results = trainer.k_fold_validate(
            dataset=dataset,
            adj_matrix=adj_matrix_tensor,
            n_splits=config.data.n_splits,
            min_train_size=config.data.min_train_size,
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            save_dir=config.training.save_dir
        )

        # 可视化K折结果
        if config.visualization.plot_training_curves:
            vis = Visualizer(save_dir=config.visualization.save_dir, dpi=config.visualization.dpi)
            vis.plot_kfold_results(fold_results, save_name="kfold_results.png")

    else:
        # 标准训练/验证分割
        print(f"Using standard train/val split ({config.data.train_ratio}/{config.data.val_ratio})")

        train_size = int(config.data.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.experiment.seed)
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )

        # 训练
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            adj_matrix=adj_matrix_tensor,
            num_epochs=config.training.num_epochs,
            save_path=str(save_path)
        )

        # 可视化训练过程
        if config.visualization.plot_training_curves:
            vis = Visualizer(save_dir=config.visualization.save_dir, dpi=config.visualization.dpi)
            vis.plot_training_curves(history, save_name="training_curves.png")

        print("\nTraining completed!")
        print(f"Best validation accuracy: {max(history['val']['accuracy']):.2f}%")

        if config.validation.compute_financial_metrics and 'IC' in history['val']:
            best_ic_idx = np.argmax(history['val']['IC'])
            print(f"Best IC: {history['val']['IC'][best_ic_idx]:.4f} at epoch {best_ic_idx + 1}")

        # 5. 最终预测和可视化
        if config.visualization.plot_confusion_matrix:
            print(f"\n{'='*60}")
            print("Step 5: Final Evaluation")
            print(f"{'='*60}")

            predictions, pred_classes = trainer.predict(val_loader, adj_matrix_tensor)

            # 获取真实标签
            val_targets = []
            for batch in val_loader:
                val_targets.extend(batch['target'].numpy())
            val_targets = np.array(val_targets)

            # 绘制混淆矩阵
            vis.plot_confusion_matrix(
                val_targets, pred_classes,
                class_names=[f'Q{i+1}' for i in range(config.data.num_classes)],
                save_name="confusion_matrix.png"
            )

            accuracy = (pred_classes == val_targets).mean() * 100
            print(f"Final validation accuracy: {accuracy:.2f}%")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model saved to: {save_path}")
    print(f"Visualizations saved to: {config.visualization.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
