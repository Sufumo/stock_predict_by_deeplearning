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
        use_dwt=config.model.use_dwt,
        num_industries=config.model.num_industries,
        use_industry_embedding=config.model.use_industry_embedding,
        embedding_fusion_alpha=config.model.embedding_fusion_alpha
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

        # 断点续训配置
        resume_from_checkpoint = getattr(config.training, 'resume_from_checkpoint', True)
        load_previous_fold = getattr(config.training, 'load_previous_fold', False)
        
        fold_results = trainer.k_fold_validate(
            dataset=dataset,
            adj_matrix=adj_matrix_tensor,
            n_splits=config.data.n_splits,
            min_train_size=config.data.min_train_size,
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            save_dir=config.training.save_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            load_previous_fold=load_previous_fold
        )

        # 可视化K折结果
        if config.visualization.plot_training_curves:
            vis = Visualizer(save_dir=config.visualization.save_dir, dpi=config.visualization.dpi)
            vis.plot_kfold_results(fold_results, save_name="kfold_results.png")

        # K折验证模式下，best_model.pth已在k_fold_validate中保存
        # 检查文件是否存在
        if not save_path.exists():
            print(f"\n⚠ Warning: best_model.pth was not created. Check individual fold models: fold_X_best.pth")

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

        # ⭐ 6. 可视化行业嵌入和子图结构
        if config.model.use_industry_embedding:
            print(f"\n{'='*60}")
            print("Step 6: Visualizing Industry Embeddings and Subgraph")
            print(f"{'='*60}")

            try:
                import json
                # 加载行业名称
                industry_list_path = Path(config.data.data_dir) / config.data.industry_list_file
                with open(industry_list_path, 'r', encoding='utf-8') as f:
                    industry_names = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load industry names: {e}")
                industry_names = None

            # 提取行业嵌入
            embeddings = model.industry_embeddings.weight.detach().cpu().numpy()

            # 可视化行业嵌入相似度
            vis.plot_embedding_similarity(
                embeddings,
                industry_names=industry_names,
                top_k=30,
                save_name="industry_embedding_similarity.png"
            )

            # 可视化一个示例batch的子图结构
            # 获取一个验证batch
            sample_batch = next(iter(val_loader))
            sample_industry_indices = sample_batch['industry_idx'].numpy()

            # 获取唯一行业索引
            unique_batch_nodes = np.unique(sample_industry_indices).tolist()

            # 构建子图(模拟_process_subgraph的逻辑)
            batch_and_neighbors = set(unique_batch_nodes)
            adj_np = adj_matrix_tensor.cpu().numpy()
            for idx in unique_batch_nodes:
                neighbors = np.where(adj_np[idx] > 0)[0]
                batch_and_neighbors.update(neighbors.tolist())

            subgraph_nodes = sorted(list(batch_and_neighbors))

            # 提取子图邻接矩阵
            subgraph_adj = adj_np[np.ix_(subgraph_nodes, subgraph_nodes)]

            # 可视化子图结构
            vis.plot_subgraph_structure(
                subgraph_nodes=subgraph_nodes,
                batch_nodes=unique_batch_nodes,
                adj_matrix=subgraph_adj,
                industry_names=industry_names,
                save_name="subgraph_structure_example.png"
            )

            print("✓ Industry embedding and subgraph visualizations completed")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    # 检查模型文件是否存在
    if save_path.exists():
        print(f"Model saved to: {save_path}")
    else:
        print(f"⚠ Model file not found at: {save_path}")
        if config.data.use_kfold:
            print(f"   Check individual fold models in: {config.training.save_dir}")
    print(f"Visualizations saved to: {config.visualization.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
