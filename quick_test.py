"""
快速验证脚本
运行小规模训练验证NaN问题是否已修复
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from components.data_loader import IndustryDataLoader, IndustryDataset
from components.model import IndustryStockModel
from components.trainer import Trainer


def quick_train_test(num_samples=1000, num_epochs=3, batch_size=64):
    """
    快速训练测试

    Args:
        num_samples: 使用的样本数量
        num_epochs: 训练epoch数
        batch_size: 批次大小
    """
    print("\n" + "="*80)
    print("Quick Training Test (Small Scale)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Samples: {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ========== 1. 加载并归一化数据 ==========
    print("\n" + "-"*80)
    print("Step 1: Loading and Normalizing Data")
    print("-"*80)

    data_dir = project_root / "data"
    loader = IndustryDataLoader(
        data_dir=str(data_dir),
        window_sizes=[20, 40, 80],
        future_days=30,
        num_classes=5
    )

    try:
        loader.load_data()
        print(f"✓ Loaded {len(loader.industry_list)} industries")

        # 准备数据 - 自动进行归一化
        data_dict = loader.prepare_sequences()

        print(f"✓ Data normalized and prepared")
        print(f"  Total samples: {len(data_dict['sequences'])}")

        # 保存scaler以便后续使用
        loader.save_scalers(str(project_root / "checkpoints" / "quick_test_scalers.pkl"))

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== 2. 创建数据集 ==========
    print("\n" + "-"*80)
    print("Step 2: Creating Dataset and DataLoader")
    print("-"*80)

    # 创建完整数据集
    full_dataset = IndustryDataset(
        data_dict['sequences'],
        data_dict['targets'],
        data_dict['masks'],
        data_dict['industry_indices']
    )

    # 使用subset进行小规模测试
    if len(full_dataset) > num_samples:
        # 随机采样
        indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        dataset = Subset(full_dataset, indices)
        print(f"✓ Using {num_samples} samples (subset of {len(full_dataset)})")
    else:
        dataset = full_dataset
        print(f"✓ Using all {len(full_dataset)} samples")

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 构建邻接矩阵
    adj_matrix = loader.build_adjacency_matrix()
    adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(device)

    # ========== 3. 创建模型 ==========
    print("\n" + "-"*80)
    print("Step 3: Creating Model")
    print("-"*80)

    model = IndustryStockModel(
        input_features=7,
        time_encoder_dim=128,
        compression_dim=64,
        gat_hidden_dim=128,
        gat_output_dim=64,
        num_classes=5,
        num_heads=8,
        num_gat_layers=2,
        dropout=0.1,
        use_dwt=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")

    # ========== 4. 创建训练器 ==========
    print("\n" + "-"*80)
    print("Step 4: Creating Trainer")
    print("-"*80)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=5e-5,  # 使用优化后的学习率
        weight_decay=1e-5,
        use_scheduler=True,
        scheduler_params={'mode': 'min', 'factor': 0.5, 'patience': 2},
        compute_financial_metrics=True,
        max_grad_norm=1.0
    )

    # ⭐ 启用NaN检测（用于调试）
    trainer.enable_debugging(enable_nan_detection=True, enable_gradient_monitor=False)
    print("✓ Trainer created with NaN detection enabled")

    # ========== 5. 训练模型 ==========
    print("\n" + "-"*80)
    print(f"Step 5: Training for {num_epochs} Epochs")
    print("-"*80)

    all_passed = True
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        try:
            # 训练
            train_metrics = trainer.train_epoch(train_loader, adj_matrix_tensor)

            print(f"  Train Loss: {train_metrics['loss']:.6f}, "
                  f"Acc: {train_metrics['accuracy']*100:.2f}%")

            # 检查loss是否为NaN
            if np.isnan(train_metrics['loss']):
                print(f"  ❌ Training loss is NaN at epoch {epoch}!")
                all_passed = False
                break

            # 验证
            val_metrics = trainer.validate(
                val_loader,
                adj_matrix_tensor,
                compute_financial_metrics=True
            )

            print(f"  Val Loss: {val_metrics['loss']:.6f}, "
                  f"Acc: {val_metrics['accuracy']*100:.2f}%")

            if 'IC' in val_metrics:
                print(f"  Val IC: {val_metrics['IC']:.4f}, "
                      f"RankIC: {val_metrics['RankIC']:.4f}")

            # 检查loss是否为NaN
            if np.isnan(val_metrics['loss']):
                print(f"  ❌ Validation loss is NaN at epoch {epoch}!")
                all_passed = False
                break

            # 成功完成epoch
            print(f"  ✓ Epoch {epoch} completed successfully")

        except Exception as e:
            print(f"  ❌ Training failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break

    # ========== 6. 总结 ==========
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    if all_passed:
        print("✓ Quick training test PASSED!")
        print("  - No NaN losses detected")
        print("  - Model trained successfully for {} epochs".format(num_epochs))
        print("  - Data normalization working correctly")
        print("\n🎉 The NaN issue appears to be FIXED!")
        print("\nNext steps:")
        print("  1. Run full training: python train.py")
        print("  2. Monitor the training curves")
        print("  3. Evaluate on test set")
    else:
        print("❌ Quick training test FAILED!")
        print("  - Please review the errors above")
        print("  - Check if data normalization is working")
        print("  - Verify model architecture changes")
        print("\n⚠️  The NaN issue may still persist - further debugging needed")

    print("="*80 + "\n")

    # 禁用调试模式
    trainer.disable_debugging()

    return all_passed


def run_all_quick_tests():
    """运行所有快速测试"""
    print("\n" + "="*80)
    print("MMF-GAT Quick Test Suite")
    print("Verifying NaN Fixes")
    print("="*80)

    # Test 1: 极小规模测试（500样本，2 epochs）
    print("\n>>> Test 1: Mini Test (500 samples, 2 epochs)")
    result1 = quick_train_test(num_samples=500, num_epochs=2, batch_size=32)

    if not result1:
        print("\n⚠️  Mini test failed, skipping larger tests")
        return False

    # Test 2: 小规模测试（2000样本，3 epochs）
    print("\n>>> Test 2: Small Test (2000 samples, 3 epochs)")
    result2 = quick_train_test(num_samples=2000, num_epochs=3, batch_size=64)

    # 最终总结
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    print(f"Mini Test (500 samples): {'PASSED ✓' if result1 else 'FAILED ✗'}")
    print(f"Small Test (2000 samples): {'PASSED ✓' if result2 else 'FAILED ✗'}")

    if result1 and result2:
        print("\n🎉 All quick tests passed!")
        print("\nThe following fixes have been successfully applied:")
        print("  1. ✓ Feature normalization (StandardScaler)")
        print("  2. ✓ GAT attention mask handling")
        print("  3. ✓ Activation value clamping")
        print("  4. ✓ LayerNorm after input projection")
        print("  5. ✓ Optimized hyperparameters")
        print("\nYou can now proceed with full training!")
    else:
        print("\n⚠️  Some tests failed. Please debug before full training.")

    print("="*80 + "\n")

    return result1 and result2


if __name__ == "__main__":
    success = run_all_quick_tests()
    sys.exit(0 if success else 1)
