"""
前向传播测试脚本
测试单个batch的前向传播，检查每层输出是否有NaN/Inf
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from components.data_loader import IndustryDataLoader, IndustryDataset
from components.model import IndustryStockModel
from components.monitor import ActivationMonitor


def test_forward_pass():
    """测试模型前向传播"""
    print("\n" + "="*80)
    print("Forward Propagation Test")
    print("="*80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. 加载数据
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

        # 准备数据（会自动进行归一化）
        data_dict = loader.prepare_sequences()
        sequences = data_dict['sequences']
        targets = data_dict['targets']
        masks = data_dict['masks']
        industry_indices = data_dict['industry_indices']

        print(f"✓ Prepared {len(sequences)} sequences")
        print(f"  Sequence shape: {sequences.shape}")

        # 检查归一化效果
        print(f"\n归一化后的特征统计:")
        for i, name in enumerate(['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Return']):
            feat_data = sequences[:, :, i].flatten()
            print(f"  {name:8s} - Mean: {feat_data.mean():7.3f}, Std: {feat_data.std():7.3f}, "
                  f"Range: [{feat_data.min():7.2f}, {feat_data.max():7.2f}]")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 创建数据集和dataloader
    print("\n" + "-"*80)
    print("Step 2: Creating DataLoader")
    print("-"*80)

    dataset = IndustryDataset(sequences, targets, masks, industry_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"✓ Created DataLoader with batch_size=32")
    print(f"  Total batches: {len(dataloader)}")

    # 构建邻接矩阵
    adj_matrix = loader.build_adjacency_matrix()
    adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(device)
    print(f"✓ Adjacency matrix shape: {adj_matrix.shape}")

    # 3. 创建模型
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
    ).to(device)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 4. 注册activation monitor
    print("\n" + "-"*80)
    print("Step 4: Setting up Activation Monitor")
    print("-"*80)

    activation_monitor = ActivationMonitor(model)
    activation_monitor.register_hooks()
    print("✓ Activation monitor registered")

    # 5. 前向传播测试
    print("\n" + "-"*80)
    print("Step 5: Forward Pass Test")
    print("-"*80)

    model.eval()  # 设置为评估模式
    all_tests_passed = True

    with torch.no_grad():
        # 获取第一个batch
        batch = next(iter(dataloader))

        sequence = batch['sequence'].to(device)
        target = batch['target'].to(device)
        mask = batch['mask'].to(device)
        industry_idx = batch['industry_idx'].to(device)

        print(f"\nBatch info:")
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Mask shape: {mask.shape}")

        # 准备不同窗口的输入
        max_window = sequence.shape[1]  # 80
        x_20 = sequence[:, -20:, :]
        x_40 = sequence[:, -40:, :]
        x_80 = sequence

        mask_20 = mask[:, -20:]
        mask_40 = mask[:, -40:]
        mask_80 = mask

        print(f"\n  x_20 shape: {x_20.shape}")
        print(f"  x_40 shape: {x_40.shape}")
        print(f"  x_80 shape: {x_80.shape}")

        try:
            # 执行前向传播
            print(f"\nExecuting forward pass...")
            predictions, time_features = model(
                x_20, x_40, x_80,
                mask_20, mask_40, mask_80,
                adj_matrix_tensor,
                industry_idx
            )

            print(f"✓ Forward pass successful")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Time features shape: {time_features.shape}")

            # 检查输出
            print(f"\nOutput statistics:")
            print(f"  Predictions - Mean: {predictions.mean().item():.4f}, "
                  f"Std: {predictions.std().item():.4f}, "
                  f"Min: {predictions.min().item():.4f}, "
                  f"Max: {predictions.max().item():.4f}")

            # 检查NaN/Inf
            if torch.isnan(predictions).any():
                print(f"  ❌ NaN detected in predictions!")
                all_tests_passed = False
            elif torch.isinf(predictions).any():
                print(f"  ❌ Inf detected in predictions!")
                all_tests_passed = False
            else:
                print(f"  ✓ No NaN/Inf in predictions")

            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predictions, target)
            print(f"\n  Loss: {loss.item():.6f}")

            if torch.isnan(loss):
                print(f"  ❌ Loss is NaN!")
                all_tests_passed = False
            elif torch.isinf(loss):
                print(f"  ❌ Loss is Inf!")
                all_tests_passed = False
            else:
                print(f"  ✓ Loss is valid")

            # 计算准确率
            pred_classes = predictions.argmax(dim=1)
            accuracy = (pred_classes == target).float().mean().item()
            print(f"  Accuracy: {accuracy*100:.2f}%")

        except Exception as e:
            print(f"\n❌ Forward pass failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_tests_passed = False

    # 6. 打印activation统计
    print("\n" + "-"*80)
    print("Step 6: Activation Statistics")
    print("-"*80)

    activation_monitor.print_summary(top_k=15)

    # 7. 检查所有层的激活值
    print("\n" + "-"*80)
    print("Step 7: Checking All Activations for NaN/Inf")
    print("-"*80)

    activation_stats = activation_monitor.get_stats()
    nan_layers = [name for name, stats in activation_stats.items() if stats['has_nan']]
    inf_layers = [name for name, stats in activation_stats.items() if stats['has_inf']]

    if nan_layers:
        print(f"\n❌ Found NaN in {len(nan_layers)} layers:")
        for layer in nan_layers:
            print(f"   - {layer}")
        all_tests_passed = False
    else:
        print("\n✓ No NaN detected in any layer")

    if inf_layers:
        print(f"\n❌ Found Inf in {len(inf_layers)} layers:")
        for layer in inf_layers:
            print(f"   - {layer}")
        all_tests_passed = False
    else:
        print("✓ No Inf detected in any layer")

    # 清理
    activation_monitor.remove_hooks()

    # 最终结果
    print("\n" + "="*80)
    if all_tests_passed:
        print("✓ All Forward Pass Tests PASSED!")
    else:
        print("❌ Some tests FAILED - check output above")
    print("="*80 + "\n")

    return all_tests_passed


def test_multiple_batches(num_batches=5):
    """测试多个batch的前向传播"""
    print("\n" + "="*80)
    print(f"Testing {num_batches} Batches Forward Propagation")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data_dir = project_root / "data"
    loader = IndustryDataLoader(
        data_dir=str(data_dir),
        window_sizes=[20, 40, 80],
        future_days=30,
        num_classes=5
    )

    loader.load_data()
    data_dict = loader.prepare_sequences()

    dataset = IndustryDataset(
        data_dict['sequences'],
        data_dict['targets'],
        data_dict['masks'],
        data_dict['industry_indices']
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    adj_matrix_tensor = torch.FloatTensor(loader.build_adjacency_matrix()).to(device)

    # 创建模型
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
    ).to(device)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    print(f"\nTesting {num_batches} batches...")
    all_passed = True

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            industry_idx = batch['industry_idx'].to(device)

            x_20 = sequence[:, -20:, :]
            x_40 = sequence[:, -40:, :]
            x_80 = sequence

            mask_20 = mask[:, -20:]
            mask_40 = mask[:, -40:]
            mask_80 = mask

            try:
                predictions, _ = model(
                    x_20, x_40, x_80,
                    mask_20, mask_40, mask_80,
                    adj_matrix_tensor,
                    industry_idx
                )

                loss = criterion(predictions, target)

                has_nan = torch.isnan(predictions).any() or torch.isnan(loss)
                has_inf = torch.isinf(predictions).any() or torch.isinf(loss)

                status = "✓" if not (has_nan or has_inf) else "✗"
                print(f"  Batch {i+1}/{num_batches}: {status} Loss={loss.item():.6f}")

                if has_nan or has_inf:
                    all_passed = False

            except Exception as e:
                print(f"  Batch {i+1}/{num_batches}: ✗ Error - {e}")
                all_passed = False

    print(f"\n{'✓ All batches passed!' if all_passed else '❌ Some batches failed!'}\n")
    return all_passed


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MMF-GAT Forward Propagation Test Suite")
    print("="*80)

    # Test 1: Single batch forward pass with detailed monitoring
    result1 = test_forward_pass()

    # Test 2: Multiple batches
    result2 = test_multiple_batches(num_batches=10)

    # Final summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Single Batch Test: {'PASSED ✓' if result1 else 'FAILED ✗'}")
    print(f"Multiple Batch Test: {'PASSED ✓' if result2 else 'FAILED ✗'}")

    if result1 and result2:
        print("\n🎉 All tests passed! Model is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    print("="*80 + "\n")
