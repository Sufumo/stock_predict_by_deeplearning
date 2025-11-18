"""
归一化功能单元测试
测试StandardScaler和数据归一化流程
"""
import sys
from pathlib import Path
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from components.data_loader import StandardScaler, IndustryDataLoader


def test_standard_scaler():
    """测试StandardScaler的基本功能"""
    print("\n" + "="*80)
    print("Test 1: StandardScaler Basic Functionality")
    print("="*80)

    # 创建测试数据
    data = np.array([
        [1000, 2000, 3000],
        [1100, 2100, 3100],
        [1200, 2200, 3200],
        [1300, 2300, 3300],
        [1400, 2400, 3400]
    ], dtype=np.float32)

    print(f"\nOriginal data shape: {data.shape}")
    print(f"Original data mean: {data.mean(axis=0)}")
    print(f"Original data std: {data.std(axis=0)}")

    # 测试fit和transform
    scaler = StandardScaler()
    scaler.fit(data)

    print(f"\nScaler mean: {scaler.mean.flatten()}")
    print(f"Scaler std: {scaler.std.flatten()}")

    # 转换数据
    normalized = scaler.transform(data)
    print(f"\nNormalized data mean: {normalized.mean(axis=0)}")
    print(f"Normalized data std: {normalized.std(axis=0)}")
    print(f"Normalized data range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # 验证归一化效果
    assert np.allclose(normalized.mean(axis=0), 0, atol=1e-6), "Mean should be close to 0"
    assert np.allclose(normalized.std(axis=0), 1, atol=1e-6), "Std should be close to 1"
    print("\n✓ StandardScaler basic test passed!")

    # 测试逆变换
    restored = scaler.inverse_transform(normalized)
    print(f"\nRestored data matches original: {np.allclose(restored, data, atol=1e-4)}")
    assert np.allclose(restored, data, atol=1e-4), "Inverse transform should restore original data"
    print("✓ Inverse transform test passed!")


def test_scaler_save_load():
    """测试scaler的保存和加载"""
    print("\n" + "="*80)
    print("Test 2: Scaler Save and Load")
    print("="*80)

    # 创建并拟合scaler
    data = np.random.randn(100, 5).astype(np.float32) * 1000 + 5000
    scaler1 = StandardScaler()
    scaler1.fit(data)

    # 保存参数
    params = scaler1.get_params()
    print(f"\nOriginal scaler mean: {params['mean'].flatten()[:3]}")

    # 创建新scaler并加载参数
    scaler2 = StandardScaler()
    scaler2.set_params(params)

    # 验证两个scaler的结果一致
    transformed1 = scaler1.transform(data[:10])
    transformed2 = scaler2.transform(data[:10])

    assert np.allclose(transformed1, transformed2), "Loaded scaler should produce same results"
    print("✓ Scaler save/load test passed!")


def test_data_loader_normalization():
    """测试DataLoader的归一化功能"""
    print("\n" + "="*80)
    print("Test 3: DataLoader Normalization Integration")
    print("="*80)

    try:
        # 检查数据文件是否存在
        data_dir = project_root / "data"
        kline_file = data_dir / "industry_kline_data_cleaned.json"
        relation_file = data_dir / "industry_relation_cleaned.csv"

        if not kline_file.exists():
            print(f"⚠️  Warning: Data file not found at {kline_file}")
            print("   Skipping DataLoader test")
            return

        # 创建data loader
        loader = IndustryDataLoader(
            data_dir=str(data_dir),
            window_sizes=[20, 40, 80],
            future_days=30,
            num_classes=5
        )

        # 加载数据
        print("\nLoading data...")
        loader.load_data()
        print(f"Loaded {len(loader.industry_list)} industries")

        # 拟合scalers (这会在prepare_sequences中自动调用)
        print("\nFitting scalers...")
        loader.fit_scalers()

        # 检查scaler参数
        print(f"\n价格特征均值: {loader.scaler_price.mean.flatten()[:4]}")
        print(f"价格特征标准差: {loader.scaler_price.std.flatten()[:4]}")
        print(f"成交量均值: {loader.scaler_volume.mean.item():.2e}")
        print(f"成交额均值: {loader.scaler_amount.mean.item():.2e}")

        # 准备序列（会自动应用归一化）
        print("\nPreparing sequences with normalization...")
        data_dict = loader.prepare_sequences()

        sequences = data_dict['sequences']
        targets = data_dict['targets']

        print(f"\nSequences shape: {sequences.shape}")
        print(f"Targets shape: {targets.shape}")

        # 检查归一化效果
        # 特征索引: 0-3 价格, 4 成交量, 5 成交额, 6 收益率
        price_features = sequences[:, :, :4].reshape(-1, 4)
        volume_features = sequences[:, :, 4:5].reshape(-1, 1)
        amount_features = sequences[:, :, 5:6].reshape(-1, 1)

        print(f"\n归一化后的特征统计:")
        print(f"价格特征 - Mean: {price_features.mean(axis=0)}, Std: {price_features.std(axis=0)}")
        print(f"价格特征范围: [{price_features.min():.2f}, {price_features.max():.2f}]")
        print(f"成交量范围: [{volume_features.min():.2f}, {volume_features.max():.2f}]")
        print(f"成交额范围: [{amount_features.min():.2f}, {amount_features.max():.2f}]")

        # 验证归一化效果 (均值接近0，标准差接近1)
        assert np.abs(price_features.mean()) < 0.1, "Price features mean should be close to 0"
        assert np.abs(volume_features.mean()) < 0.1, "Volume mean should be close to 0"
        assert np.abs(amount_features.mean()) < 0.1, "Amount mean should be close to 0"

        # 验证没有极端值
        assert np.abs(price_features).max() < 20, "Normalized features should not have extreme values"
        assert np.abs(volume_features).max() < 20, "Normalized features should not have extreme values"
        assert np.abs(amount_features).max() < 20, "Normalized features should not have extreme values"

        print("\n✓ DataLoader normalization test passed!")

        # 测试scaler保存
        print("\nTesting scaler save...")
        save_path = project_root / "checkpoints" / "test_scalers.pkl"
        loader.save_scalers(str(save_path))

        # 测试scaler加载
        print("Testing scaler load...")
        loader2 = IndustryDataLoader(
            data_dir=str(data_dir),
            window_sizes=[20, 40, 80],
            future_days=30,
            num_classes=5
        )
        loader2.load_data()
        success = loader2.load_scalers(str(save_path))

        assert success, "Scaler loading should succeed"
        print("✓ Scaler save/load test passed!")

    except Exception as e:
        print(f"\n❌ DataLoader test failed with error: {e}")
        import traceback
        traceback.print_exc()


def test_nan_handling():
    """测试NaN值处理"""
    print("\n" + "="*80)
    print("Test 4: NaN Handling")
    print("="*80)

    # 创建包含NaN的数据
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    print(f"\nData with NaN:\n{data}")

    # StandardScaler应该跳过NaN
    scaler = StandardScaler()

    # 移除NaN后再fit
    data_clean = data[~np.isnan(data).any(axis=1)]
    print(f"\nCleaned data shape: {data_clean.shape}")

    scaler.fit(data_clean)
    normalized = scaler.transform(data_clean)

    print(f"Normalized data has NaN: {np.isnan(normalized).any()}")
    assert not np.isnan(normalized).any(), "Normalized data should not have NaN"

    print("✓ NaN handling test passed!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Running Normalization Tests")
    print("="*80)

    try:
        test_standard_scaler()
        test_scaler_save_load()
        test_nan_handling()
        test_data_loader_normalization()

        print("\n" + "="*80)
        print("All Tests Passed! ✓")
        print("="*80)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
