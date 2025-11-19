"""
验证脚本 - 检查所有改进是否正确集成
"""
import torch
import json
from pathlib import Path

def verify_setup():
    """验证项目设置"""
    print("=" * 60)
    print("MMF-GAT Setup Verification")
    print("=" * 60)

    # 1. 检查配置文件
    print("\n1. Checking configuration files...")
    config_path = Path("config/default_config.yaml")
    if config_path.exists():
        print("   ✓ Configuration file found")
        with open(config_path, 'r') as f:
            content = f.read()
            if 'use_industry_embedding' in content:
                print("   ✓ Industry embedding config present")
            else:
                print("   ✗ Industry embedding config missing")
    else:
        print("   ✗ Configuration file not found")

    # 2. 检查数据文件
    print("\n2. Checking data files...")
    data_files = [
        "data/industry_kline_data_cleaned.json",
        "data/industry_relation_cleaned.csv",
        "data/industry_list.json"
    ]
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} not found")

    # 3. 检查行业数量
    print("\n3. Checking industry count...")
    try:
        with open("data/industry_list.json", 'r', encoding='utf-8') as f:
            industries = json.load(f)
            num_industries = len(industries)
            print(f"   ✓ Found {num_industries} industries")
            if num_industries == 86:
                print("   ✓ Industry count matches config (86)")
            else:
                print(f"   ⚠ Industry count mismatch: {num_industries} vs expected 86")
    except Exception as e:
        print(f"   ✗ Error loading industry list: {e}")

    # 4. 检查模型组件
    print("\n4. Checking model components...")
    try:
        from components.model import IndustryStockModel
        from components.visualizer import Visualizer
        from components.config_loader import load_config_with_cli
        print("   ✓ All components imported successfully")

        # 创建测试模型
        model = IndustryStockModel(
            num_industries=86,
            use_industry_embedding=True,
            embedding_fusion_alpha=1.0
        )

        # 检查嵌入层
        if hasattr(model, 'industry_embeddings') and model.industry_embeddings is not None:
            print(f"   ✓ Industry embeddings initialized: {model.industry_embeddings.weight.shape}")
        else:
            print("   ✗ Industry embeddings not found")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        embedding_params = model.industry_embeddings.weight.numel()
        print(f"   ✓ Total model parameters: {total_params:,}")
        print(f"   ✓ Embedding parameters: {embedding_params:,} ({embedding_params/total_params*100:.2f}%)")

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")

    # 5. 检查可视化方法
    print("\n5. Checking visualization methods...")
    try:
        vis = Visualizer(save_dir="./visualizations")
        required_methods = [
            'plot_subgraph_structure',
            'plot_embedding_similarity',
            'plot_subgraph_attention_summary'
        ]
        for method_name in required_methods:
            if hasattr(vis, method_name):
                print(f"   ✓ {method_name}")
            else:
                print(f"   ✗ {method_name} not found")
    except Exception as e:
        print(f"   ✗ Error checking visualizer: {e}")

    # 6. 检查依赖包
    print("\n6. Checking dependencies...")
    try:
        import networkx as nx
        print(f"   ✓ networkx {nx.__version__}")
    except ImportError:
        print("   ✗ networkx not installed (required for subgraph visualization)")

    required_packages = ['torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'pywt']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"   ✓ {pkg}")
        except ImportError:
            print(f"   ✗ {pkg} not installed")

    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    print("\nTo start training with industry embeddings:")
    print("  python train.py --use_industry_embedding")
    print("\nTo adjust fusion weight:")
    print("  python train.py --embedding_fusion_alpha 0.7")
    print("\nTo disable embeddings (baseline):")
    print("  python train.py --use_industry_embedding False")
    print("=" * 60)

if __name__ == "__main__":
    verify_setup()
