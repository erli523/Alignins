"""
调试 Model Replacement 攻击的振荡问题
打印详细的中间信息
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from torch.nn.utils import parameters_to_vector

# 模拟聚合过程
def simulate_aggregation():
    """
    模拟 Model Replacement 的聚合过程
    """
    print("=" * 60)
    print("Model Replacement 聚合过程模拟")
    print("=" * 60)
    
    # 参数设置
    num_malicious = 4
    num_benign = 16
    data_per_client = 2500
    mal_scaling = 5.0  # 恶意更新放大倍数
    server_lr = 1.0
    
    # 模拟更新向量（随机）
    update_dim = 100  # 简化，实际是几百万维
    
    mal_update_norm = 0.1  # 恶意更新的范数
    ben_update_norm = 0.15  # 良性更新的范数
    
    print(f"\n配置:")
    print(f"  恶意客户端: {num_malicious}, 数据量: {data_per_client}")
    print(f"  良性客户端: {num_benign}, 数据量: {data_per_client}")
    print(f"  恶意放大倍数: {mal_scaling}")
    print(f"  Server LR: {server_lr}")
    
    # 模拟聚合
    total_data = (num_malicious + num_benign) * data_per_client
    
    mal_contribution = num_malicious * data_per_client * mal_scaling * mal_update_norm
    ben_contribution = num_benign * data_per_client * 1.0 * ben_update_norm
    
    mal_weight = (num_malicious * data_per_client * mal_scaling) / (num_malicious * data_per_client * mal_scaling + num_benign * data_per_client)
    ben_weight = (num_benign * data_per_client) / (num_malicious * data_per_client * mal_scaling + num_benign * data_per_client)
    
    print(f"\n聚合权重:")
    print(f"  恶意贡献权重: {mal_weight:.4f} ({mal_weight*100:.2f}%)")
    print(f"  良性贡献权重: {ben_weight:.4f} ({ben_weight*100:.2f}%)")
    print(f"  权重和: {mal_weight + ben_weight:.4f}")
    
    # 计算聚合后的更新范数
    aggregated_norm = mal_weight * mal_update_norm + ben_weight * ben_update_norm
    
    print(f"\n更新范数:")
    print(f"  恶意更新范数: {mal_update_norm}")
    print(f"  良性更新范数: {ben_update_norm}")
    print(f"  聚合后范数: {aggregated_norm:.4f}")
    
    print(f"\n结论:")
    if mal_weight > 0.5:
        print(f"  ✅ 恶意更新主导聚合 ({mal_weight*100:.1f}% > 50%)")
        print(f"  → 模型会被严重污染")
        print(f"  → ASR 应该很高 (~90%+)")
    else:
        print(f"  ⚠️ 良性更新主导聚合 ({ben_weight*100:.1f}% > 50%)")
        print(f"  → 模型应该相对正常")
        print(f"  → ASR 应该较低 (~30%)")

if __name__ == "__main__":
    simulate_aggregation()
    
    print("\n" + "=" * 60)
    print("测试不同放大倍数的影响")
    print("=" * 60)
    
    for scaling in [1.0, 2.0, 3.0, 5.0, 10.0]:
        mal_total = 4 * 2500 * scaling
        ben_total = 16 * 2500 * 1.0
        mal_weight = mal_total / (mal_total + ben_total)
        print(f"\n放大 {scaling:.1f} 倍: 恶意权重 {mal_weight:.2%}")
        if mal_weight > 0.6:
            print(f"  → 强主导，可能导致振荡")
        elif mal_weight > 0.5:
            print(f"  → 主导，ASR 应该高")
        else:
            print(f"  → 不主导，ASR 应该低")

