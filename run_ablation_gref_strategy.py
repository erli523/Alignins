#!/usr/bin/env python3
"""
AlphaGuard g_ref Strategy Ablation Study
测试不同参考向量计算策略对防御效果的影响

实验配置:
1. median: 中位数（当前默认）- 对异常值鲁棒
2. geometric_median: 几何中位数 - 高维空间的鲁棒估计
3. krum_select: Krum选择 - 选择距离其他更新最近的单个更新
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="AlphaGuard g_ref 参考向量计算策略消融实验"
    )
    
    # 基础参数
    parser.add_argument("--attack", default="badnet",
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", 
                                "altermin", "model_replacement"],
                        help="攻击方法 (默认: badnet)")
    parser.add_argument("--data", default="cifar10",
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--num_corrupt", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--non_iid", action="store_true", default=True,
                        help="使用 Non-IID 数据分布")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Non-IID Dirichlet 分布参数")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU 设备 (默认: cuda:0)")
    parser.add_argument("--exp_prefix", default=None,
                        help="实验名称前缀，默认自动生成")
    
    # 接受其他未知参数并直接传递给 federated.py
    args, unknown = parser.parse_known_args()
    
    # g_ref 策略配置
    gref_strategies = [
        # (strategy_name, 描述)
        ("median", "中位数（默认）- 对异常值鲁棒，计算高效"),
        ("geometric_median", "几何中位数 - 高维空间的鲁棒统计量"),
        ("krum_select", "Krum选择 - 选择与其他更新距离最小的单个更新"),
    ]
    
    # 定位 federated.py 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "src", "federated.py"),
        os.path.join(script_dir, "federated.py"),
        os.path.join(os.path.dirname(script_dir), "src", "federated.py"),
    ]
    fed_py = None
    for path in candidates:
        if os.path.isfile(path):
            fed_py = path
            break
    if fed_py is None:
        print("错误: 无法找到 src/federated.py", file=sys.stderr)
        sys.exit(1)
    
    # 生成实验名称前缀
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    iid_str = f"noniid_{args.beta}" if args.non_iid else "iid"
    prefix = args.exp_prefix or f"alphaguard_gref_{args.data}_{args.attack}_{iid_str}_{ts}"
    
    # 构建基础命令
    base = [
        sys.executable or "python", "-u", fed_py,
        "--data", args.data,
        "--attack", args.attack,
        "--aggr", "alphaguard",
        "--num_agents", str(args.num_agents),
        "--num_corrupt", str(args.num_corrupt),
        "--rounds", str(args.rounds),
        "--device", args.device
    ]
    
    # 添加 Non-IID 参数
    if args.non_iid:
        base += ["--non_iid", "--beta", str(args.beta)]
    
    # 添加透传的未知参数
    if unknown:
        base += unknown
    
    print("="*70)
    print("AlphaGuard g_ref Strategy Ablation Study")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Attack: {args.attack}")
    print(f"Defense: alphaguard")
    print(f"Num Agents: {args.num_agents}")
    print(f"Num Corrupt: {args.num_corrupt}")
    print(f"Rounds: {args.rounds}")
    print(f"Data Distribution: {iid_str}")
    print(f"\ng_ref Computation Strategies:")
    for strategy, desc in gref_strategies:
        print(f"  - {strategy:20s}: {desc}")
    if unknown:
        print(f"\nExtra args: {' '.join(unknown)}")
    print("="*70)
    
    for idx, (strategy, desc) in enumerate(gref_strategies, 1):
        run_name = f"gref_{strategy}"
        cmd = base + [
            "--ag_gref_strategy", strategy,
            "--exp_name_extra", run_name
        ]
        
        print(f"\n[{idx}/{len(gref_strategies)}] Running: {strategy}")
        print(f"  Description: {desc}")
        print(f"[CMD] {' '.join(cmd)}")
        print("-"*70)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] {strategy} completed")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {strategy} failed with exit code {e.returncode}", file=sys.stderr)
            # 可以选择继续或停止
            # sys.exit(1)  # 如果想在第一次失败时停止，取消注释这一行
    
    print("\n" + "="*70)
    print("[DONE] All g_ref strategy experiments completed!")
    print("="*70)
    print("\n实验结果分析建议:")
    print("  1. 比较三种策略的 Clean ACC 和 ASR")
    print("  2. 查看日志中 'Reference norm' (g_ref 的 L2 范数)")
    print("  3. 观察不同策略下 Stage 3 的检测率")
    print("  4. 分析 geometric_median 是否比 median 更鲁棒")
    print("  5. 检查 krum_select 的稳定性（可能受单个更新影响）")


if __name__ == "__main__":
    main()

