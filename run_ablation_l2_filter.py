#!/usr/bin/env python3
"""
AlphaGuard L2 Filter Ablation Study
测试不同 L2 范数过滤边界对 Stage 1 参考向量质量的影响

实验配置:
1. 无过滤 (0-100): 保留所有客户端
2. 宽松过滤 (10-90): 保留中间 80% 的客户端
3. 极度严格过滤 (40-60): 保留中间 20% 的客户端
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="AlphaGuard L2 范数过滤边界消融实验"
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
    
    # L2 过滤配置
    l2_configs = [
        # (lower, upper, 名称, 描述)
        (0, 100, "no_filter", "无过滤 - 保留所有客户端"),
        (10, 90, "loose", "宽松过滤 - 保留中间80%客户端"),
        (40, 60, "very_strict", "极度严格过滤 - 保留中间20%客户端"),
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
    prefix = args.exp_prefix or f"alphaguard_l2filter_{args.data}_{args.attack}_{iid_str}_{ts}"
    
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
    print("AlphaGuard L2 Filter Ablation Study")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Attack: {args.attack}")
    print(f"Defense: alphaguard")
    print(f"Num Agents: {args.num_agents}")
    print(f"Num Corrupt: {args.num_corrupt}")
    print(f"Rounds: {args.rounds}")
    print(f"Data Distribution: {iid_str}")
    print(f"\nL2 Filter Configurations:")
    for lower, upper, name, desc in l2_configs:
        print(f"  - [{lower:3d}, {upper:3d}] ({name:15s}): {desc}")
    if unknown:
        print(f"\nExtra args: {' '.join(unknown)}")
    print("="*70)
    
    for idx, (lower, upper, name, desc) in enumerate(l2_configs, 1):
        run_name = f"l2_{name}"
        cmd = base + [
            "--ag_l2_lower", str(lower),
            "--ag_l2_upper", str(upper),
            "--exp_name_extra", run_name
        ]
        
        print(f"\n[{idx}/{len(l2_configs)}] Running: {desc}")
        print(f"  L2 Filter Range: [{lower}, {upper}]")
        print(f"[CMD] {' '.join(cmd)}")
        print("-"*70)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] {name} completed")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {name} failed with exit code {e.returncode}", file=sys.stderr)
            # 可以选择继续或停止
            # sys.exit(1)  # 如果想在第一次失败时停止，取消注释这一行
    
    print("\n" + "="*70)
    print("[DONE] All L2 filter experiments completed!")
    print("="*70)
    print("\n实验结果分析建议:")
    print("  1. 比较三种配置的 Clean ACC 和 ASR")
    print("  2. 查看日志中 Stage 1 的 'Trusted clients' 数量")
    print("  3. 观察 'Reference norm' (g_ref 的 L2 范数)")
    print("  4. 分析 Stage 3 中恶意客户端的检测率")


if __name__ == "__main__":
    main()

