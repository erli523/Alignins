#!/usr/bin/env python3
"""
AlphaGuard Top-K Ablation Study
测试不同 Top-K 比例对 AlphaGuard 防御效果的影响
遍历 ag_topk: 0.0001, 0.001, 0.01, 0.1, 1.0
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="AlphaGuard Top-K 参数消融实验"
    )
    
    # 必需参数
    parser.add_argument("--attack", default="badnet",
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", 
                                "altermin", "model_replacement"],
                        help="攻击方法 (默认: badnet)")
    
    # 可选参数
    parser.add_argument("--data", default="cifar10",
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--num_corrupt", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--topk_values", type=str, default="0.0001,0.001,0.01,0.1,1.0",
                        help="逗号分隔的 Top-K 值，例如 '0.0001,0.001,0.01,0.1,1.0'")
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
    
    # 解析 Top-K 值列表
    topk_values = [float(x.strip()) for x in args.topk_values.split(",")]
    
    # 验证 Top-K 范围
    for tk in topk_values:
        if tk <= 0 or tk > 1:
            print(f"错误: Top-K 必须在 (0, 1] 范围内，但得到 {tk}", file=sys.stderr)
            sys.exit(1)
    
    # 定位 federated.py 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "src", "federated.py"),          # 脚本在仓库根目录
        os.path.join(script_dir, "federated.py"),                  # 脚本在 src/ 目录
        os.path.join(os.path.dirname(script_dir), "src", "federated.py"),  # 上一级目录
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
    prefix = args.exp_prefix or f"alphaguard_topk_{args.data}_{args.attack}_{iid_str}_{ts}"
    
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
    print("AlphaGuard Top-K Ablation Study")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Attack: {args.attack}")
    print(f"Defense: alphaguard")
    print(f"Num Agents: {args.num_agents}")
    print(f"Num Corrupt: {args.num_corrupt}")
    print(f"Rounds: {args.rounds}")
    print(f"Data Distribution: {iid_str}")
    print(f"Testing Top-K values: {topk_values}")
    if unknown:
        print(f"Extra args: {' '.join(unknown)}")
    print("="*70)
    
    for idx, tk in enumerate(topk_values, 1):
        run_name = f"topk_{tk}"
        cmd = base + [
            "--ag_topk", str(tk),
            "--exp_name_extra", run_name
        ]
        print(f"\n[{idx}/{len(topk_values)}] Running with Top-K={tk}")
        print(f"[CMD] {' '.join(cmd)}")
        print("-"*70)
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Top-K={tk} completed")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Top-K={tk} failed with exit code {e.returncode}", file=sys.stderr)
            # 可以选择继续或停止
            # sys.exit(1)  # 如果想在第一次失败时停止，取消注释这一行
    
    print("\n" + "="*70)
    print("[DONE] All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    main()

