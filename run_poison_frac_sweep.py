#!/usr/bin/env python3
"""
测试不同 poisoning data fraction 下的攻击防御效果
遍历 poison_frac: 0.1, 0.3, 0.5, 0.7
其他参数使用 federated.py 的默认设置
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="测试不同 poisoning data fraction 下的攻击防御效果"
    )
    
    # 必需参数：手动指定攻击和防御方法
    parser.add_argument("--attack", required=True,
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", 
                                "altermin", "model_replacement"],
                        help="攻击方法")
    parser.add_argument("--aggr", required=True,
                        choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", 
                                "foolsgold", "rfa", "alphaguard", "mars"],
                        help="防御方法")
    
    # 可选参数
    parser.add_argument("--data", default="cifar10",
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--num_corrupt", type=int, default=4)
    parser.add_argument("--poison_fracs", type=str, default="0.1,0.3,0.5,0.7",
                        help="逗号分隔的 poison_frac 值，例如 '0.1,0.3,0.5,0.7'")
    parser.add_argument("--exp_prefix", default=None,
                        help="实验名称前缀，默认自动生成")
    
    # 接受其他未知参数并直接传递给 federated.py (例如: --rounds 100 --device cuda:0 --non_iid)
    args, unknown = parser.parse_known_args()
    
    # 解析 poison_frac 列表
    poison_fracs = [float(x.strip()) for x in args.poison_fracs.split(",")]
    
    # 验证 poison_frac 范围
    for pf in poison_fracs:
        if pf <= 0 or pf > 1:
            print(f"错误: poison_frac 必须在 (0, 1] 范围内，但得到 {pf}", file=sys.stderr)
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
    prefix = args.exp_prefix or f"{args.data}_{args.attack}_{args.aggr}_{ts}"
    
    # 构建基础命令
    base = [
        sys.executable or "python", "-u", fed_py,
        "--data", args.data,
        "--attack", args.attack,
        "--aggr", args.aggr,
        "--num_agents", str(args.num_agents),
        "--num_corrupt", str(args.num_corrupt)
    ]
    
    # 添加透传的未知参数
    if unknown:
        base += unknown
    
    print(f"[INFO] Base: {' '.join(base)}")
    if unknown:
        print(f"[INFO] Extra args: {' '.join(unknown)}")
    print(f"[INFO] Testing poison_frac values: {poison_fracs}")
    
    for pf in poison_fracs:
        run_name = f"{prefix}_pf{pf}"
        cmd = base + [
            "--poison_frac", str(pf),
            "--exp_name_extra", run_name
        ]
        print(f"\n[RUN] poison_frac={pf}")
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()

