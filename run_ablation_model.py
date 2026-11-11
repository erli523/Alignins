#!/usr/bin/env python3
"""
AlphaGuard Model Architecture Ablation Study
测试不同模型结构对 AlphaGuard 防御效果的影响

实验配置:
1. ResNet9 (6.6M) - 轻量级残差网络（默认）
2. VGG9 (11.6M) - 经典卷积网络，无残差连接
3. ResNet18 (11.2M) - 标准 ResNet，更深的网络
4. MobileNetV2 (2.3M) - 极轻量级，深度可分离卷积
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="AlphaGuard 模型结构消融实验"
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
    parser.add_argument("--models", type=str, default="resnet9,vgg9,resnet18,mobilenetv2",
                        help="逗号分隔的模型列表，例如 'resnet9,vgg9,resnet18,mobilenetv2'")
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
    
    # 解析模型列表
    model_list = [x.strip() for x in args.models.split(",")]
    
    # 模型信息字典
    model_info = {
        "resnet9": ("ResNet9", "6.6M", "轻量级残差网络（默认）"),
        "vgg9": ("VGG9", "11.6M", "经典卷积网络，无残差连接"),
        "resnet18": ("ResNet18", "11.2M", "标准ResNet，更深的网络"),
        "mobilenetv2": ("MobileNetV2", "2.3M", "极轻量级，深度可分离卷积"),
    }
    
    # 验证模型名称
    valid_models = set(model_info.keys())
    for model_name in model_list:
        if model_name not in valid_models:
            print(f"错误: 未知的模型 '{model_name}'", file=sys.stderr)
            print(f"可用模型: {', '.join(valid_models)}", file=sys.stderr)
            sys.exit(1)
    
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
    prefix = args.exp_prefix or f"alphaguard_model_{args.data}_{args.attack}_{iid_str}_{ts}"
    
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
    print("AlphaGuard Model Architecture Ablation Study")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Attack: {args.attack}")
    print(f"Defense: alphaguard")
    print(f"Num Agents: {args.num_agents}")
    print(f"Num Corrupt: {args.num_corrupt}")
    print(f"Rounds: {args.rounds}")
    print(f"Data Distribution: {iid_str}")
    print(f"\nModel Architectures to Test:")
    for model_name in model_list:
        name, params, desc = model_info[model_name]
        print(f"  - {name:15s} ({params:6s}): {desc}")
    if unknown:
        print(f"\nExtra args: {' '.join(unknown)}")
    print("="*70)
    
    for idx, model_name in enumerate(model_list, 1):
        name, params, desc = model_info[model_name]
        run_name = f"model_{model_name}"
        cmd = base + [
            "--model", model_name,
            "--exp_name_extra", run_name
        ]
        
        print(f"\n[{idx}/{len(model_list)}] Running: {name} ({params})")
        print(f"  Description: {desc}")
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
    print("[DONE] All model architecture experiments completed!")
    print("="*70)
    print("\n实验结果分析建议:")
    print("  1. 比较不同模型的 Clean ACC 和 ASR")
    print("  2. 分析参数量对防御效果的影响")
    print("  3. 观察残差连接 (ResNet) vs 无残差 (VGG) 的差异")
    print("  4. 评估轻量级模型 (MobileNetV2) 的防御能力")
    print("  5. 查看日志中 Stage 3 的 Top-K 检测效果")


if __name__ == "__main__":
    main()

