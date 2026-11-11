#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def parse_betas(s: str) -> list[float]:
    parts = s.replace(",", " ").split()
    vals = []
    for p in parts:
        if not p.strip():
            continue
        v = float(p)
        if v <= 0:
            raise argparse.ArgumentTypeError("beta must be > 0")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError("at least one beta is required")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Sweep non-IID beta with client sampling (100 total, 20% per round, 20 malicious)."
    )
    parser.add_argument("--attack", required=True,
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin", "model_replacement"])
    parser.add_argument("--aggr", required=True,
                        choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", "foolsgold", "rfa", "alphaguard", "mars"])
    parser.add_argument("--data", default="cifar10",
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--betas", type=parse_betas, default=[ 0.7,0.9],
                        help="Non-IID Dirichlet betas, e.g. '0.3,0.5,0.7'")
    parser.add_argument("--exp_prefix", default=None,
                        help="Prefix for exp_name_extra; default auto-generated")

    # 接受未知参数并原样透传（如 --device cuda:0 等）
    args, unknown = parser.parse_known_args()

    # 定位 federated.py（兼容从仓库根或 src/ 下执行）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "src", "federated.py"),          # 脚本在仓库根
        os.path.join(script_dir, "federated.py"),                  # 脚本在 src/ 内
        os.path.join(os.path.dirname(script_dir), "src", "federated.py"),  # 上一级目录再进 src/
    ]
    fed_py = None
    for path in candidates:
        if os.path.isfile(path):
            fed_py = path
            break
    if fed_py is None:
        print("Error: cannot locate src/federated.py", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.exp_prefix or f"{args.data}_{args.attack}_{args.aggr}_noniid_{ts}"

    # 固定 cross-device 设定
    num_agents = 50
    agent_frac = 0.2   # 每轮 20 个参与
    num_corrupt = 10  # 恶意客户端固定 20 个（ID 0..19）

    base = [sys.executable or "python", "-u", fed_py,
            "--data", args.data,
            "--attack", args.attack,
            "--aggr", args.aggr,
            "--num_agents", str(num_agents),
            "--agent_frac", str(agent_frac),
            "--num_corrupt", str(num_corrupt),
            "--non_iid"]

    # 透传用户附加参数（如 --device cuda:0 等）
    if unknown:
        base += unknown

    print(f"[INFO] Base (fixed): {' '.join(base)}")
    for beta in args.betas:
        run_name = f"{prefix}_beta{beta}"
        cmd = base + ["--beta", str(beta),
                      "--exp_name_extra", run_name]
        print(f"\n[RUN] beta={beta}")
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\n[DONE]")


if __name__ == "__main__":
    main()


