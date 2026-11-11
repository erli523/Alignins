#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import sys
from datetime import datetime


def parse_ratios(s: str) -> list[int]:
    parts = s.replace(",", " ").split()
    vals = []
    for p in parts:
        if not p.strip():
            continue
        v = int(p)
        if v <= 0 or v > 100:
            raise argparse.ArgumentTypeError("ratio must be in (0,100]")
        vals.append(v)
    if not vals:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Minimal sweep: run federated.py multiple times with different malicious ratios."
    )
    parser.add_argument("--attack", required=True,
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin", "model_replacement"])
    parser.add_argument("--aggr", required=True,
                        choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", "foolsgold", "rfa", "alphaguard", "mars"])
    parser.add_argument("--data", default="cifar10",
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--num_agents", type=int, default=20)
    parser.add_argument("--ratios", type=parse_ratios, default=[5, 10, 15, 20, 25,30],
                        help="Percent list, e.g. '10,20,30,40,50'")
    parser.add_argument("--exp_prefix", default=None,
                        help="Prefix for exp_name_extra; default auto-generated")

    # Accept unknown args and pass them through directly (e.g., --non_iid --beta 0.5 --device cuda:0)
    args, unknown = parser.parse_known_args()

    # Robustly locate federated.py whether this script is in repo root or src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "src", "federated.py"),          # script at repo root
        os.path.join(script_dir, "federated.py"),                  # script inside src/
        os.path.join(os.path.dirname(script_dir), "src", "federated.py"),  # one dir up then src/
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
    prefix = args.exp_prefix or f"{args.data}_{args.attack}_{args.aggr}_{ts}"

    base = [sys.executable or "python", "-u", fed_py,
            "--data", args.data,
            "--attack", args.attack,
            "--aggr", args.aggr,
            "--num_agents", str(args.num_agents)]

    # Append passthrough flags (unknown to this script) verbatim
    if unknown:
        base += unknown

    print(f"[INFO] Base: {' '.join(base)}")
    for r in args.ratios:
        num_corrupt = max(1, math.floor(args.num_agents * (r / 100.0)))
        run_name = f"{prefix}_ratio{r}_nc{num_corrupt}"
        cmd = base + ["--num_corrupt", str(num_corrupt),
                      "--exp_name_extra", run_name]
        print(f"\n[RUN] ratio={r}% -> num_corrupt={num_corrupt}")
        print(f"[CMD] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\n[DONE]")


if __name__ == "__main__":
    main()


