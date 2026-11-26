"""Run with python -m research.defense_bench; no changes to native entry points."""

import argparse
import json
import math
from pathlib import Path


def source_manifest():
    return json.loads(Path(__file__).with_name("sources.json").read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--method", choices=list(source_manifest()["methods"]))
    parser.add_argument("--data", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--attack", choices=["badnet", "DBA", "neurotoxin", "pgd"], default="badnet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=20)
    parser.add_argument("--num-corrupt", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--poison-frac", type=float, default=0.3)
    parser.add_argument("--local-ep", type=int, default=2)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--client-lr", type=float, default=0.1)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--rlr-threshold", type=int, default=8)
    parser.add_argument("--flame-noise", type=float, default=0.001)
    parser.add_argument("--lockdown-density", type=float, default=0.25)
    parser.add_argument("--lockdown-anneal", type=float, default=0.0001)
    parser.add_argument("--lockdown-threshold", type=int)
    parser.add_argument("--lockdown-same-mask", type=int, choices=[0, 1], default=1)
    parser.add_argument("--prodigal-topk", type=float, default=0.6)
    parser.add_argument("--prodigal-threshold", type=int)
    parser.add_argument("--crfl-clip", type=float)
    parser.add_argument("--crfl-noise", type=float)
    parser.add_argument("--crfl-test-noise", type=float)
    parser.add_argument("--crfl-samples", type=int, default=100)
    args = parser.parse_args(argv)
    if args.list_methods:
        return args
    if args.method is None:
        parser.error("--method is required")
    record = source_manifest()["methods"][args.method]
    if record["status"].startswith("blocked"):
        parser.error(record["reason"])
    for name in ["rounds", "num_agents", "local_ep", "bs", "threads", "crfl_samples"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    for name in ["client_lr", "server_lr", "beta", "lr_decay"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    for name in ["poison_frac", "lockdown_density", "lockdown_anneal", "prodigal_topk"]:
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f"{name} must lie in [0, 1]")
    if args.lockdown_density == 0 or args.prodigal_topk == 0:
        parser.error("Density and Top-K ratio must be positive")
    if not math.isfinite(args.flame_noise) or args.flame_noise < 0:
        parser.error("FLAME noise must be finite and nonnegative")
    if not 0 <= args.num_corrupt < args.num_agents or args.seed < 0 or args.seed >= 2**32:
        parser.error("Invalid malicious-client count or seed")
    if args.max_batches < 0 or args.test_limit < 0:
        parser.error("Smoke limits must be nonnegative")
    if args.method in ["flame", "crfl", "prodigal"] and args.server_lr != 1:
        parser.error("This method follows the source equations with server_lr=1")
    if args.method == "rlr" and not 0 <= args.rlr_threshold <= args.num_agents:
        parser.error("RLR threshold exceeds the participant count")
    if args.method == "flame" and args.num_agents < 3:
        parser.error("FLAME requires at least 3 participants")
    if args.lockdown_threshold is None:
        args.lockdown_threshold = max(1, args.num_agents // 2)
    if args.prodigal_threshold is None:
        args.prodigal_threshold = args.num_agents // 2
    if not 1 <= args.lockdown_threshold <= args.num_agents:
        parser.error("Invalid Lockdown vote threshold")
    if not 0 <= args.prodigal_threshold < args.num_agents:
        parser.error("Invalid Prodigal vote threshold (strict > comparison)")
    if args.method == "lockdown" and args.attack not in ["badnet", "DBA"]:
        parser.error("Native sparse attack paths differ from dense attacks; this adapter validates Lockdown with BadNet/DBA only")
    if args.method == "crfl":
        for name in ["crfl_clip", "crfl_noise", "crfl_test_noise"]:
            value = getattr(args, name)
            if value is None or not math.isfinite(value) or value <= 0:
                parser.error(f"Explicit positive --{name.replace('_', '-')} required; CIFAR has no original CRFL schedule")
    if not args.dry_run and args.output is None:
        parser.error("--output must specify a NEW experiment directory")
    if args.output:
        args.output = args.output.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_methods:
        for method, entry in source_manifest()["methods"].items():
            print(f"{method:12} {entry['status']}")
    elif args.dry_run:
        print(json.dumps({"settings": vars(args), "source": source_manifest()["methods"][args.method]}, indent=2, default=str))
    else:
        from .runner import train
        train(args)


if __name__ == "__main__":
    main()
