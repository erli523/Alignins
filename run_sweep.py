#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
try:
    import yaml
except Exception:
    yaml = None


def positive_int(value: str) -> int:
    iv = int(value)
    if iv <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return iv


def comma_or_space_separated_ints(value: str) -> list[int]:
    raw = value.replace(",", " ").split()
    vals = []
    for r in raw:
        if not r.strip():
            continue
        iv = int(r)
        if iv <= 0 or iv > 100:
            raise argparse.ArgumentTypeError("ratios must be in (0,100]")
        vals.append(iv)
    if not vals:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Batch run federated training with varying malicious client ratios."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="YAML config file path. CLI args override config.")
    parser.add_argument("--attack", type=str, required=False,
                        choices=["badnet", "DBA", "neurotoxin", "pgd", "dynamic_trigger", "altermin", "model_replacement"],
                        help="Attack method (maps to --attack in src/federated.py)")
    parser.add_argument("--aggr", type=str, required=False,
                        choices=["avg", "alignins", "rlr", "mkrum", "mmetric", "lockdown", "foolsgold", "rfa", "alphaguard", "mars"],
                        help="Defense/Aggregation method (maps to --aggr)")
    parser.add_argument("--data", type=str, default=None,
                        choices=["mnist", "fmnist", "cifar10", "cifar100", "tinyimagenet"],
                        help="Dataset")
    parser.add_argument("--num_agents", type=positive_int, default=None,
                        help="Total number of clients (maps to --num_agents)")
    parser.add_argument("--rounds", type=positive_int, default=None,
                        help="Override rounds; if omitted, federated.py will pick dataset default")
    parser.add_argument("--local_ep", type=positive_int, default=None,
                        help="Override local epochs per round")
    parser.add_argument("--bs", type=positive_int, default=None,
                        help="Override batch size")
    parser.add_argument("--agent_frac", type=float, default=None,
                        help="Fraction of sampled clients each round")
    parser.add_argument("--poison_frac", type=float, default=None,
                        help="Poisoned data fraction for backdoor-type attacks")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target label for backdoor attacks")
    # Federated.py common flags (convenience; still can be passed via --extra)
    parser.add_argument("--non_iid", action="store_true",
                        help="Use non-iid partition (maps to --non_iid)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Dirichlet beta for non-iid (maps to --beta)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string like 'cuda:0' (maps to --device)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Prefix name for logging directory/files")
    parser.add_argument("--ratios", type=comma_or_space_separated_ints, default=None,
                        help="Malicious client ratios (percent), e.g. '10,20,30,40,50'")
    parser.add_argument("--extra", type=str, nargs=argparse.REMAINDER, default=[],
                        help="Any extra args passed through to src/federated.py (place after '--')")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    fed_py = os.path.join(project_root, "src", "federated.py")
    if not os.path.isfile(fed_py):
        print(f"Error: cannot find {fed_py}", file=sys.stderr)
        sys.exit(1)

    # Load config (if available)
    cfg: Dict[str, Any] = {}
    # Helper to resolve absolute path for config arg
    def resolve_path(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    # Load config(s)
    cfgs: List[Dict[str, Any]] = []
    cfg_sources: List[str] = []
    if args.config:
        cfg_path = resolve_path(args.config)
        if os.path.isdir(cfg_path):
            # Load all .yaml/.yml files in directory (sorted)
            entries = sorted(os.listdir(cfg_path))
            yaml_files = [e for e in entries if e.lower().endswith((".yaml", ".yml"))]
            if not yaml_files:
                print(f"Error: no .yaml/.yml files under directory: {cfg_path}", file=sys.stderr)
                sys.exit(1)
            if yaml is None:
                print("Error: PyYAML not available. Please install pyyaml.", file=sys.stderr)
                sys.exit(1)
            for yf in yaml_files:
                fp = os.path.join(cfg_path, yf)
                with open(fp, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                    if not isinstance(loaded, dict):
                        print(f"Error: config root must be a mapping: {fp}", file=sys.stderr)
                        sys.exit(1)
                    cfgs.append(loaded)
                    cfg_sources.append(fp)
        elif os.path.isfile(cfg_path):
            if yaml is None:
                print("Error: PyYAML not available. Please install pyyaml.", file=sys.stderr)
                sys.exit(1)
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    print("Error: config root must be a mapping.", file=sys.stderr)
                    sys.exit(1)
                cfgs.append(loaded)
                cfg_sources.append(cfg_path)
        else:
            # If user provided explicit config path and it does not exist, error out
            if args.config != "config.yaml":
                print(f"Error: cannot find config path: {cfg_path}", file=sys.stderr)
                sys.exit(1)

    # If no config supplied or found, use empty default
    if not cfgs:
        cfgs = [{}]
        cfg_sources = ["<defaults>"]

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Iterate configs (single or batch)
    for cfg, cfg_src in zip(cfgs, cfg_sources):
        def cfg_get(key: str, default: Any = None) -> Any:
            return cfg.get(key, default)

        # Resolve values: CLI overrides config; config overrides None defaults
        attack = args.attack or cfg_get("attack")
        aggr = args.aggr or cfg_get("aggr")
        data = args.data or cfg_get("data", "cifar10")
        num_agents = args.num_agents or cfg_get("num_agents", 20)
        rounds = args.rounds or cfg_get("rounds")
        local_ep = args.local_ep or cfg_get("local_ep")
        bs = args.bs or cfg_get("bs")
        agent_frac = args.agent_frac if args.agent_frac is not None else cfg_get("agent_frac")
        poison_frac = args.poison_frac if args.poison_frac is not None else cfg_get("poison_frac")
        target_class = args.target_class if args.target_class is not None else cfg_get("target_class")
        ratios = args.ratios or cfg_get("ratios", [5, 10, 15, 20, 25,30])
        # federated alignment extras
        cfg_non_iid = bool(cfg_get("non_iid", False))
        non_iid = args.non_iid or cfg_non_iid
        beta = args.beta if args.beta is not None else cfg_get("beta")
        device = args.device if args.device is not None else cfg_get("device")
        extra_from_cfg = cfg_get("extra", [])
        extra: List[str] = []
        if isinstance(extra_from_cfg, list):
            extra = [str(x) for x in extra_from_cfg]
        elif isinstance(extra_from_cfg, str):
            extra = extra_from_cfg.split()
        exp_prefix = args.exp_name or cfg_get("exp_name") or f"{data}_{attack}_{aggr}_{time_tag}"

        if not attack or not aggr:
            print(f"Error: 'attack' and 'aggr' must be specified (via config or CLI). Source={cfg_src}", file=sys.stderr)
            sys.exit(1)

        # Build constant base cmd
        python_exec = sys.executable or "python"
        base_cmd = [python_exec, "-u", fed_py,
                    "--data", data,
                    "--attack", attack,
                    "--aggr", aggr,
                    "--num_agents", str(num_agents)]

        # Optional overrides
        if rounds is not None:
            base_cmd += ["--rounds", str(rounds)]
        if local_ep is not None:
            base_cmd += ["--local_ep", str(local_ep)]
        if bs is not None:
            base_cmd += ["--bs", str(bs)]
        if agent_frac is not None:
            base_cmd += ["--agent_frac", str(agent_frac)]
        if poison_frac is not None:
            base_cmd += ["--poison_frac", str(poison_frac)]
        if target_class is not None:
            base_cmd += ["--target_class", str(target_class)]
        if non_iid:
            base_cmd += ["--non_iid"]
        if beta is not None:
            base_cmd += ["--beta", str(beta)]
        if device is not None:
            base_cmd += ["--device", str(device)]

        # Extra passthrough arguments
        passthrough: List[str] = []
        if extra:
            passthrough += extra
        if args.extra:
            passthrough += args.extra
        if passthrough:
            base_cmd += passthrough

        print(f"\n[CONFIG] {cfg_src}")
        print(f"[INFO] Base command: {' '.join(base_cmd)}")
        print(f"[INFO] Experiment prefix: {exp_prefix}")

        for ratio in ratios:
            # Compute number of corrupt clients from percentage
            num_corrupt = max(1, math.floor(num_agents * (ratio / 100.0)))
            run_name = f"{exp_prefix}_ratio{ratio}_nc{num_corrupt}"

            cmd = base_cmd + ["--num_corrupt", str(num_corrupt),
                              "--exp_name_extra", run_name]

            print(f"\n[RUN] ratio={ratio}% -> num_corrupt={num_corrupt}")
            print(f"[CMD] {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Run failed for ratio={ratio}% (num_corrupt={num_corrupt}).", file=sys.stderr)
                print(f"[ERROR] Return code: {e.returncode}", file=sys.stderr)
                sys.exit(e.returncode)

    print("\n[DONE] All runs finished successfully.")


if __name__ == "__main__":
    main()


