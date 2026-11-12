#!/usr/bin/env python3
"""B 组完整实现：配对正常训练、差集定位、坐标替换、全客户端 FedAvg。

运行：python src/ramp_b.py --device cuda:0
修改 RAMPB.build_reference / select_coordinates / repair_update / aggregate
即可测试方案。本文件不调用 research_oracle 或 research/oracle_*。
使用真实恶意身份和配对正常更新，属于 Oracle 机制实验。
"""
import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Subset

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import models
import utils
from agent import Agent


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_vector(model):
    return parameters_to_vector(list(model.state_dict().values())).detach()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class RAMPB:
    """完整方案均在此类中；不调用原生 AlphaGuard 淘汰或外部研究脚本。"""

    def __init__(self, args, model):
        self.args = args
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.layout, offset = [], 0
        for name, value in model.state_dict().items():
            if name not in trainable:
                raise ValueError("This study requires the buffer-free ResNet9")
            self.layout.append(
                dict(name=name, start=offset, numel=value.numel(), shape=list(value.shape))
            )
            offset += value.numel()
        self.dimension = offset
        self.k = max(1, int(offset * args.topk_ratio))

    def paired_train(self, agent, model, criterion, round_number):
        """同一初始模型/RNG，训练投毒和正常分支；随后恢复投毒分支的状态。"""
        before_rng, before_model = rng_state(), model_state(model)
        attack = agent.local_train(model, criterion, round_number, neurotoxin_mask={})
        after_rng, after_model, after_mode = rng_state(), model_state(model), model.training
        clean_agent = copy.copy(agent)
        clean_agent.args = copy.copy(agent.args)
        clean_agent.args.num_corrupt = 0
        clean_agent.is_malicious = 0
        clean_agent.train_dataset = utils.DatasetSplit(agent.clean_backup_dataset, agent.data_idxs)
        clean_agent.train_loader = DataLoader(
            clean_agent.train_dataset,
            batch_size=self.args.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        try:
            model.load_state_dict(before_model)
            restore_rng(before_rng)
            clean = (
                clean_agent.local_train(model, criterion, round_number, neurotoxin_mask={})
                .detach()
                .cpu()
                .clone()
            )
        finally:
            model.load_state_dict(after_model)
            model.train(after_mode)
            restore_rng(after_rng)
        return attack, clean

    @torch.no_grad()
    def build_reference(self, updates):
        """修改点 1：参数更新 L2 中间区间选参考候选，坐标 median 或 mean。"""
        ids = list(updates)
        matrix = torch.stack([updates[c].detach() for c in ids])
        if matrix.shape[1] != self.dimension or not bool(torch.isfinite(matrix).all()):
            raise ValueError("Invalid update dimension or non-finite values")
        norms = matrix.norm(dim=1)
        lower = torch.quantile(norms, self.args.l2_lower / 100)
        upper = torch.quantile(norms, self.args.l2_upper / 100)
        candidate = torch.where((norms >= lower) & (norms <= upper))[0]
        fallback = candidate.numel() < min(self.args.min_reference, len(ids))
        if fallback:
            candidate = torch.arange(len(ids), device=matrix.device)
        values = matrix[candidate]
        ref = values.median(dim=0).values if self.args.reference == "median" else values.mean(dim=0)
        info = dict(
            reference_clients=[ids[i] for i in candidate.cpu().tolist()],
            fallback=fallback,
            l2_low=float(lower),
            l2_high=float(upper),
            reference_abs_below_epsilon_fraction=float(
                (ref.abs() < self.args.epsilon).float().mean()
            ),
        )
        return ref, norms, info

    @torch.no_grad()
    def select_coordinates(self, update, reference):
        """修改点 2：原始/相对 Top-K；差集 = relative Top-K 减 raw Top-K。"""
        absolute = update.abs()
        relative = absolute / (reference.abs() + self.args.epsilon)
        raw = np.sort(torch.topk(absolute, self.k, sorted=False).indices.cpu().numpy())
        top = torch.topk(relative, self.k, sorted=False)
        rel = np.sort(top.indices.cpu().numpy())
        return dict(
            raw=raw,
            relative=rel,
            promoted=np.setdiff1d(rel, raw, assume_unique=True),
            score=float(top.values.mean()),
        )

    @torch.no_grad()
    def repair_update(self, attack_update, clean_update, selected_indices):
        """修改点 3：选中坐标如何修复；当前直接替换成配对正常更新。"""
        result = attack_update.clone()
        index = torch.as_tensor(selected_indices, dtype=torch.long, device=attack_update.device)
        clean = clean_update.to(device=attack_update.device, dtype=attack_update.dtype)
        result[index] = clean[index]
        return result

    @torch.no_grad()
    def aggregate(self, model, updates, clean_updates, data_sizes, round_number):
        """修改点 4：Oracle 指定恶意客户端修复；全部客户端按数据量加权。"""
        ids = list(updates)
        malicious = [c for c in ids if c < self.args.num_corrupt]
        if set(clean_updates) != set(malicious):
            raise ValueError("Missing or stale paired normal updates")
        ref, norms, reference = self.build_reference(updates)
        features = {c: self.select_coordinates(updates[c], ref) for c in ids}
        scores = np.array([features[c]["score"] for c in ids])
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        total = sum(data_sizes[c] for c in ids)
        applied, rows, layers, selected_total = {}, [], [], 0
        for position, cid in enumerate(ids):
            feature = features[cid]
            z = abs(feature["score"] - median) / max(1.4826 * mad, 1e-12)
            row = dict(
                round=round_number,
                client_id=cid,
                role="malicious" if cid in malicious else "benign",
                reference_candidate=cid in reference["reference_clients"],
                param_l2=float(norms[position]),
                relative_score=feature["score"],
                z_score=z,
                k=self.k,
                promoted_count=len(feature["promoted"]),
                selected_count=0,
                aggregation_weight=data_sizes[cid] / total,
                attack_clean_l2="",
                replacement_l2=0.0,
                oracle_error_energy_reduction_fraction="",
                decision="KEEP",
            )
            if cid in malicious:
                clean = clean_updates[cid].to(updates[cid].device)
                if not bool(torch.isfinite(clean).all()):
                    raise ValueError("Non-finite normal update")
                selected = feature["promoted"]
                applied[cid] = self.repair_update(updates[cid], clean, selected)
                error, change = updates[cid] - clean, applied[cid] - updates[cid]
                energy = float(error.square().sum())
                remaining = float((applied[cid] - clean).square().sum())
                row.update(
                    selected_count=len(selected),
                    attack_clean_l2=float(error.norm()),
                    replacement_l2=float(change.norm()),
                    decision="REPAIR",
                    oracle_error_energy_reduction_fraction=(energy - remaining) / energy
                    if energy
                    else 0.0,
                )
                selected_total += len(selected)
                for layer in self.layout:
                    count = int(
                        np.count_nonzero(
                            (selected >= layer["start"])
                            & (selected < layer["start"] + layer["numel"])
                        )
                    )
                    layers.append(
                        dict(
                            round=round_number,
                            client_id=cid,
                            layer=layer["name"],
                            numel=layer["numel"],
                            selected_count=count,
                        )
                    )
            else:
                applied[cid] = updates[cid]
            logging.info(
                "Global %d | Client %d %-9s | L2=%.6g | score=%.6g | Z=%.6g | reference=%s | promoted=%d | replaced=%d | weight=%.6f | %s",
                round_number,
                cid,
                row["role"],
                row["param_l2"],
                row["relative_score"],
                z,
                row["reference_candidate"],
                row["promoted_count"],
                row["selected_count"],
                row["aggregation_weight"],
                row["decision"],
            )
            rows.append(row)
        weighted = 0
        for cid in ids:
            weighted += data_sizes[cid] * applied[cid]
        delta = weighted / total
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Non-finite aggregate")
        before = model_vector(model).clone()
        utils.vector_to_model((before + self.args.server_lr * delta).float(), model)
        return dict(
            rows=rows,
            layers=layers,
            features=features,
            reference=reference,
            selected_coordinates=selected_total,
            aggregate_delta_l2=float((model_vector(model) - before).norm()),
        )


class Tee:
    def __init__(self, original, handle):
        self.original, self.handle = original, handle

    def write(self, text):
        self.original.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        if not self.handle.closed:
            self.handle.flush()


class Records:
    def __init__(self, output):
        self.handles, self.writers = {}, {}
        schemas = dict(
            metrics=[
                "round",
                "clean_acc",
                "asr",
                "clean_target_rate",
                "triggered_original_acc",
                "participants",
                "malicious_participants",
                "selected_coordinates",
                "aggregate_delta_l2",
                "elapsed_seconds",
            ],
            clients=[
                "round",
                "client_id",
                "role",
                "reference_candidate",
                "param_l2",
                "relative_score",
                "z_score",
                "k",
                "promoted_count",
                "selected_count",
                "aggregation_weight",
                "attack_clean_l2",
                "replacement_l2",
                "oracle_error_energy_reduction_fraction",
                "decision",
            ],
            layers=["round", "client_id", "layer", "numel", "selected_count"],
        )
        for name, fields in schemas.items():
            handle = (output / "tables" / f"{name}.csv").open("w", newline="", encoding="utf-8")
            self.handles[name] = handle
            self.writers[name] = csv.DictWriter(handle, fieldnames=fields)
            self.writers[name].writeheader()
            handle.flush()

    def write(self, name, rows):
        self.writers[name].writerows(rows)
        self.handles[name].flush()

    def close(self):
        for handle in self.handles.values():
            handle.close()


def evaluation_loaders(test, args):
    """沿用原框架触发器、测试样本范围、初始化顺序。"""
    indices = (test.targets != args.target_class).nonzero().flatten().tolist()
    attacked = copy.deepcopy(test)
    utils.poison_dataset(attacked, args, indices, poison_all=True)
    original = copy.deepcopy(test)
    utils.poison_dataset(original, args, indices, poison_all=True, modify_label=False)

    def loader(dataset):
        return DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False
        )

    return dict(
        clean=loader(test),
        asr=loader(utils.DatasetSplit(attacked, indices)),
        triggered=loader(utils.DatasetSplit(original, indices)),
        clean_non_target=loader(Subset(test, indices)),
    )


@torch.no_grad()
def accuracy(model, loader, device, target=None):
    correct, count = 0, 0
    for images, labels in loader:
        predicted = model(images.to(device)).argmax(dim=1)
        correct += int((predicted == (labels.to(device) if target is None else target)).sum())
        count += predicted.numel()
    if not count:
        raise ValueError("Empty evaluation subset")
    return correct / count


@torch.no_grad()
def evaluate(model, loaders, args):
    model.eval()
    clean_acc = accuracy(model, loaders["clean"], args.device)
    state = rng_state()  # 额外评估同分批的非目标干净图片，不改变训练 RNG。
    try:
        clean_target = accuracy(
            model, loaders["clean_non_target"], args.device, target=args.target_class
        )
    finally:
        restore_rng(state)
    return dict(
        clean_acc=clean_acc,
        clean_target_rate=clean_target,
        asr=accuracy(model, loaders["asr"], args.device),
        triggered_original_acc=accuracy(model, loaders["triggered"], args.device),
    )


def generate_report(output, metadata):
    """独立 B 组报告，不依赖四组实验报告工具。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "tables/metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# RAMP B：差集坐标替换",
        "",
        f'状态：{metadata["status"]}；完成 {len(rows)} 个全局轮次。',
        "",
        "完整实现位于 src/ramp_b.py。复用框架数据、模型、本地训练模块；全部客户端按数据量加权 FedAvg，不淘汰。",
        "",
        "真实恶意身份指定修复对象，配对正常更新提供替换值，属于 Oracle 验证。",
        "",
        "poison_frac 是非目标类候选集的比例。Z 仅展示，不参与淘汰。",
        "",
    ]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        x = [int(r["round"]) for r in rows]
        for ax, keys in zip(
            axes, [("clean_acc", "triggered_original_acc"), ("asr", "clean_target_rate")]
        ):
            for key in keys:
                ax.plot(x, [100 * float(r[key]) for r in rows], label=key)
            ax.set(xlabel="Global round", ylabel="Percent", ylim=(0, 100))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("RAMP B: promoted-coordinate Oracle repair")
        fig.savefig(output / "figures/training.png", dpi=170)
        plt.close(fig)
        lines += [
            f'最后一轮 ACC：{float(rows[-1]["clean_acc"]):.2%}；ASR：{float(rows[-1]["asr"]):.2%}。',
            "",
            "![Training](figures/training.png)",
            "",
        ]
    lines += [
        "## 修改位置",
        "",
        "- RAMPB.build_reference：参考更新。",
        "- RAMPB.select_coordinates：坐标选择。",
        "- RAMPB.repair_update：修复规则。",
        "- RAMPB.aggregate：修复对象和聚合。",
        "- parse_args：默认参数；train：训练循环；evaluate：评估。",
        "",
        "## 文件",
        "",
        "run.json：设置、状态、源码与数据指纹；console.log：完整输出；tables：逐轮/客户端/层统计、划分与投毒索引；"
        "snapshots：默认最后一轮聚合前模型、恶意配对更新和坐标；final_model.pt：最后全局模型；source_snapshot：src 源码。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", choices=["cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["resnet9"], default="resnet9")
    parser.add_argument("--attack", choices=["badnet"], default="badnet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num_agents", "--clients", type=int, default=20)
    parser.add_argument("--num_corrupt", "--corrupt", type=int, default=4)
    parser.add_argument("--agent_frac", type=float, default=1.0)
    parser.add_argument("--poison_frac", "--poison-frac", type=float, default=0.3)
    parser.add_argument("--local_ep", "--local-ep", type=int, default=2)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--client_lr", type=float, default=0.1)
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--lr_decay", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--non_iid", dest="non_iid", action="store_true", default=True)
    parser.add_argument("--iid", dest="non_iid", action="store_false")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--target_class", type=int, default=7)
    parser.add_argument("--pattern_type", default="plus")
    parser.add_argument("--topk_ratio", "--topk-ratio", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--reference", choices=["median", "mean"], default="median")
    parser.add_argument("--l2_lower", type=float, default=25.0)
    parser.add_argument("--l2_upper", type=float, default=75.0)
    parser.add_argument("--min_reference", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, choices=[0], default=0)
    parser.add_argument("--name", default="b_only_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot_rounds", "--snapshot-rounds", type=int, nargs="*", default=[])
    parser.add_argument("--train_limit", "--train-limit", type=int, default=0, help="验证专用；0 为完整数据")
    parser.add_argument("--test_limit", "--test-limit", type=int, default=0, help="验证专用；0 为完整测试集")
    args = parser.parse_args(argv)
    if (
        args.rounds < 1
        or args.local_ep < 1
        or not 0 <= args.num_corrupt < args.num_agents
        or args.bs < 1
        or not 0 < args.agent_frac <= 1
        or math.floor(args.agent_frac * args.num_agents) < 1
        or not 0 <= args.poison_frac <= 1
        or not 0 < args.topk_ratio <= 1
        or not math.isfinite(args.epsilon)
        or args.epsilon <= 0
        or not 0 <= args.l2_lower < args.l2_upper <= 100
        or args.min_reference < 1
        or not 0 <= args.target_class < 10
        or args.beta <= 0
        or args.seed < 0
        or args.train_limit < 0
        or args.test_limit < 0
        or any(r < 1 or r > args.rounds for r in args.snapshot_rounds)
    ):
        parser.error("Invalid experiment parameters")
    if not args.name or args.name in (".", "..") or "/" in args.name or "\\" in args.name:
        parser.error("--name must be a single directory name")
    args.aggr, args.num_target, args.cease_poison = "avg", 10, 100000
    return args


def output_path(args):
    split = f"noniid_beta{args.beta:g}" if args.non_iid else "iid"
    return (
        args.output
        if args.output is not None
        else ROOT
        / "research_results/RAMP_oracle_B"
        / f"{args.data}_{args.model}"
        / split
        / f"clients{args.num_agents}_malicious{args.num_corrupt}_{100*args.num_corrupt/args.num_agents:g}pct"
        / args.attack
        / args.name
    ).resolve()


def train(args):
    output = output_path(args)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("tables", "figures", "snapshots", "source_snapshot"):
        (output / name).mkdir()
    metadata = dict(
        status="initializing",
        settings=vars(args).copy(),
        observed_rounds=0,
        seed=args.seed,
        implementation="self-contained src/ramp_b.py",
        formula="abs(delta)/(abs(reference)+epsilon)",
        oracle="known malicious IDs; paired normal updates",
        source_sha256={},
    )
    for file in sorted(SRC.glob("*.py")):
        metadata["source_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        shutil.copy2(file, output / "source_snapshot" / file.name)
    save_json(output / "run.json", metadata)
    stdout, stderr = sys.stdout, sys.stderr
    console = (output / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(stdout, console), Tee(stderr, console)
    logger = logging.getLogger()
    old_handlers = list(logger.handlers)
    for handler in old_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    records, failure, started = Records(output), None, time.perf_counter()
    try:
        os.chdir(SRC)
        if str(args.device).startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        logging.info("Standalone RAMP B | output=%s | settings=%s", output, args)
        logging.info(
            "Every-round Oracle repair; weighted FedAvg includes all participants; filtering=OFF"
        )
        train_data, test_data = utils.get_datasets(args.data)
        for dataset, limit in ((train_data, args.train_limit), (test_data, args.test_limit)):
            if limit:
                dataset.data = dataset.data[:limit].copy()
                dataset.targets = dataset.targets[:limit].clone()
        groups = (
            utils.distribute_data_dirichlet(train_data, args)
            if args.non_iid
            else utils.distribute_data(train_data, args, n_classes=10)
        )
        if any(len(indices) < args.bs for indices in groups.values()):
            raise ValueError(
                "A client has fewer than one full batch; reduce --bs or increase data size"
            )
        loaders = evaluation_loaders(test_data, args)
        model = models.get_model(args.data, args).to(args.device)
        method = RAMPB(args, model)
        metadata.update(
            initial_model_sha256=tensor_hash(model_vector(model)),
            layout=method.layout,
            parameter_count=method.dimension,
            torch_version=torch.__version__,
        )
        agents, sizes, poison = [], {}, {}
        for cid in range(args.num_agents):
            agent = Agent(cid, args, train_data, groups[cid], backdoor_train_dataset=None)
            agent.is_malicious = int(cid < args.num_corrupt)
            agents.append(agent)
            sizes[cid] = agent.n_data
            if agent.is_malicious:
                poison[cid] = [
                    int(i)
                    for i in groups[cid]
                    if int(agent.clean_backup_dataset.targets[i])
                    != int(agent.train_dataset.dataset.targets[i])
                ]
        partition = {int(cid): list(map(int, ids)) for cid, ids in groups.items()}
        metadata.update(
            partition_sha256=hashlib.sha256(
                json.dumps(partition, sort_keys=True).encode()
            ).hexdigest(),
            poison_plan_sha256=hashlib.sha256(
                json.dumps(poison, sort_keys=True).encode()
            ).hexdigest(),
            status="running",
        )
        save_json(
            output / "tables/data_partition.json",
            dict(partition=partition, poisoned_indices=poison),
        )
        save_json(output / "run.json", metadata)
        criterion, snapshots = torch.nn.CrossEntropyLoss().to(args.device), {
            args.rounds,
            *args.snapshot_rounds,
        }
        for rnd in range(1, args.rounds + 1):
            baseline = model_vector(model).clone()
            ids = sorted(
                np.random.choice(
                    args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False
                ).tolist()
            )
            malicious = [c for c in ids if c < args.num_corrupt]
            logging.info(
                "Global %d | participants=%s | malicious=%s | benign=%s",
                rnd,
                ids,
                malicious,
                [c for c in ids if c not in malicious],
            )
            updates, clean = {}, {}
            for cid in ids:
                if cid in malicious:
                    updates[cid], clean[cid] = method.paired_train(
                        agents[cid], model, criterion, rnd
                    )
                else:
                    updates[cid] = agents[cid].local_train(
                        model, criterion, rnd, neurotoxin_mask={}
                    )
                utils.vector_to_model(baseline.clone(), model)
            if rnd == 1:
                metadata["first_round_attack_sha256"] = {
                    str(c): tensor_hash(updates[c]) for c in ids
                }
                metadata["first_round_clean_sha256"] = {
                    str(c): tensor_hash(clean[c]) for c in malicious
                }
            before = model_state(model) if rnd in snapshots else None
            details = method.aggregate(model, updates, clean, sizes, rnd)
            logging.info(
                "Global %d | reference=%s | fallback=%s | replaced_coordinates=%d",
                rnd,
                details["reference"]["reference_clients"],
                details["reference"]["fallback"],
                details["selected_coordinates"],
            )
            records.write("clients", details["rows"])
            records.write("layers", details["layers"])
            if rnd in snapshots:
                torch.save(
                    dict(
                        round=rnd,
                        global_before=before,
                        attack_updates={c: updates[c].detach().cpu() for c in malicious},
                        clean_updates=clean,
                        reference=details["reference"],
                        layout=method.layout,
                        indices={
                            str(c): {
                                key: torch.as_tensor(details["features"][c][key].copy())
                                for key in ("raw", "relative", "promoted")
                            }
                            for c in malicious
                        },
                        client_weights={c: sizes[c] / sum(sizes[i] for i in ids) for c in ids},
                    ),
                    output / "snapshots" / f"global_{rnd:03d}.pt",
                )
            scores = evaluate(model, loaders, args)
            records.write(
                "metrics",
                [
                    dict(
                        round=rnd,
                        **scores,
                        participants=len(ids),
                        malicious_participants=len(malicious),
                        selected_coordinates=details["selected_coordinates"],
                        aggregate_delta_l2=details["aggregate_delta_l2"],
                        elapsed_seconds=time.perf_counter() - started,
                    )
                ],
            )
            logging.info(
                "Global %d | Clean ACC=%.6f | ASR=%.6f | Clean target rate=%.6f | Triggered original ACC=%.6f",
                rnd,
                scores["clean_acc"],
                scores["asr"],
                scores["clean_target_rate"],
                scores["triggered_original_acc"],
            )
            metadata["observed_rounds"] = rnd
            save_json(output / "run.json", metadata)
            del updates, clean, details
        final = model_state(model)
        torch.save(dict(round=args.rounds, state_dict=final), output / "final_model.pt")
        metadata.update(status="completed", final_model_sha256=tensor_hash(model_vector(model)))
    except BaseException as error:
        failure = error
        metadata.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        traceback.print_exc()
    finally:
        records.close()
        metadata.update(
            elapsed_seconds=time.perf_counter() - started,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        )
        save_json(output / "run.json", metadata)
        try:
            generate_report(output, metadata)
            metadata["report_status"] = "generated"
        except Exception as error:
            metadata["report_status"] = f"failed: {error}"
            traceback.print_exc()
            if failure is None:
                failure = error
        save_json(output / "run.json", metadata)
        print(f'B status={metadata["status"]}; report={output / "report.md"}', flush=True)
        logger.removeHandler(handler)
        handler.close()
        for old in old_handlers:
            logger.addHandler(old)
        sys.stdout, sys.stderr = stdout, stderr
        console.close()
    if failure is not None:
        raise SystemExit(130 if isinstance(failure, KeyboardInterrupt) else 1)


def main():
    os.environ.setdefault("TZ", "Asia/Shanghai")
    if hasattr(time, "tzset"):
        time.tzset()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if output_path(args).exists():
        raise SystemExit(f"Refuse to overwrite {output_path(args)}; use a new --name or --output")
    train(args)


if __name__ == "__main__":
    main()
