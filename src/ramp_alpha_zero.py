"""RAMP Alpha-Zero：恶意客户端的相对比值 Top-K 中，严格超过阈值的更新坐标置零。
运行：python src/ramp_alpha_zero.py --device cuda:0 --topk_ratio 0.005 --alpha_threshold 0.8
默认先取已知良性客户端，在其 L2 中间 25%-75% 的候选上取 median 参考。
按相对幅值取 Top-K，再与严格阈值取交集；只处理已知恶意客户端。
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


class RAMPAlphaZero:
    """完整逐坐标相对幅值阈值置零方案。"""

    def __init__(self, args, model):
        self.args = args
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.layout, offset = ([], 0)
        for name, value in model.state_dict().items():
            if name not in trainable:
                raise ValueError("This study requires the buffer-free ResNet9")
            self.layout.append(
                dict(name=name, start=offset, numel=value.numel(), shape=list(value.shape))
            )
            offset += value.numel()
        self.dimension = offset
        self.k = max(1, int(self.dimension * args.topk_ratio))

    @torch.no_grad()
    def build_reference(self, updates):
        """先取已知良性客户端，再在其内部按 L2 筛选并构造参考。"""
        ids = list(updates)
        matrix = torch.stack([updates[c].detach() for c in ids])
        if matrix.shape[1] != self.dimension or not bool(torch.isfinite(matrix).all()):
            raise ValueError("Invalid update dimension or non-finite values")
        norms = matrix.norm(dim=1)
        benign_rows = [i for i, cid in enumerate(ids) if cid >= self.args.num_corrupt]
        if not benign_rows:
            raise ValueError("Cannot construct oracle benign reference: no benign participant")
        benign = torch.tensor(benign_rows, dtype=torch.long, device=matrix.device)
        benign_norms = norms[benign]
        lower = torch.quantile(benign_norms, self.args.l2_lower / 100)
        upper = torch.quantile(benign_norms, self.args.l2_upper / 100)
        candidate = benign[(benign_norms >= lower) & (benign_norms <= upper)]
        fallback = candidate.numel() < min(self.args.min_reference, len(benign_rows))
        if fallback:
            candidate = benign
        values = matrix[candidate]
        ref = values.median(dim=0).values if self.args.reference == "median" else values.mean(dim=0)
        reference_ids = [ids[i] for i in candidate.cpu().tolist()]
        assert all((cid >= self.args.num_corrupt for cid in reference_ids))
        info = dict(
            reference_scope="oracle_benign_l2",
            reference_pool_clients=[ids[i] for i in benign_rows],
            reference_clients=reference_ids,
            reference_strategy=self.args.reference,
            fallback=fallback,
            l2_low=float(lower),
            l2_high=float(upper),
            reference_abs_below_epsilon_fraction=float(
                (ref.abs() < self.args.epsilon).float().mean()
            ),
        )
        logging.info(
            "Reference scope=oracle_benign_l2 | benign_pool=%s | L2=[%.6g, %.6g] | reference=%s | method=%s | fallback=%s",
            info["reference_pool_clients"],
            float(lower),
            float(upper),
            reference_ids,
            self.args.reference,
            fallback,
        )
        return (ref, norms, info)

    @torch.no_grad()
    def select_coordinates(self, update, reference):
        """按比值排名取前 K，且比值严格超过阈值，两个条件同时满足才选中。"""
        relative = update.abs() / (reference.abs() + self.args.epsilon)
        if not bool(torch.isfinite(relative).all()):
            raise ValueError("Non-finite relative coordinates")
        top = torch.topk(relative, self.k, sorted=False).indices
        selected = top[relative[top] > self.args.alpha_threshold]
        return dict(
            relative_topk=top.cpu().numpy(),
            above_threshold=selected.cpu().numpy(),
            threshold_count=int((relative > self.args.alpha_threshold).sum()),
            score=float(relative.mean()),
        )

    @torch.no_grad()
    def zero_update(self, update, selected_indices):
        """将选中坐标的 delta 置零；不是把全局模型参数置零。"""
        result = update.clone()
        index = torch.as_tensor(selected_indices, dtype=torch.long, device=update.device)
        result[index] = 0
        return result

    @torch.no_grad()
    def aggregate(self, model, updates, data_sizes, round_number):
        ids = list(updates)
        ref, norms, reference = self.build_reference(updates)
        features = {c: self.select_coordinates(updates[c], ref) for c in ids}
        scores = np.array([features[c]["score"] for c in ids])
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        total = sum((data_sizes[c] for c in ids))
        applied, rows, layers, selected_total = ({}, [], [], 0)
        for position, cid in enumerate(ids):
            target = cid < self.args.num_corrupt
            selected = features[cid]["above_threshold"] if target else np.empty(0, dtype=np.int64)
            applied[cid] = self.zero_update(updates[cid], selected) if target else updates[cid]
            removed_l2 = float((updates[cid] - applied[cid]).norm())
            z = abs(features[cid]["score"] - median) / max(1.4826 * mad, 1e-12)
            row = dict(
                round=round_number,
                client_id=cid,
                role="malicious" if cid < self.args.num_corrupt else "benign",
                reference_candidate=cid in reference["reference_clients"],
                param_l2=float(norms[position]),
                relative_score=features[cid]["score"],
                z_score=z,
                topk_count=self.k,
                threshold_count=features[cid]["threshold_count"],
                above_threshold_count=len(features[cid]["above_threshold"]),
                selected_count=len(selected),
                aggregation_weight=data_sizes[cid] / total,
                removed_l2=removed_l2,
                after_zero_l2=float(applied[cid].norm()),
                decision="ZERO" if target else "KEEP",
            )
            rows.append(row)
            selected_total += len(selected)
            if target:
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
            logging.info(
                "Global %d | Client %d %-9s | L2=%.6g | score=%.6g | Z=%.6g | reference=%s | topk=%d | above_threshold_in_topk=%d | zeroed=%d | removed_L2=%.6g | weight=%.6f | %s",
                round_number,
                cid,
                row["role"],
                row["param_l2"],
                row["relative_score"],
                z,
                row["reference_candidate"],
                self.k,
                row["above_threshold_count"],
                len(selected),
                removed_l2,
                row["aggregation_weight"],
                row["decision"],
            )
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
        self.original, self.handle = (original, handle)

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
        self.handles, self.writers = ({}, {})
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
                "topk_count",
                "threshold_count",
                "above_threshold_count",
                "selected_count",
                "aggregation_weight",
                "removed_l2",
                "after_zero_l2",
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
    correct, count = (0, 0)
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
    state = rng_state()
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "tables/metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    threshold = metadata["settings"]["alpha_threshold"]
    lines = [
        "# RAMP Alpha-Zero",
        "",
        f"状态：{metadata['status']}；完成 {len(rows)} 轮；阈值 {threshold}。",
        "",
        "规则：按 abs(local_delta)/(abs(reference)+epsilon) 从大到小取前 K，再选择比值严格大于阈值的坐标，将其本地 delta 置零。无差集、无配对正常训练。",
        f"Top-K 比例：{metadata['settings']['topk_ratio']}；K=max(1, floor(模型维度×比例))；不足阈值不补足。",
        "参考：已知良性客户端内部按 L2 取中间区间，再取 median（默认）；不足仅回退全部良性。",
        "只处理已知恶意客户端；全部客户端保留原数据量权重 FedAvg；Z 仅展示。",
        "score 是全部坐标相对幅值的均值，不是原先 Top-K 均值，不能与原日志分数直接比较。",
        "",
    ]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for ax, keys in zip(
            axes, [("clean_acc", "triggered_original_acc"), ("asr", "clean_target_rate")]
        ):
            for key in keys:
                ax.plot(
                    [int(r["round"]) for r in rows], [100 * float(r[key]) for r in rows], label=key
                )
            ax.set(xlabel="Global round", ylabel="Percent", ylim=(0, 100))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("RAMP Alpha-Zero: relative magnitude threshold masking")
        fig.savefig(output / "figures/training.png", dpi=170)
        plt.close(fig)
        lines += ["![Training](figures/training.png)", ""]
    lines += [
        "本次独立保存 console.log、run.json、逐轮/客户端/层 CSV、报告/图片、最终模型和源码。默认最后一轮保存更新与选中坐标。",
        "修改 RAMPAlphaZero.build_reference / select_coordinates / zero_update / aggregate 可调整完整方案。",
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
    parser.add_argument("--alpha_threshold", "--threshold", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--reference", choices=["median", "mean"], default="median")
    parser.add_argument("--l2_lower", type=float, default=25.0)
    parser.add_argument("--l2_upper", type=float, default=75.0)
    parser.add_argument("--min_reference", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, choices=[0], default=0)
    parser.add_argument("--name", default="alpha_topk0p1pct_gt0p8_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot_rounds", "--snapshot-rounds", type=int, nargs="*", default=[])
    parser.add_argument("--train_limit", "--train-limit", type=int, default=0, help="验证专用；0 为完整数据")
    parser.add_argument("--test_limit", "--test-limit", type=int, default=0, help="验证专用；0 为完整测试集")
    args = parser.parse_args(argv)
    if (
        args.rounds < 1
        or args.local_ep < 1
        or (not 0 <= args.num_corrupt < args.num_agents)
        or (args.bs < 1)
        or (not 0 < args.agent_frac <= 1)
        or (math.floor(args.agent_frac * args.num_agents) < 1)
        or (not 0 <= args.poison_frac <= 1)
        or (not math.isfinite(args.topk_ratio) or not 0 < args.topk_ratio <= 1)
        or (not math.isfinite(args.alpha_threshold) or args.alpha_threshold < 0)
        or (not math.isfinite(args.epsilon))
        or (args.epsilon <= 0)
        or (not 0 <= args.l2_lower < args.l2_upper <= 100)
        or (args.min_reference < 1)
        or (not 0 <= args.target_class < 10)
        or (args.beta <= 0)
        or (args.seed < 0)
        or (args.train_limit < 0)
        or (args.test_limit < 0)
        or any((r < 1 or r > args.rounds for r in args.snapshot_rounds))
    ):
        parser.error("Invalid experiment parameters")
    if not args.name or args.name in (".", "..") or "/" in args.name or ("\\" in args.name):
        parser.error("--name must be a single directory name")
    args.zero_scope = "oracle_malicious"
    args.aggr, args.num_target, args.cease_poison = ("avg", 10, 100000)
    return args


def output_path(args):
    split = f"noniid_beta{args.beta:g}" if args.non_iid else "iid"
    return (
        args.output
        if args.output is not None
        else ROOT
        / "research_results/RAMP_alpha_zero"
        / f"{args.data}_{args.model}"
        / split
        / f"clients{args.num_agents}_malicious{args.num_corrupt}_{100 * args.num_corrupt / args.num_agents:g}pct"
        / args.attack
        / args.zero_scope
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
        reference_scope="oracle_benign_l2",
        implementation="self-contained src/ramp_alpha_zero.py",
        formula="relative Top-K intersect {abs(delta)/(abs(reference)+epsilon) > alpha_threshold}",
        zero_rule="selected local update coordinates set to zero; client data weights unchanged",
        source_sha256={},
    )
    for file in sorted(SRC.glob("*.py")):
        metadata["source_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        shutil.copy2(file, output / "source_snapshot" / file.name)
    save_json(output / "run.json", metadata)
    stdout, stderr = (sys.stdout, sys.stderr)
    console = (output / "console.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = (Tee(stdout, console), Tee(stderr, console))
    logger = logging.getLogger()
    old_handlers = list(logger.handlers)
    for handler in old_handlers:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    records, failure, started = (Records(output), None, time.perf_counter())
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
        logging.info("Standalone RAMP Alpha-Zero | output=%s | settings=%s", output, args)
        logging.info(
            "Every-round coordinate zeroing; weighted FedAvg includes all participants; filtering=OFF"
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
        if any((len(indices) < args.bs for indices in groups.values())):
            raise ValueError(
                "A client has fewer than one full batch; reduce --bs or increase data size"
            )
        loaders = evaluation_loaders(test_data, args)
        model = models.get_model(args.data, args).to(args.device)
        method = RAMPAlphaZero(args, model)
        metadata.update(
            initial_model_sha256=tensor_hash(model_vector(model)),
            layout=method.layout,
            parameter_count=method.dimension,
            torch_version=torch.__version__,
        )
        agents, sizes, poison = ([], {}, {})
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
        criterion, snapshots = (
            torch.nn.CrossEntropyLoss().to(args.device),
            {args.rounds, *args.snapshot_rounds},
        )
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
            updates = {}
            for cid in ids:
                updates[cid] = agents[cid].local_train(model, criterion, rnd, neurotoxin_mask={})
                utils.vector_to_model(baseline.clone(), model)
            if rnd == 1:
                metadata["first_round_attack_sha256"] = {
                    str(c): tensor_hash(updates[c]) for c in ids
                }
            before = model_state(model) if rnd in snapshots else None
            details = method.aggregate(model, updates, sizes, rnd)
            logging.info(
                "Global %d | reference=%s | fallback=%s | zeroed_coordinates=%d",
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
                        attack_updates={
                            c: updates[c].detach().cpu()
                            for c in ids
                            if c < args.num_corrupt
                        },
                        zero_scope=args.zero_scope,
                        reference=details["reference"],
                        layout=method.layout,
                        indices={
                            str(c): {
                                key: torch.as_tensor(details["features"][c][key].copy())
                                for key in ("relative_topk", "above_threshold")
                            }
                            for c in ids
                            if c < args.num_corrupt
                        },
                        client_weights={c: sizes[c] / sum((sizes[i] for i in ids)) for c in ids},
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
            del updates, details
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
        print(
            f"RAMP Alpha-Zero status={metadata['status']}; report={output / 'report.md'}",
            flush=True,
        )
        logger.removeHandler(handler)
        handler.close()
        for old in old_handlers:
            logger.addHandler(old)
        sys.stdout, sys.stderr = (stdout, stderr)
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
