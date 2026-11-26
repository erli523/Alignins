"""Full-participation comparison runner using the existing dataset/model/agents."""

import copy
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import algorithms
from .__main__ import source_manifest
from .native_compat import bind_pgd_projection

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def native_helpers():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import ramp_attack_eval
    return ramp_attack_eval


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def file_hashes():
    paths = sorted(SRC.glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    paths.append(Path(__file__).with_name("sources.json"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def native_args(settings, helpers):
    args = helpers.parse_args([
        "--data", settings.data, "--attack", settings.attack, "--output", str(settings.output),
        "--rounds", str(settings.rounds), "--client_lr", str(settings.client_lr),
        "--device", settings.device, "--max_batches", str(settings.max_batches),
        "--test_limit", str(settings.test_limit),
    ])
    for name in ["num_agents", "num_corrupt", "beta", "poison_frac", "local_ep", "bs", "server_lr", "lr_decay", "threads"]:
        setattr(args, name, getattr(settings, name))
    args.aggr = "lockdown" if settings.method == "lockdown" else "avg"
    args.dense_ratio = settings.lockdown_density if settings.method == "lockdown" else 0.25
    args.anneal_factor = settings.lockdown_anneal
    args.dis_check_gradient = False
    args.mask_init = "ERK"
    args.same_mask = settings.lockdown_same_mask
    return args


def make_agents(settings, args, helpers, train_dataset, groups, model):
    sparse_masks = None
    if settings.method == "lockdown":
        from agent_sparse import Agent as AgentClass
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed + 91001)
            sparsities = helpers.core.utils.calculate_sparsities(args, model.state_dict(), distribution="ERK")
            shared = helpers.core.utils.init_masks(model.state_dict(), sparsities)
            sparse_masks = [copy.deepcopy(shared) if args.same_mask else helpers.core.utils.init_masks(model.state_dict(), sparsities)
                            for client_id in range(args.num_agents)]
    else:
        AgentClass = helpers.Agent
    agents, sizes = [], {}
    for client_id in range(args.num_agents):
        agent = AgentClass(client_id, args, train_dataset, groups[client_id],
                           mask=None if sparse_masks is None else sparse_masks[client_id])
        agent.is_malicious = int(client_id < args.num_corrupt)
        if args.attack == "pgd":
            bind_pgd_projection(agent)
        if settings.max_batches:
            subset = Subset(agent.train_dataset, range(min(agent.n_data, settings.max_batches * args.bs)))
            agent.train_loader = DataLoader(subset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
        if len(agent.train_loader) == 0:
            raise ValueError(f"Client {client_id} has no complete training batch")
        agents.append(agent)
        sizes[client_id] = agent.n_data
    return agents, sizes


def aggregate(settings, base, updates, sizes, round_number, generator, personal, masks):
    if settings.method in ["fedavg", "lockdown"]:
        return algorithms.fedavg(base, updates, sizes, settings.server_lr)
    if settings.method == "rlr":
        return algorithms.rlr(base, updates, sizes, settings.rlr_threshold, settings.server_lr)
    if settings.method == "flame":
        return algorithms.flame(base, updates, settings.flame_noise, generator)
    if settings.method == "crfl":
        return algorithms.crfl(base, updates, sizes, settings.crfl_clip, settings.crfl_noise,
                               round_number == settings.rounds, generator)
    if settings.method == "prodigal":
        local_models = {client_id: personal[client_id] + update for client_id, update in updates.items()}
        return algorithms.prodigal(local_models, personal, masks, settings.prodigal_threshold)
    raise ValueError(f"No implementation for {settings.method}")


def run_experiment(settings, metadata, helpers):
    core = helpers.core
    args = native_args(settings, helpers)
    torch.set_num_threads(settings.threads)
    if settings.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(settings.device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_rng(settings.seed)
    train_dataset, test_dataset = core.utils.get_datasets(args.data)
    if len(train_dataset) < 10 * args.num_agents:
        raise ValueError("Native Dirichlet partitioner requires at least ten examples per client")
    if settings.test_limit:
        test_dataset.data = test_dataset.data[:settings.test_limit].copy()
        test_dataset.targets = test_dataset.targets[:settings.test_limit].clone()
    groups = core.utils.distribute_data_dirichlet(train_dataset, args)
    metadata["partition_sha256"] = hashlib.sha256(json.dumps(
        {int(key): list(map(int, values)) for key, values in groups.items()}, sort_keys=True,
    ).encode()).hexdigest()
    model = core.models.get_model(args.data, args).to(args.device)
    layout = algorithms.ParameterLayout(model)
    metadata["initial_model_sha256"] = core.tensor_hash(layout.vector(model))
    agents, sizes = make_agents(settings, args, helpers, train_dataset, groups, model)
    loaders = helpers.evaluation_loaders(test_dataset, args, agents)
    metadata.update(status="running", local_data_sizes=sizes, model_parameters=layout.size)
    core.save_json(settings.output / "config.json", metadata)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    personal = {client_id: layout.vector(model) for client_id in sizes} if settings.method == "prodigal" else None
    feedback_mask = {}
    fields = ["round", "clean_acc", "asr", "clean_target_rate", "triggered_original_acc", "round_seconds", "elapsed_seconds", "aggregate_delta_l2", "evaluation"]
    rows = []
    started = time.perf_counter()
    with (settings.output / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics, \
            (settings.output / "rounds.jsonl").open("w", encoding="utf-8") as details:
        writer = csv.DictWriter(metrics, fieldnames=fields)
        writer.writeheader()
        for round_number in range(1, settings.rounds + 1):
            tick = time.perf_counter()
            base = layout.vector(model)
            updates, masks = {}, {}
            old_masks = [copy.deepcopy(agent.mask) for agent in agents] if settings.method == "lockdown" else []
            for client_id, agent in enumerate(agents):
                seed_rng((settings.seed + 1000003 * round_number + 1009 * client_id) % 2**32)
                layout.load(model, personal[client_id] if personal is not None else base)
                if settings.method == "lockdown":
                    update = agent.local_train(model, criterion, round_number, global_mask={}, neurotoxin_mask=feedback_mask, updates_dict={})
                else:
                    update = agent.local_train(model, criterion, round_number, neurotoxin_mask=feedback_mask)
                updates[client_id] = update.detach().cpu().clone()
                if settings.method == "prodigal":
                    masks[client_id] = layout.topk_gradients(model, settings.prodigal_topk)
            generator = torch.Generator().manual_seed(settings.seed + 2000003 * round_number)
            result = aggregate(settings, base, updates, sizes, round_number, generator, personal, masks)
            layout.load(model, result.model)
            personal = result.personalized
            feedback_mask = layout.feedback_mask(result.feedback, args.dense_ratio)
            evaluation = "global"
            eval_model = model
            if settings.method == "lockdown":
                eval_model = algorithms.consensus_model(model, old_masks, settings.lockdown_threshold)
                evaluation = "consensus_cleaned"
            elif settings.method == "crfl":
                clipped_model = copy.deepcopy(model)
                layout.load(clipped_model, result.evaluation_model)
                eval_model = algorithms.SmoothedClassifier(clipped_model, settings.crfl_test_noise, settings.crfl_samples,
                                                           settings.seed + round_number * 3000017)
                del clipped_model
                evaluation = "smoothed_votes_no_certificate"
            state = core.rng_state()
            try:
                scores = core.evaluate(eval_model, loaders, args)
            finally:
                core.restore_rng(state)
            row = dict(round=round_number, **scores, evaluation=evaluation,
                       aggregate_delta_l2=float((result.model - base).norm()),
                       round_seconds=time.perf_counter() - tick, elapsed_seconds=time.perf_counter() - started)
            rows.append(row)
            writer.writerow(row)
            metrics.flush()
            details.write(json.dumps({"round": round_number, **result.diagnostics}) + "\n")
            details.flush()
            metadata["observed_rounds"] = round_number
            core.save_json(settings.output / "config.json", metadata)
            logging.info("Round %d | ACC %.6f | ASR %.6f | %s", round_number, scores["clean_acc"], scores["asr"], evaluation)
            del updates, masks, result, eval_model, old_masks
    tail = rows[-10:]
    core.save_json(settings.output / "summary.json", {
        "method": settings.method, "rounds": len(rows), "window_count": len(tail),
        "clean_acc_last10": sum(row["clean_acc"] for row in tail) / len(tail),
        "asr_last10": sum(row["asr"] for row in tail) / len(tail),
        "smoke_only": metadata["smoke_only"], "certified": False,
        "evaluation": rows[-1]["evaluation"],
    })


def train(settings):
    helpers = native_helpers()
    metadata = {
        "status": "initializing", "observed_rounds": 0,
        "started_at": datetime.now(timezone.utc).isoformat(), "settings": vars(settings),
        "source": source_manifest()["methods"][settings.method],
        "source_sha256": file_hashes(),
        "smoke_only": bool(settings.max_batches or settings.test_limit),
        "protocol": "Full participation; native CIFAR/ResNet9/attack agents; deterministic per-client/round RNG",
        "certified": False,
        "compatibility": ["Instance-local PGD vector-to-state return contract; native radius and projection timing unchanged"] if settings.attack == "pgd" else [],
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "torchvision", "scikit-learn"]},
    }
    if settings.method == "flame":
        metadata["versions"]["hdbscan"] = importlib.metadata.version("hdbscan")
    settings.output.mkdir(parents=True, exist_ok=False)
    helpers.core.save_json(settings.output / "config.json", metadata)
    old_cwd, old_stdout, old_stderr = Path.cwd(), sys.stdout, sys.stderr
    logger = logging.getLogger()
    old_handlers, old_level = logger.handlers[:], logger.level
    with (settings.output / "training.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            sys.stdout, sys.stderr = helpers.core.Tee(old_stdout, log), helpers.core.Tee(old_stderr, log)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)
            os.chdir(SRC)
            run_experiment(settings, metadata, helpers)
            metadata["status"] = "completed"
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            helpers.core.save_json(settings.output / "config.json", metadata)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            logger.handlers, logger.level = old_handlers, old_level
            os.chdir(old_cwd)
