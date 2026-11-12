"""Oracle coordinate replacement, isolated from the native training entry."""
import copy
import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


GROUPS = ('A_no_repair', 'B_promoted_repair', 'C_random_repair', 'D_full_clean')


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def layout_of(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    layout, offset, full_offset = [], 0, 0
    for name, value in model.state_dict().items():
        if name in trainable:
            layout.append(dict(name=name, numel=value.numel(), start=offset,
                               full_start=full_offset, shape=list(value.shape)))
            offset += value.numel()
        full_offset += value.numel()
    # This study is explicitly restricted to the existing buffer-free ResNet9.
    if offset != full_offset:
        raise ValueError('Oracle study requires the buffer-free ResNet9 state layout')
    return layout, offset


@torch.no_grad()
def locate(updates, layout, ratio, epsilon, reference='median'):
    ids = list(updates)
    matrix = torch.stack([updates[c].detach() for c in ids])
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError('Non-finite uploaded update')
    norms = matrix.norm(dim=1)
    low, high = torch.quantile(norms, .25), torch.quantile(norms, .75)
    candidate = torch.where((norms >= low) & (norms <= high))[0]
    fallback = candidate.numel() < min(8, len(ids))
    if fallback:
        candidate = torch.arange(len(ids), device=matrix.device)
    selected = matrix[candidate]
    ref = selected.median(dim=0).values if reference == 'median' else selected.mean(dim=0)
    denominator = ref.abs() + epsilon
    k = max(1, int(matrix.shape[1] * ratio))
    results = {}
    candidate_ids = [ids[i] for i in candidate.cpu().tolist()]
    for row, cid in enumerate(ids):
        absolute = matrix[row].abs()
        relative = absolute / denominator
        raw = np.sort(torch.topk(absolute, k, sorted=False).indices.cpu().numpy())
        rel = np.sort(torch.topk(relative, k, sorted=False).indices.cpu().numpy())
        promoted = np.setdiff1d(rel, raw, assume_unique=True)
        results[cid] = dict(raw=raw, relative=rel, promoted=promoted,
                            param_l2=float(norms[row]),
                            relative_score=float(torch.topk(relative, k, sorted=False).values.mean()),
                            reference_candidate=cid in candidate_ids)
    return results, dict(k=k, reference_client_ids=candidate_ids,
                        reference_fallback=fallback, l2_low=float(low), l2_high=float(high),
                        reference_abs_below_epsilon_fraction=float((ref.abs() < epsilon).float().mean()))


def layer_random(layout, raw, promoted, seed, round_number, client_id):
    """Same layer counts and budget; same eligible universe as prior Top-K null."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_number, client_id, 9271]))
    pieces = []
    for layer in layout:
        start, end = layer['start'], layer['start'] + layer['numel']
        count = int(np.count_nonzero((promoted >= start) & (promoted < end)))
        if count:
            blocked = raw[(raw >= start) & (raw < end)]
            eligible = np.setdiff1d(np.arange(start, end), blocked, assume_unique=True)
            if eligible.size < count:
                raise ValueError('Layer random budget exceeds eligible coordinates')
            pieces.append(rng.choice(eligible, size=count, replace=False))
    return np.sort(np.concatenate(pieces)).astype(np.int64) if pieces else np.empty(0, np.int64)


def apply_replacement(attack, clean, indices, group):
    if group == GROUPS[0]:
        return attack
    if group == GROUPS[3]:
        return clean.to(attack.device, dtype=attack.dtype).clone()
    if group not in GROUPS:
        raise ValueError(group)
    result = attack.clone()
    idx = torch.as_tensor(indices, dtype=torch.long, device=attack.device)
    result[idx] = clean.to(attack.device, dtype=attack.dtype)[idx]
    return result


def cpu_model_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def matched_clean_target_rate(model, clean_loader, args):
    """Use the ASR loader's non-target image ordering/batches, without consuming RNG.

    ResNet9 uses batch statistics even in eval mode, so reuse of predictions from
    the all-class clean loader would give a different batch composition.
    """
    from torch.utils.data import DataLoader, Subset
    targets = torch.as_tensor(clean_loader.dataset.targets)
    indices = torch.where(targets != args.target_class)[0].tolist()
    loader = DataLoader(Subset(clean_loader.dataset, indices), batch_size=clean_loader.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    correct, count = 0, 0
    with preserve_rng():
        for inputs, _ in loader:
            predicted = model(inputs.to(args.device)).argmax(dim=1)
            correct += int((predicted == args.target_class).sum())
            count += predicted.numel()
    return correct / count if count else float('nan')


def paired_local_train(original, agent, model, criterion, round_number, kwargs):
    """Run attack first, replay the clean branch from exactly the same RNG/model."""
    import utils
    from torch.utils.data import DataLoader
    before_rng, before_model = rng_state(), cpu_model_state(model)
    attack = original(agent, model, criterion, round_number, **kwargs)
    after_rng, after_model, after_mode = rng_state(), cpu_model_state(model), model.training
    clean_agent = copy.copy(agent)
    clean_agent.args = copy.copy(agent.args)
    clean_agent.args.num_corrupt = 0
    clean_agent.is_malicious = 0
    clean_agent.train_dataset = utils.DatasetSplit(agent.clean_backup_dataset, agent.data_idxs)
    clean_agent.train_loader = DataLoader(clean_agent.train_dataset, batch_size=agent.args.bs,
                                         shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    try:
        model.load_state_dict(before_model)
        restore_rng(before_rng)
        clean = original(clean_agent, model, criterion, round_number, **kwargs).detach().cpu().clone()
    finally:
        model.load_state_dict(after_model)
        model.train(after_mode)
        restore_rng(after_rng)
    if not bool(torch.isfinite(clean).all()):
        raise ValueError('Non-finite paired clean update')
    return attack, clean
