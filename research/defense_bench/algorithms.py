"""Defense equations; provenance and adaptation boundaries are in SOURCES.md."""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RoundResult:
    model: torch.Tensor
    feedback: torch.Tensor
    diagnostics: dict
    personalized: dict | None = None
    evaluation_model: torch.Tensor | None = None


class ParameterLayout:
    def __init__(self, model):
        parameters = dict(model.named_parameters())
        if list(model.state_dict()) != list(parameters):
            raise ValueError("This adapter supports parameter-only models (native ResNet9), not models with persistent buffers")
        self.parts = []
        offset = 0
        for name, parameter in parameters.items():
            self.parts.append((name, slice(offset, offset + parameter.numel()), parameter.shape))
            offset += parameter.numel()
        self.size = offset

    def vector(self, model):
        return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])

    @torch.no_grad()
    def load(self, model, vector):
        if vector.ndim != 1 or vector.numel() != self.size or not torch.isfinite(vector).all():
            raise ValueError("Invalid model vector")
        parameters = dict(model.named_parameters())
        for name, section, shape in self.parts:
            parameters[name].copy_(vector[section].reshape(shape))

    def topk_gradients(self, model, ratio):
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("Top-K ratio must be in (0, 1]")
        mask = torch.zeros(self.size, dtype=torch.bool)
        parameters = dict(model.named_parameters())
        for name, section, shape in self.parts:
            gradient = parameters[name].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise ValueError(f"Missing/nonfinite final local minibatch gradient: {name}")
            count = int(gradient.numel() * ratio)
            indices = gradient.detach().reshape(-1).abs().topk(count).indices.cpu()
            mask[section][indices] = True
        return mask

    def feedback_mask(self, update, ratio):
        result = {}
        for name, section, shape in self.parts:
            values = update[section]
            indices = (-values.abs()).topk(int(values.numel() * ratio)).indices
            mask = torch.zeros_like(values)
            mask[indices] = 1
            result[name] = mask.reshape(shape)
        return result


def checked_updates(base, updates):
    if not updates:
        raise ValueError("At least one client update is required")
    if base.ndim != 1 or not torch.isfinite(base).all():
        raise ValueError("Invalid global model")
    for client_id, update in updates.items():
        if update.shape != base.shape or update.device != base.device or not torch.isfinite(update).all():
            raise ValueError(f"Invalid update from client {client_id}")


def weighted_mean(updates, sizes):
    weights = [float(sizes[client_id]) for client_id in updates]
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Client sample counts must be finite and positive")
    total = sum(weights)
    result = torch.zeros_like(next(iter(updates.values())))
    for update, weight in zip(updates.values(), weights):
        result.add_(update, alpha=weight / total)
    return result


def fedavg(base, updates, sizes, server_lr=1.0):
    checked_updates(base, updates)
    average = weighted_mean(updates, sizes)
    return RoundResult(base + server_lr * average, average, {"retained_clients": list(updates)})


def rlr(base, updates, sizes, threshold, server_lr=1.0):
    checked_updates(base, updates)
    if not 0 <= threshold <= len(updates):
        raise ValueError("RLR threshold must lie between zero and participant count")
    agreement = torch.zeros_like(base)
    for update in updates.values():
        agreement.add_(update.sign())
    positive = agreement.abs() >= threshold
    rates = torch.where(positive, server_lr, -server_lr)
    average = weighted_mean(updates, sizes)
    return RoundResult(base + rates * average, average, {
        "retained_clients": list(updates), "flipped_coordinates": int((~positive).sum()),
    })


def flame(base, updates, noise_multiplier=0.001, generator=None):
    checked_updates(base, updates)
    if len(updates) < 3:
        raise ValueError("FLAME requires at least three participants")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("FLAME noise multiplier must be finite and nonnegative")
    try:
        import hdbscan
    except ImportError as error:
        raise ImportError("FLAME requires hdbscan; install defense_bench/requirements.txt in a separate environment") from error
    client_ids = list(updates)
    models = torch.stack([base + updates[client_id] for client_id in client_ids]).double()
    norms = models.norm(dim=1)
    if (norms == 0).any():
        raise ValueError("FLAME cosine distance is undefined for a zero local model")
    models.div_(norms[:, None])
    distances = (1 - models @ models.T).clamp_(0, 2).cpu().numpy()
    np.fill_diagonal(distances, 0)
    del models
    labels = hdbscan.HDBSCAN(
        metric="precomputed", min_cluster_size=len(updates) // 2 + 1,
        min_samples=1, allow_single_cluster=True,
    ).fit_predict(distances)
    accepted = [client_id for client_id, label in zip(client_ids, labels) if label >= 0]
    if len(accepted) <= len(updates) // 2:
        raise RuntimeError("FLAME found no strict-majority cluster; refusing a silent FedAvg fallback")
    update_norms = torch.stack([update.norm() for update in updates.values()])
    bound = torch.quantile(update_norms, 0.5)
    average = torch.zeros_like(base)
    for client_id in accepted:
        norm = updates[client_id].norm()
        scale = min(1.0, float(bound / norm)) if norm > 0 else 1.0
        average.add_(updates[client_id], alpha=scale / len(accepted))
    noise = torch.randn(base.shape, dtype=base.dtype, device=base.device, generator=generator)
    result = base + average + noise * noise_multiplier * bound
    return RoundResult(result, average, {
        "retained_clients": accepted, "cluster_labels": labels.tolist(),
        "clip_bound": float(bound), "noise_std": float(noise_multiplier * bound),
    })


def crfl(base, updates, sizes, clip_norm, noise_std, final_round, generator=None):
    checked_updates(base, updates)
    if not math.isfinite(clip_norm) or clip_norm <= 0 or not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("CRFL requires a positive clipping norm and nonnegative noise standard deviation")
    average = weighted_mean(updates, sizes)
    result = base + average
    before = result.norm()
    result.mul_(min(1.0, clip_norm / float(before)) if before > 0 else 1.0)
    clipped_norm = float(result.norm())
    evaluation_model = result.clone()
    if not final_round:
        result.add_(torch.randn(result.shape, dtype=result.dtype, device=result.device, generator=generator), alpha=noise_std)
    return RoundResult(result, average, {
        "retained_clients": list(updates), "preclip_model_norm": float(before),
        "clipped_model_norm": clipped_norm, "clip_bound": clip_norm,
        "training_noise_std": 0.0 if final_round else noise_std,
    }, evaluation_model=evaluation_model)


def prodigal(local_models, initial_models, topk_masks, threshold):
    client_ids = list(local_models)
    if not client_ids or set(client_ids) != set(initial_models) or set(client_ids) != set(topk_masks):
        raise ValueError("Prodigal needs matching local models, personalized starting models, and gradient masks")
    if not 0 <= threshold < len(client_ids):
        raise ValueError("Prodigal uses votes > threshold; threshold must be below participant count")
    base = initial_models[client_ids[0]]
    checked_updates(base, local_models)
    checked_updates(base, initial_models)
    votes = torch.zeros_like(base, dtype=torch.int32)
    for client_id in client_ids:
        mask = topk_masks[client_id]
        if mask.dtype != torch.bool or mask.shape != base.shape:
            raise ValueError("Invalid Prodigal Top-K mask")
        votes.add_(mask)
    common = votes > threshold
    unit_sizes = dict.fromkeys(client_ids, 1)
    average = weighted_mean(local_models, unit_sizes)
    feedback = average - weighted_mean(initial_models, unit_sizes)
    shared = torch.where(common, average, 0)
    personal, outlier_counts = {}, {}
    for client_id in client_ids:
        specific = topk_masks[client_id] & ~common
        personal[client_id] = shared + torch.where(specific, local_models[client_id], 0)
        outlier_counts[client_id] = int(specific.sum())
    return RoundResult(shared, feedback, {
        "retained_clients": client_ids, "common_coordinates": int(common.sum()),
        "personal_coordinates": outlier_counts,
    }, personalized=personal)


def consensus_model(model, old_masks, threshold):
    if not old_masks or not 1 <= threshold <= len(old_masks):
        raise ValueError("Invalid Lockdown consensus threshold")
    result = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            votes = sum(mask[name].to(parameter.device) for mask in old_masks)
            parameter.mul_(votes >= threshold)
    return result


class SmoothedClassifier(torch.nn.Module):
    """Monte Carlo parameter-smoothing votes, NOT a robustness certificate."""

    def __init__(self, model, sigma, samples, seed):
        super().__init__()
        if not math.isfinite(sigma) or sigma < 0 or samples < 1:
            raise ValueError("Invalid smoothing configuration")
        self.model = copy.deepcopy(model).eval()
        self.base = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        self.sigma, self.samples, self.seed = sigma, samples, seed

    @torch.no_grad()
    def forward(self, inputs):
        generator = torch.Generator(device=inputs.device).manual_seed(self.seed)
        counts = None
        for draw in range(self.samples):
            for name, parameter in self.model.named_parameters():
                noise = torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=generator)
                parameter.copy_(self.base[name] + noise * self.sigma)
            logits = self.model(inputs)
            if counts is None:
                counts = torch.zeros_like(logits)
            counts.scatter_add_(1, logits.argmax(1, keepdim=True), torch.ones_like(logits[:, :1]))
        return counts / self.samples
