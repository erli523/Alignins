"""Isolated RAMP core: native detection/masking, adaptive non-detected mean-Z penalty.
Use src/ramp_zmean_eval.py; existing experiments are unchanged.
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


def zmean_penalties(z_scores, remaining_indices, strength=1.0):
    """Baseline uses ONLY native non-detected indices, never true client roles."""
    if remaining_indices.numel() == 0:
        raise ValueError("No non-detected clients to estimate Z baseline")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Invalid Z penalty strength")
    if not bool(torch.isfinite(z_scores).all()):
        raise ValueError("Nonfinite Z scores")
    baseline = z_scores[remaining_indices].mean()
    excess = torch.clamp(z_scores - baseline, min=0)
    penalties = torch.exp(-strength * excess)
    # No minimum-weight floor; baseline guarantees at least one unit penalty.
    return penalties, baseline, excess


class RAMPAGAlphaZero:
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
        self.alphaguard_params = dict(
            l2_filter_lower_percentile=args.l2_lower,
            l2_filter_upper_percentile=args.l2_upper,
            min_trusted_clients=args.min_reference,
            gref_strategy=args.reference,
            epsilon=args.epsilon,
            top_k_percentile=args.ag_topk_ratio,
            robust_alpha_exclude_ratio=0.15,
            normal_range_lower_ratio=0.9,
            normal_range_upper_ratio=1.1,
            penalty_strength=args.penalty_strength,
            min_weight_threshold=0.01,
            alpha_outlier_num=min(args.num_corrupt + 2, args.num_agents // 2),
        )

    @torch.no_grad()
    def build_reference(self, updates):
        """全部参与更新进入原生 AlphaGuard Stage 1；不按真实身份预筛选。"""
        ids = list(updates)
        matrix = torch.stack([updates[c].detach() for c in ids])
        if matrix.shape[1] != self.dimension or not bool(torch.isfinite(matrix).all()):
            raise ValueError("Invalid update dimension or non-finite values")
        ref, candidate = self._alphaguard_stage1(matrix)
        norms = matrix.norm(dim=1)
        lower = torch.quantile(norms, self.args.l2_lower / 100)
        upper = torch.quantile(norms, self.args.l2_upper / 100)
        middle_count = int(((norms >= lower) & (norms <= upper)).sum())
        info = dict(
            reference_scope="alphaguard_all_participants_l2",
            reference_pool_clients=ids,
            reference_clients=[ids[i] for i in candidate.cpu().tolist()],
            reference_strategy=self.args.reference,
            fallback=middle_count < self.args.min_reference,
            l2_low=float(lower),
            l2_high=float(upper),
            reference_abs_below_epsilon_fraction=float(
                (ref.abs() < self.args.epsilon).float().mean()
            ),
        )
        logging.info(
            "Reference scope=%s | pool=%s | reference=%s | method=%s | fallback=%s",
            info["reference_scope"],
            ids,
            info["reference_clients"],
            self.args.reference,
            info["fallback"],
        )
        return ref, norms, info

    def _compute_g_ref(self, updates, trusted_indices, strategy="median"):
        """
        计算参考向量 g_ref (支持多种策略)

        Args:
            updates: 所有客户端的更新 tensor, shape (num_clients, model_dim)
            trusted_indices: 被信任的客户端索引
            strategy: 计算策略 ("median", "geometric_median", "krum_select")

        Returns:
            g_ref: 参考向量, shape (model_dim,)
        """
        selected_updates = updates[trusted_indices]
        if strategy == "median":
            return torch.median(selected_updates, dim=0)[0]
        elif strategy == "geometric_median":
            logging.info("  Using geometric median for g_ref computation")
            from geom_median.torch import compute_geometric_median

            updates_list = [selected_updates[i].cpu() for i in range(len(selected_updates))]
            weights = torch.ones(len(updates_list))
            g_ref = compute_geometric_median(updates_list, weights).median
            return g_ref.to(self.args.device)
        elif strategy == "krum_select":
            logging.info("  Using Krum selection for g_ref computation")
            n = len(selected_updates)
            distances = torch.zeros(n, device=selected_updates.device)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        distances[i] += torch.norm(selected_updates[i] - selected_updates[j])
            best_idx = torch.argmin(distances)
            logging.info("  Krum selected client index (in trusted set): %d" % best_idx)
            return selected_updates[best_idx]
        else:
            logging.warning(f"  Unknown g_ref strategy '{strategy}', using median as fallback")
            return torch.median(selected_updates, dim=0)[0]

    def _alphaguard_stage1(self, updates):
        """Stage 1: L2 robust reference selection"""
        num_clients = len(updates)
        logging.info("------------ Stage 1: L2 Robust Reference Selection ------------")
        logging.info("Processing %d client updates" % num_clients)
        l2_norms = torch.norm(updates, dim=1)
        logging.info("L2 norms statistics:")
        logging.info("  Min:  %.4f" % l2_norms.min())
        logging.info("  Max:  %.4f" % l2_norms.max())
        logging.info("  Mean: %.4f" % l2_norms.mean())
        logging.info("  Std:  %.4f" % l2_norms.std())
        lower_threshold = torch.quantile(
            l2_norms, self.alphaguard_params["l2_filter_lower_percentile"] / 100
        )
        upper_threshold = torch.quantile(
            l2_norms, self.alphaguard_params["l2_filter_upper_percentile"] / 100
        )
        trusted_mask = (l2_norms >= lower_threshold) & (l2_norms <= upper_threshold)
        trusted_indices = torch.where(trusted_mask)[0]
        if len(trusted_indices) < self.alphaguard_params["min_trusted_clients"]:
            logging.info("  Warning: Not enough trusted clients, using all clients")
            trusted_indices = torch.arange(num_clients, device=updates.device)
        strategy = self.alphaguard_params.get("gref_strategy", "median")
        g_ref = self._compute_g_ref(updates, trusted_indices, strategy)
        logging.info("Stage 1 Results:")
        logging.info("  Trusted clients: %d/%d" % (len(trusted_indices), num_clients))
        logging.info("  L2 threshold: [%.4f, %.4f]" % (lower_threshold, upper_threshold))
        logging.info("  g_ref strategy: %s" % strategy)
        logging.info("  Reference norm: %.4f" % torch.norm(g_ref))
        return (g_ref, trusted_indices)

    def _alphaguard_stage2(self, updates, g_ref):
        """Stage 2: Alpha vector computation"""
        logging.info("------------ Stage 2: Alpha Vector Computation ------------")
        epsilon = self.args.epsilon
        g_ref_safe = torch.where(torch.abs(g_ref) < epsilon, torch.sign(g_ref) * epsilon, g_ref)
        alpha_vectors = updates / g_ref_safe
        alpha_vectors = torch.nan_to_num(alpha_vectors, nan=0.0, posinf=1.0, neginf=-1.0)
        alpha_norms = torch.norm(alpha_vectors, dim=1)
        logging.info("Alpha vectors statistics:")
        logging.info("  Shape:      %s" % str(alpha_vectors.shape))
        logging.info("  Norm min:   %.4f" % alpha_norms.min())
        logging.info("  Norm max:   %.4f" % alpha_norms.max())
        logging.info("  Norm mean:  %.4f" % alpha_norms.mean())
        logging.info("  Norm std:   %.4f" % alpha_norms.std())
        logging.info("  Epsilon:    %.9g" % epsilon)
        return alpha_vectors

    def _alphaguard_stage3(self, updates, alpha_vectors):
        """Stage 3: Robust detection and penalty"""
        logging.info("------------ Stage 3: Robust Detection and Penalty ------------")
        num_clients = len(alpha_vectors)
        model_dim = alpha_vectors.shape[1]
        logging.info("Processing %d clients with model dimension %d" % (num_clients, model_dim))
        logging.info(
            "Top-K percentage: %.1f%%" % (self.alphaguard_params["top_k_percentile"] * 100)
        )
        k = max(1, int(self.alphaguard_params["top_k_percentile"] * model_dim))
        abs_alpha = torch.abs(alpha_vectors)
        top_k_vals, _ = torch.topk(abs_alpha, k, dim=1)
        top_k_mean = torch.mean(top_k_vals, dim=1)
        top_k_mean = torch.nan_to_num(top_k_mean, nan=1.0, posinf=10.0, neginf=0.0)
        logging.info("Top-K mean statistics:")
        logging.info("  Min:  %.4f" % top_k_mean.min())
        logging.info("  Max:  %.4f" % top_k_mean.max())
        logging.info("  Mean: %.4f" % top_k_mean.mean())
        logging.info("  Std:  %.4f" % top_k_mean.std())
        if hasattr(self, "_current_client_ids"):
            logging.info("Top-K values by client:")
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
                logging.info("  Client %d%s: %.4f" % (client_id, mal_str, top_k_mean[i]))
        exclude_count = max(
            1, int(self.alphaguard_params["robust_alpha_exclude_ratio"] * num_clients)
        )
        if num_clients <= 2 * exclude_count:
            exclude_count = max(1, num_clients // 4)
        sorted_vals, _ = torch.sort(top_k_mean)
        if len(sorted_vals) > 2 * exclude_count:
            trimmed_vals = sorted_vals[exclude_count:-exclude_count]
        else:
            trimmed_vals = sorted_vals
        robust_reference = torch.mean(trimmed_vals)
        if torch.isnan(robust_reference) or torch.isinf(robust_reference) or robust_reference == 0:
            robust_reference = torch.median(top_k_mean)
            logging.info("  Using median as fallback reference: %.4f" % robust_reference)
        normal_lower = robust_reference * self.alphaguard_params["normal_range_lower_ratio"]
        normal_upper = robust_reference * self.alphaguard_params["normal_range_upper_ratio"]
        logging.info("Robust reference analysis:")
        logging.info("  Reference: %.4f" % robust_reference)
        logging.info("  Lower bound: %.4f" % normal_lower)
        logging.info("  Upper bound: %.4f" % normal_upper)
        deviation_scores = torch.zeros_like(top_k_mean)
        mask_lower = top_k_mean < normal_lower
        mask_upper = top_k_mean > normal_upper
        if normal_lower > 0:
            deviation_scores[mask_lower] = (normal_lower - top_k_mean[mask_lower]) / (
                normal_lower + 1e-06
            )
        if normal_upper > 0:
            deviation_scores[mask_upper] = (top_k_mean[mask_upper] - normal_upper) / (
                normal_upper + 1e-06
            )
        deviation_scores = torch.clamp(deviation_scores, 0, 10)
        logging.info("Deviation scores:")
        logging.info("  Min:  %.4f" % deviation_scores.min())
        logging.info("  Max:  %.4f" % deviation_scores.max())
        logging.info("  Mean: %.4f" % deviation_scores.mean())
        logging.info("  Clients below lower: %d" % mask_lower.sum())
        logging.info("  Clients above upper: %d" % mask_upper.sum())
        weights = torch.exp(-self.alphaguard_params["penalty_strength"] * deviation_scores)
        weights = torch.clamp(weights, min=self.alphaguard_params["min_weight_threshold"])
        logging.info("Legacy weight diagnostics (UNUSED; actual Z-mean weights follow):")
        logging.info("  Min weight:  %.4f" % weights.min())
        logging.info("  Max weight:  %.4f" % weights.max())
        logging.info("  Mean weight: %.4f" % weights.mean())
        logging.info("  Weight std:  %.4f" % weights.std())
        if hasattr(self, "_current_client_ids"):
            num_malicious_selected = sum(
                (1 for cid in self._current_client_ids if cid < self.args.num_corrupt)
            )
        else:
            num_malicious_selected = min(self.args.num_corrupt, num_clients)
        num_to_remove = min(num_malicious_selected + 1, num_clients - 1)
        logging.info("Removal strategy (based on actual malicious participants):")
        logging.info("  Clients selected this round: %d" % num_clients)
        logging.info("  Malicious clients selected: %d" % num_malicious_selected)
        logging.info(
            "  Num to remove: %d (malicious + 1, will keep at least %d)"
            % (num_to_remove, num_clients - num_to_remove)
        )
        median_topk = torch.median(top_k_mean)
        mad = torch.median(torch.abs(top_k_mean - median_topk))
        if mad < 1e-06:
            q75, q25 = (torch.quantile(top_k_mean, 0.75), torch.quantile(top_k_mean, 0.25))
            iqr = q75 - q25
            mad = iqr / 1.35
            if mad < 1e-06:
                mad = torch.std(top_k_mean)
        modified_z_scores = 0.6745 * torch.abs(top_k_mean - median_topk) / (mad + 1e-06)
        logging.info("Bidirectional Anomaly Detection (Top-K based):")
        logging.info("  Median Top-K: %.4f" % median_topk)
        logging.info("  MAD: %.4f" % mad)
        logging.info(
            "  Q25: %.4f, Q75: %.4f"
            % (torch.quantile(top_k_mean, 0.25), torch.quantile(top_k_mean, 0.75))
        )
        if hasattr(self, "_current_client_ids"):
            logging.info("Modified Z-scores by client:")
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
                deviation = top_k_mean[i] - median_topk
                direction = "HIGH" if deviation > 0 else "LOW"
                logging.info(
                    "  Client %d%s: Top-K=%.4f, Deviation=%.4f (%s), Z-score=%.4f"
                    % (
                        client_id,
                        mal_str,
                        top_k_mean[i],
                        abs(deviation),
                        direction,
                        modified_z_scores[i],
                    )
                )
        _, sorted_indices_by_zscore = torch.sort(modified_z_scores)
        remaining_indices = sorted_indices_by_zscore[:-num_to_remove]
        removed_indices = sorted_indices_by_zscore[-num_to_remove:]
        removed_topk_values = top_k_mean[removed_indices]
        removed_above_median = (removed_topk_values > median_topk).sum().item()
        removed_below_median = (removed_topk_values < median_topk).sum().item()
        logging.info("Client Selection (MAD-based Outlier Removal):")
        logging.info("  Total removed: %d" % num_to_remove)
        logging.info("  Removed indices: %s" % str(removed_indices.tolist()))
        if hasattr(self, "_current_client_ids"):
            removed_client_ids = [
                self._current_client_ids[idx.item()]
                if isinstance(idx, torch.Tensor)
                else self._current_client_ids[idx]
                for idx in removed_indices
            ]
            removed_client_ids_str = ", ".join(
                [
                    f"{cid}(MAL)" if cid < self.args.num_corrupt else f"{cid}"
                    for cid in removed_client_ids
                ]
            )
            logging.info("  Removed client IDs: [%s]" % removed_client_ids_str)
        logging.info(
            "  Removed Top-K means: %s" % str([round(x, 4) for x in removed_topk_values.tolist()])
        )
        logging.info(
            "  Removed Z-scores: %s"
            % str([round(x, 4) for x in modified_z_scores[removed_indices].tolist()])
        )
        logging.info(
            "  Above median: %d, Below median: %d" % (removed_above_median, removed_below_median)
        )
        logging.info("  Remaining clients: %d/%d" % (len(remaining_indices), num_clients))
        if hasattr(self, "_current_client_ids"):
            num_benign_selected = num_clients - num_malicious_selected
            removed_indices_list = (
                removed_indices.tolist()
                if isinstance(removed_indices, torch.Tensor)
                else list(removed_indices)
            )
            malicious_detected = sum(
                (
                    1
                    for idx in removed_indices_list
                    if self._current_client_ids[idx] < self.args.num_corrupt
                )
            )
            benign_removed = sum(
                (
                    1
                    for idx in removed_indices_list
                    if self._current_client_ids[idx] >= self.args.num_corrupt
                )
            )
            logging.info("Detection Analysis:")
            logging.info(
                "  Malicious selected: %d/%d (this round)" % (num_malicious_selected, num_clients)
            )
            logging.info(
                "  Benign selected: %d/%d (this round)" % (num_benign_selected, num_clients)
            )
            logging.info(
                "  Malicious detected: %d/%d" % (malicious_detected, num_malicious_selected)
            )
            logging.info("  Benign removed: %d/%d" % (benign_removed, num_benign_selected))
            if num_malicious_selected > 0:
                logging.info(
                    "  Detection rate: %.2f%%" % (malicious_detected / num_malicious_selected * 100)
                )
            else:
                logging.info("  Detection rate: N/A (no malicious clients selected)")
            if num_benign_selected > 0:
                logging.info(
                    "  False positive: %.2f%%" % (benign_removed / num_benign_selected * 100)
                )
            else:
                logging.info("  False positive: N/A (no benign clients selected)")
        else:
            malicious_detected = sum((1 for idx in removed_indices if idx < self.args.num_corrupt))
            benign_removed = sum((1 for idx in removed_indices if idx >= self.args.num_corrupt))
            num_benign = num_clients - self.args.num_corrupt
            logging.info("Detection Analysis (fallback):")
            logging.info("  Malicious detected: %d" % malicious_detected)
            logging.info("  Benign removed: %d" % benign_removed)
            if num_benign > 0:
                logging.info("  False positive: %.2f%%" % (benign_removed / num_benign * 100))
            else:
                logging.info("  False positive: N/A (no benign clients selected)")
        logging.info("Final Statistics:")
        logging.info("  Reference value: %.4f" % robust_reference)
        logging.info("  Normal range: [%.4f, %.4f]" % (normal_lower, normal_upper))
        logging.info("  Selected clients: %d/%d" % (len(remaining_indices), num_clients))
        logging.info(
            "  UNUSED legacy weight range: [%.4f, %.4f]"
            % (weights[remaining_indices].min(), weights[remaining_indices].max())
        )
        self.last_detection = dict(
            scores=top_k_mean.detach().clone(),
            z_scores=modified_z_scores.detach().clone(),
            median=median_topk.detach().clone(),
            mad=mad.detach().clone(),
            removed_indices=removed_indices.detach().clone(),
            remaining_indices=remaining_indices.detach().clone(),
            weights=weights.detach().clone(),
            num_to_remove=num_to_remove,
            k=k,
            num_malicious_selected=num_malicious_selected,
        )
        return (remaining_indices, weights)

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
            coordinate_ratio_mean=float(relative.mean()),
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
        self._current_client_ids = ids
        matrix = torch.stack([updates[c].detach() for c in ids])
        if len(ids) < 2:
            raise ValueError("Native AlphaGuard detection requires at least two participants")
        alpha = self._alphaguard_stage2(matrix, ref)
        self._alphaguard_stage3(matrix, alpha)
        detection = self.last_detection
        detected_ids = [ids[i] for i in detection["removed_indices"].cpu().tolist()]
        detected_set = set(detected_ids)
        logging.info(
            "Global %d | native AlphaGuard detected=%s | detection_topk_ratio=%g | detection_k=%d | zero_topk_ratio=%g | zero_k=%d | alpha_threshold=%g | client_exclusion=OFF",
            round_number,
            detected_ids,
            self.args.ag_topk_ratio,
            detection["k"],
            self.args.topk_ratio,
            self.k,
            self.args.alpha_threshold,
        )
        data_total = sum((data_sizes[c] for c in ids))
        penalties, z_base, z_excess = zmean_penalties(
            detection["z_scores"], detection["remaining_indices"], self.args.penalty_strength
        )
        penalties = penalties.to(device=matrix.device, dtype=matrix.dtype)
        baseline_ids = [ids[i] for i in detection["remaining_indices"].cpu().tolist()]
        logging.info("Global %d | Z-mean penalty | baseline_clients=%s | z_base=%.8g | lambda=%.8g | minimum_penalty_floor=NONE",
                     round_number, baseline_ids, float(z_base), self.args.penalty_strength)
        penalty_sum = penalties.sum()
        if (
            not bool(torch.isfinite(penalties).all())
            or bool((penalties < 0).any())
            or not bool(torch.isfinite(penalty_sum))
            or float(penalty_sum) <= 0
        ):
            raise ValueError("Invalid AlphaGuard penalty weights")
        normalized_weights = penalties / penalty_sum
        logging.info(
            "Global %d | aggregation=zmean_penalty_all_participants | penalty_sum=%.8g | normalized_weight_sum=%.8g",
            round_number,
            float(penalty_sum),
            float(normalized_weights.sum()),
        )
        applied, rows, layers, selected_total = ({}, [], [], 0)
        for position, cid in enumerate(ids):
            target = cid in detected_set
            selected = features[cid]["above_threshold"] if target else np.empty(0, dtype=np.int64)
            applied[cid] = self.zero_update(updates[cid], selected) if target else updates[cid]
            removed_l2 = float((updates[cid] - applied[cid]).norm())
            z = float(detection["z_scores"][position])
            row = dict(
                round=round_number,
                client_id=cid,
                role="malicious" if cid < self.args.num_corrupt else "benign",
                reference_candidate=cid in reference["reference_clients"],
                param_l2=float(norms[position]),
                relative_score=float(detection["scores"][position]),
                coordinate_ratio_mean=features[cid]["coordinate_ratio_mean"],
                detected_anomaly=target,
                detection_topk_count=detection["k"],
                z_score=z,
                z_base=float(z_base),
                z_excess=float(z_excess[position]),
                baseline_member=cid in baseline_ids,
                topk_count=self.k,
                threshold_count=features[cid]["threshold_count"],
                above_threshold_count=len(features[cid]["above_threshold"]),
                selected_count=len(selected),
                data_weight=data_sizes[cid] / data_total,
                penalty_weight=float(penalties[position]),
                aggregation_weight=float(normalized_weights[position]),
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
                "Global %d | Client %d %-9s | L2=%.6g | score=%.6g | Z=%.6g | reference=%s | detected=%s | detection_topk=%d | zero_topk=%d | above_threshold_in_topk=%d | zeroed=%d | removed_L2=%.6g | penalty=%.6g | weight=%.6f | %s",
                round_number,
                cid,
                row["role"],
                row["param_l2"],
                row["relative_score"],
                z,
                row["reference_candidate"],
                target,
                detection["k"],
                self.k,
                row["above_threshold_count"],
                len(selected),
                removed_l2,
                row["penalty_weight"],
                row["aggregation_weight"],
                row["decision"],
            )
        delta = torch.zeros_like(applied[ids[0]])
        for position, cid in enumerate(ids):
            delta += normalized_weights[position] * applied[cid]
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Non-finite aggregate")
        before = model_vector(model).clone()
        utils.vector_to_model((before + self.args.server_lr * delta).float(), model)
        return dict(
            z_base=float(z_base),
            baseline_clients=baseline_ids,
            rows=rows,
            layers=layers,
            features=features,
            reference=reference,
            reference_vector=ref.detach().clone(),
            detected_clients=detected_ids,
            detection_scores=detection["scores"].cpu(),
            detection_z_scores=detection["z_scores"].cpu(),
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
                "coordinate_ratio_mean",
                "detected_anomaly",
                "detection_topk_count",
                "z_score",
                "topk_count",
                "threshold_count",
                "above_threshold_count",
                "selected_count",
                "data_weight",
                "penalty_weight",
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
        "# RAMP AG Alpha-Zero",
        "",
        f"状态：{metadata['status']}；完成 {len(rows)} 轮；阈值 {threshold}。",
        "",
        "规则：按 abs(local_delta)/(abs(reference)+epsilon) 从大到小取前 K，再选择比值严格大于阈值的坐标，将其本地 delta 置零。无差集、无配对正常训练。",
        f"检测 Top-K 比例：{metadata['settings']['ag_topk_ratio']}；置零 Top-K 比例：{metadata['settings']['topk_ratio']}；K=max(1, floor(模型维度×比例))；不足阈值不补足。",
        "参考：原生 AlphaGuard Stage 1，全部参与更新按 L2 取中间区间，再取 median（默认）；不足最少参考人数时回退全部参与者。可能包含恶意客户端。",
        "原 AlphaGuard 的 Z 排序决定异常客户端名单；异常客户端执行坐标置零，全部参与客户端使用原生惩罚权重并统一归一化，不再按数据量加权，不整客户端剔除。",
        "score/Z 由原生 AlphaGuard Alpha 与独立检测 Top-K 计算；默认 ag_topk_ratio=0.001。topk_ratio 仅控制后续置零，coordinate_ratio_mean 为后续比值全坐标均值。",
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
        fig.suptitle("RAMP AG Alpha-Zero: relative magnitude threshold masking")
        fig.savefig(output / "figures/training.png", dpi=170)
        plt.close(fig)
        lines += ["![Training](figures/training.png)", ""]
    lines += [
        "本次独立保存 console.log、run.json、逐轮/客户端/层 CSV、报告/图片、最终模型和源码。默认最后一轮保存更新与选中坐标。",
        "修改 RAMPAGAlphaZero.build_reference / select_coordinates / zero_update / aggregate 可调整完整方案。",
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
    parser.add_argument(
        "--ag_topk_ratio",
        "--ag-topk-ratio",
        "--ag_topk",
        type=float,
        default=0.001,
        help="原 AlphaGuard 检测用 Top-K 比例，独立于坐标置零比例",
    )
    parser.add_argument("--topk_ratio", "--topk-ratio", type=float, default=0.001)
    parser.add_argument("--alpha_threshold", "--threshold", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument(
        "--reference", choices=["median", "geometric_median", "krum_select"], default="median"
    )
    parser.add_argument("--l2_lower", type=float, default=25.0)
    parser.add_argument("--l2_upper", type=float, default=75.0)
    parser.add_argument("--min_reference", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, choices=[0], default=0)
    parser.add_argument("--penalty_strength", "--penalty-strength", type=float, default=4.0)
    parser.add_argument("--name", default="ag_penalty_topk0p1pct_gt0p8_v3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot_rounds", "--snapshot-rounds", type=int, nargs="*", default=[])
    parser.add_argument("--train_limit", "--train-limit", type=int, default=0, help="验证专用；0 为完整数据")
    parser.add_argument("--test_limit", "--test-limit", type=int, default=0, help="验证专用；0 为完整测试集")
    args = parser.parse_args(argv)
    if (
        (not math.isfinite(args.penalty_strength) or args.penalty_strength < 0)
        or args.rounds < 1
        or args.local_ep < 1
        or (not 0 <= args.num_corrupt < args.num_agents)
        or (args.bs < 1)
        or (not 0 < args.agent_frac <= 1)
        or (math.floor(args.agent_frac * args.num_agents) < 2)
        or (not 0 <= args.poison_frac <= 1)
        or (not math.isfinite(args.ag_topk_ratio) or not 0 < args.ag_topk_ratio <= 1)
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
    args.zero_scope = "alphaguard_detected"
    args.aggr, args.num_target, args.cease_poison = ("avg", 10, 100000)
    return args


def output_path(args):
    split = f"noniid_beta{args.beta:g}" if args.non_iid else "iid"
    return (
        args.output
        if args.output is not None
        else ROOT
        / "research_results/RAMP_ag_alpha_zero"
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
        reference_scope="alphaguard_all_participants_l2",
        implementation="self-contained src/ramp_ag_alpha_zero.py",
        formula="relative Top-K intersect {abs(delta)/(abs(reference)+epsilon) > alpha_threshold}",
        zero_rule="selected local update coordinates set to zero; native penalty weights normalized over all participants",
        alphaguard_source_sha256="633b0213f3b5875ebf7bf420747b3498d0c944dde68a763314d0061fee79c45a",
        reference_implementation="Original AlphaGuard reference, stage2 Alpha and stage3 detection copied; added diagnostic capture only",
        coordinate_rule="abs(delta)/(abs(reference)+epsilon), not original AlphaGuard Stage 2 signed denominator",
        client_filtering="No whole-client exclusion. Intervention targets are native AlphaGuard detected anomalies.",
        aggregation_rule="sum_i p_i*masked_delta_i/sum_i p_i across all participants; p_i is native pre-mask penalty; no data-size factor",
        implementation_version="native_detection_penalty_all_v3",
        detection_score_rule="Native signed Alpha stage2 -> abs Top-ag_topk_ratio mean -> native torch median/MAD Z and ranked removal set",
        detection_count_rule="Preserved native rule: min(actual malicious participants+1, participants-1); native detector uses truth to set count, not to pick individual targets",
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
        logging.info("Standalone RAMP AG Alpha-Zero | output=%s | settings=%s", output, args)
        logging.info(
            "Native AlphaGuard detection selects intervention targets; no whole-client exclusion; all participants use native penalty weights normalized over all participants"
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
        method = RAMPAGAlphaZero(args, model)
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
                            c: updates[c].detach().cpu() for c in ids if c < args.num_corrupt
                        },
                        intervention_updates={
                            c: updates[c].detach().cpu() for c in details["detected_clients"]
                        },
                        zero_scope=args.zero_scope,
                        reference=details["reference"],
                        detected_clients=details["detected_clients"],
                        detection_scores=details["detection_scores"],
                        detection_z_scores=details["detection_z_scores"],
                        reference_vector=details["reference_vector"].cpu(),
                        layout=method.layout,
                        indices={
                            str(c): {
                                key: torch.as_tensor(details["features"][c][key].copy())
                                for key in ("relative_topk", "above_threshold")
                            }
                            for c in ids
                            if c in details["detected_clients"]
                        },
                        client_weights={
                            r["client_id"]: r["aggregation_weight"] for r in details["rows"]
                        },
                        penalty_weights={
                            r["client_id"]: r["penalty_weight"] for r in details["rows"]
                        },
                        data_weights={c: sizes[c] / sum((sizes[i] for i in ids)) for c in ids},
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
            f"RAMP AG Alpha-Zero status={metadata['status']}; report={output / 'report.md'}",
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

