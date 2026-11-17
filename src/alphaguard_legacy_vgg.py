"""AlphaGuard methods restored from the 2025-11-08 12:12 experiment backup.
Used only by the isolated VGG9 batch; existing defenses are unchanged.
"""
import torch
import logging
from aggregation import Aggregation

class LegacyAlphaGuard(Aggregation):

    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.alphaguard_params = {'epsilon': 0.001, 'l2_filter_lower_percentile': 25, 'l2_filter_upper_percentile': 75, 'min_trusted_clients': 8, 'top_k_percentile': 0.001, 'robust_alpha_exclude_ratio': 0.15, 'normal_range_lower_ratio': 0.9, 'normal_range_upper_ratio': 1.1, 'penalty_strength': 2.0, 'min_weight_threshold': 0.01, 'alpha_outlier_num': min(args.num_corrupt + 2, args.num_agents // 2)}
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []

    def agg_alphaguard(self, agent_updates_dict):
        """AlphaGuard defense method implementation"""
        logging.info('------------ AlphaGuard Defense ------------')
        logging.info('Configuration:')
        logging.info('  Total clients: %d' % len(agent_updates_dict))
        logging.info('  Malicious clients: %d' % self.args.num_corrupt)
        logging.info('  Poison fraction: %.2f' % self.args.poison_frac)
        updates_list = []
        client_ids = []
        for _id, update in agent_updates_dict.items():
            updates_list.append(update)
            client_ids.append(_id)
        updates_tensor = torch.stack(updates_list, dim=0)
        self._current_client_ids = client_ids
        g_ref, trusted_indices = self._alphaguard_stage1(updates_tensor)
        alpha_vectors = self._alphaguard_stage2(updates_tensor, g_ref)
        remaining_indices, weights = self._alphaguard_stage3(updates_tensor, alpha_vectors)
        selected_updates = updates_tensor[remaining_indices]
        selected_weights = weights[remaining_indices].to(self.args.device)
        selected_weights = selected_weights / selected_weights.sum()
        selected_weights = selected_weights.view(-1, 1)
        aggregated_update = torch.sum(selected_updates * selected_weights, dim=0)
        logging.info('Final aggregation:')
        logging.info('  Selected clients: %d/%d' % (len(remaining_indices), len(updates_tensor)))
        logging.info('  Weight sum: %.4f' % selected_weights.sum())
        return aggregated_update

    def _alphaguard_stage1(self, updates):
        """Stage 1: L2 robust reference selection"""
        num_clients = len(updates)
        logging.info('------------ Stage 1: L2 Robust Reference Selection ------------')
        logging.info('Processing %d client updates' % num_clients)
        l2_norms = torch.norm(updates, dim=1)
        logging.info('L2 norms statistics:')
        logging.info('  Min:  %.4f' % l2_norms.min())
        logging.info('  Max:  %.4f' % l2_norms.max())
        logging.info('  Mean: %.4f' % l2_norms.mean())
        logging.info('  Std:  %.4f' % l2_norms.std())
        lower_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_lower_percentile'] / 100)
        upper_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_upper_percentile'] / 100)
        trusted_mask = (l2_norms >= lower_threshold) & (l2_norms <= upper_threshold)
        trusted_indices = torch.where(trusted_mask)[0]
        if len(trusted_indices) < self.alphaguard_params['min_trusted_clients']:
            logging.info('  Warning: Not enough trusted clients, using all clients')
            trusted_indices = torch.arange(num_clients, device=updates.device)
        g_ref = torch.median(updates[trusted_indices], dim=0)[0]
        logging.info('Stage 1 Results:')
        logging.info('  Trusted clients: %d/%d' % (len(trusted_indices), num_clients))
        logging.info('  L2 threshold: [%.4f, %.4f]' % (lower_threshold, upper_threshold))
        logging.info('  Reference norm: %.4f' % torch.norm(g_ref))
        return (g_ref, trusted_indices)

    def _alphaguard_stage2(self, updates, g_ref):
        """Stage 2: Alpha vector computation"""
        logging.info('------------ Stage 2: Alpha Vector Computation ------------')
        epsilon = 0.001
        g_ref_safe = torch.where(torch.abs(g_ref) < epsilon, torch.sign(g_ref) * epsilon, g_ref)
        alpha_vectors = updates / g_ref_safe
        alpha_vectors = torch.nan_to_num(alpha_vectors, nan=0.0, posinf=1.0, neginf=-1.0)
        alpha_norms = torch.norm(alpha_vectors, dim=1)
        logging.info('Alpha vectors statistics:')
        logging.info('  Shape:      %s' % str(alpha_vectors.shape))
        logging.info('  Norm min:   %.4f' % alpha_norms.min())
        logging.info('  Norm max:   %.4f' % alpha_norms.max())
        logging.info('  Norm mean:  %.4f' % alpha_norms.mean())
        logging.info('  Norm std:   %.4f' % alpha_norms.std())
        logging.info('  Epsilon:    %.6f' % epsilon)
        return alpha_vectors

    def _alphaguard_stage3(self, updates, alpha_vectors):
        """Stage 3: Robust detection and penalty"""
        logging.info('------------ Stage 3: Robust Detection and Penalty ------------')
        num_clients = len(alpha_vectors)
        model_dim = alpha_vectors.shape[1]
        logging.info('Processing %d clients with model dimension %d' % (num_clients, model_dim))
        logging.info('Top-K percentage: %.1f%%' % (self.alphaguard_params['top_k_percentile'] * 100))
        k = max(1, int(self.alphaguard_params['top_k_percentile'] * model_dim))
        abs_alpha = torch.abs(alpha_vectors)
        top_k_vals, _ = torch.topk(abs_alpha, k, dim=1)
        top_k_mean = torch.mean(top_k_vals, dim=1)
        top_k_mean = torch.nan_to_num(top_k_mean, nan=1.0, posinf=10.0, neginf=0.0)
        logging.info('Top-K mean statistics:')
        logging.info('  Min:  %.4f' % top_k_mean.min())
        logging.info('  Max:  %.4f' % top_k_mean.max())
        logging.info('  Mean: %.4f' % top_k_mean.mean())
        logging.info('  Std:  %.4f' % top_k_mean.std())
        if hasattr(self, '_current_client_ids'):
            logging.info('Top-K values by client:')
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = ' (MAL)' if client_id < self.args.num_corrupt else ''
                logging.info('  Client %d%s: %.4f' % (client_id, mal_str, top_k_mean[i]))
        exclude_count = max(1, int(self.alphaguard_params['robust_alpha_exclude_ratio'] * num_clients))
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
            logging.info('  Using median as fallback reference: %.4f' % robust_reference)
        normal_lower = robust_reference * self.alphaguard_params['normal_range_lower_ratio']
        normal_upper = robust_reference * self.alphaguard_params['normal_range_upper_ratio']
        logging.info('Robust reference analysis:')
        logging.info('  Reference: %.4f' % robust_reference)
        logging.info('  Lower bound: %.4f' % normal_lower)
        logging.info('  Upper bound: %.4f' % normal_upper)
        deviation_scores = torch.zeros_like(top_k_mean)
        mask_lower = top_k_mean < normal_lower
        mask_upper = top_k_mean > normal_upper
        if normal_lower > 0:
            deviation_scores[mask_lower] = (normal_lower - top_k_mean[mask_lower]) / (normal_lower + 1e-06)
        if normal_upper > 0:
            deviation_scores[mask_upper] = (top_k_mean[mask_upper] - normal_upper) / (normal_upper + 1e-06)
        deviation_scores = torch.clamp(deviation_scores, 0, 10)
        logging.info('Deviation scores:')
        logging.info('  Min:  %.4f' % deviation_scores.min())
        logging.info('  Max:  %.4f' % deviation_scores.max())
        logging.info('  Mean: %.4f' % deviation_scores.mean())
        logging.info('  Clients below lower: %d' % mask_lower.sum())
        logging.info('  Clients above upper: %d' % mask_upper.sum())
        weights = torch.exp(-self.alphaguard_params['penalty_strength'] * deviation_scores)
        weights = torch.clamp(weights, min=self.alphaguard_params['min_weight_threshold'])
        logging.info('Weight Distribution:')
        logging.info('  Min weight:  %.4f' % weights.min())
        logging.info('  Max weight:  %.4f' % weights.max())
        logging.info('  Mean weight: %.4f' % weights.mean())
        logging.info('  Weight std:  %.4f' % weights.std())
        num_to_remove = min(self.alphaguard_params['alpha_outlier_num'], num_clients - 1)
        median_topk = torch.median(top_k_mean)
        mad = torch.median(torch.abs(top_k_mean - median_topk))
        if mad < 1e-06:
            q75, q25 = (torch.quantile(top_k_mean, 0.75), torch.quantile(top_k_mean, 0.25))
            iqr = q75 - q25
            mad = iqr / 1.35
            if mad < 1e-06:
                mad = torch.std(top_k_mean)
        modified_z_scores = 0.6745 * torch.abs(top_k_mean - median_topk) / (mad + 1e-06)
        logging.info('Bidirectional Anomaly Detection (Top-K based):')
        logging.info('  Median Top-K: %.4f' % median_topk)
        logging.info('  MAD: %.4f' % mad)
        logging.info('  Q25: %.4f, Q75: %.4f' % (torch.quantile(top_k_mean, 0.25), torch.quantile(top_k_mean, 0.75)))
        if hasattr(self, '_current_client_ids'):
            logging.info('Modified Z-scores by client:')
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = ' (MAL)' if client_id < self.args.num_corrupt else ''
                deviation = top_k_mean[i] - median_topk
                direction = 'HIGH' if deviation > 0 else 'LOW'
                logging.info('  Client %d%s: Top-K=%.4f, Deviation=%.4f (%s), Z-score=%.4f' % (client_id, mal_str, top_k_mean[i], abs(deviation), direction, modified_z_scores[i]))
        _, sorted_indices_by_zscore = torch.sort(modified_z_scores)
        remaining_indices = sorted_indices_by_zscore[:-num_to_remove]
        removed_indices = sorted_indices_by_zscore[-num_to_remove:]
        removed_topk_values = top_k_mean[removed_indices]
        removed_above_median = (removed_topk_values > median_topk).sum().item()
        removed_below_median = (removed_topk_values < median_topk).sum().item()
        logging.info('Client Selection (MAD-based Outlier Removal):')
        logging.info('  Total removed: %d' % num_to_remove)
        logging.info('  Removed indices: %s' % str(removed_indices.tolist()))
        logging.info('  Removed Top-K means: %s' % str([round(x, 4) for x in removed_topk_values.tolist()]))
        logging.info('  Removed Z-scores: %s' % str([round(x, 4) for x in modified_z_scores[removed_indices].tolist()]))
        logging.info('  Above median: %d, Below median: %d' % (removed_above_median, removed_below_median))
        logging.info('  Remaining clients: %d/%d' % (len(remaining_indices), num_clients))
        malicious_detected = sum((1 for idx in removed_indices if idx < self.args.num_corrupt))
        benign_removed = sum((1 for idx in removed_indices if idx >= self.args.num_corrupt))
        logging.info('Detection Analysis:')
        logging.info('  Malicious detected: %d/%d' % (malicious_detected, self.args.num_corrupt))
        logging.info('  Benign removed: %d/%d' % (benign_removed, num_clients - self.args.num_corrupt))
        if self.args.num_corrupt > 0:
            logging.info('  Detection rate: %.2f%%' % (malicious_detected / self.args.num_corrupt * 100))
            logging.info('  False positive: %.2f%%' % (benign_removed / (num_clients - self.args.num_corrupt) * 100))
        logging.info('Final Statistics:')
        logging.info('  Reference value: %.4f' % robust_reference)
        logging.info('  Normal range: [%.4f, %.4f]' % (normal_lower, normal_upper))
        logging.info('  Selected clients: %d/%d' % (len(remaining_indices), num_clients))
        logging.info('  Weight range: [%.4f, %.4f]' % (weights[remaining_indices].min(), weights[remaining_indices].max()))
        return (remaining_indices, weights)
