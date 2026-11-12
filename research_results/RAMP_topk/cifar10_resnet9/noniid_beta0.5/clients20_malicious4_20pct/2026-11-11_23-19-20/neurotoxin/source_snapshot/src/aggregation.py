import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import torch.nn as nn 



class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        # AlphaGuard parameters (optimized for this project)
        self.alphaguard_params = {
            "epsilon": 1e-3,                      # Numerical stability
            "l2_filter_lower_percentile": getattr(args, 'ag_l2_lower', 25),     # Filter bottom (default: 25%)
            "l2_filter_upper_percentile": getattr(args, 'ag_l2_upper', 75),     # Filter top (default: 75%)
            "min_trusted_clients": 8,             # At least 8 clients (40% of 20)
            "top_k_percentile": getattr(args, 'ag_topk', 0.001),             # Top 1% of parameters (more focused)
            "robust_alpha_exclude_ratio": 0.15,   # Exclude extreme 15%
            "normal_range_lower_ratio": 0.9,      # Tighter lower bound (was 0.3)
            "normal_range_upper_ratio": 1.1,      # Tighter upper bound (was 3.0)
            "penalty_strength": 2.0,              # Stronger penalty (was 1.5)
            "min_weight_threshold": 0.01,         # Lower minimum weight (was 0.05)
            "alpha_outlier_num": min(args.num_corrupt + 2, args.num_agents // 2),  # Adaptive removal
            "gref_strategy": getattr(args, 'ag_gref_strategy', 'median'),  # Reference vector strategy (default: median)
        }
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):


        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)

        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == "alphaguard":
            aggregated_updates = self.agg_alphaguard(agent_updates_dict)
        elif self.args.aggr == "mars":
            aggregated_updates = self.agg_mars(agent_updates_dict, global_model)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        # 修复: weights 需要在 CPU 上以便 geom_median 库可以转换为 numpy
        weights = torch.ones(n)  
        # 将 local_updates 移到 CPU 进行几何中位数计算
        local_updates_cpu = [update.cpu() for update in local_updates]
        gw = compute_geometric_median(local_updates_cpu, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates.cpu()-gw, dim=1)))
            gw = compute_geometric_median(local_updates_cpu, weights).median

        # 将结果移回原设备
        aggregated_model = gw.to(self.args.device)
        return aggregated_model
    

    def agg_alphaguard(self, agent_updates_dict):
        """AlphaGuard defense method implementation"""
        logging.info("------------ AlphaGuard Defense ------------")
        logging.info("Configuration:")
        logging.info("  Total clients: %d" % len(agent_updates_dict))
        logging.info("  Malicious clients: %d" % self.args.num_corrupt)
        logging.info("  Poison fraction: %.2f" % self.args.poison_frac)
        
        # Convert updates to tensor format and keep track of client IDs
        updates_list = []
        client_ids = []
        for _id, update in agent_updates_dict.items():
            updates_list.append(update)
            client_ids.append(_id)
        updates_tensor = torch.stack(updates_list, dim=0)
        
        # Store client_ids for later use
        self._current_client_ids = client_ids
        
        # Log the mapping between array indices and client IDs for debugging
        logging.info("Client ID mapping (array_index -> client_id):")
        for idx, cid in enumerate(client_ids):
            mal_str = " (MAL)" if cid < self.args.num_corrupt else ""
            logging.info("  Index %d -> Client %d%s" % (idx, cid, mal_str))
        
        # Stage 1: L2 robust reference selection
        g_ref, trusted_indices = self._alphaguard_stage1(updates_tensor)
        
        # Stage 2: Alpha vector computation
        alpha_vectors = self._alphaguard_stage2(updates_tensor, g_ref)
        
        # Stage 3: Robust detection and penalty
        remaining_indices, weights = self._alphaguard_stage3(updates_tensor, alpha_vectors)
        
        # Final aggregation with weighted average
        selected_updates = updates_tensor[remaining_indices]
        selected_weights = weights[remaining_indices].to(self.args.device)
        
        # Normalize weights
        selected_weights = selected_weights / selected_weights.sum()
        selected_weights = selected_weights.view(-1, 1)
        
        aggregated_update = torch.sum(selected_updates * selected_weights, dim=0)
        
        # Convert remaining_indices to list for logging
        remaining_indices_list = remaining_indices.tolist() if isinstance(remaining_indices, torch.Tensor) else list(remaining_indices)
        
        # Log remaining client IDs
        if hasattr(self, '_current_client_ids'):
            remaining_client_ids = [self._current_client_ids[idx] for idx in remaining_indices_list]
            remaining_client_ids_str = ", ".join([f"{cid}(MAL)" if cid < self.args.num_corrupt else f"{cid}" 
                                                 for cid in remaining_client_ids])
            logging.info("Final aggregation:")
            logging.info("  Remaining array indices: %s" % str(remaining_indices_list))
            logging.info("  Remaining client IDs: [%s]" % remaining_client_ids_str)
            logging.info("  Selected clients: %d/%d" % (len(remaining_indices), len(updates_tensor)))
        else:
            logging.info("Final aggregation:")
            logging.info("  Selected clients: %d/%d" % (len(remaining_indices), len(updates_tensor)))
        logging.info("  Weight sum: %.4f" % selected_weights.sum())
        
        return aggregated_update

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
            # 中位数 - 对异常值鲁棒，计算高效（默认）
            return torch.median(selected_updates, dim=0)[0]
        
        elif strategy == "geometric_median":
            # 几何中位数 - 高维空间的鲁棒统计量
            logging.info("  Using geometric median for g_ref computation")
            from geom_median.torch import compute_geometric_median
            # 将数据移到 CPU 进行计算（geom_median 库要求）
            updates_list = [selected_updates[i].cpu() for i in range(len(selected_updates))]
            weights = torch.ones(len(updates_list))
            g_ref = compute_geometric_median(updates_list, weights).median
            return g_ref.to(self.args.device)
        
        elif strategy == "krum_select":
            # Krum 选择 - 选择距离其他更新最近的单个更新
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
            # 默认使用中位数（向后兼容 + 安全fallback）
            logging.warning(f"  Unknown g_ref strategy '{strategy}', using median as fallback")
            return torch.median(selected_updates, dim=0)[0]

    def _alphaguard_stage1(self, updates):
        """Stage 1: L2 robust reference selection"""
        num_clients = len(updates)
        
        logging.info("------------ Stage 1: L2 Robust Reference Selection ------------")
        logging.info("Processing %d client updates" % num_clients)
        
        # Compute L2 norms
        l2_norms = torch.norm(updates, dim=1)
        
        logging.info("L2 norms statistics:")
        logging.info("  Min:  %.4f" % l2_norms.min())
        logging.info("  Max:  %.4f" % l2_norms.max())
        logging.info("  Mean: %.4f" % l2_norms.mean())
        logging.info("  Std:  %.4f" % l2_norms.std())
        
        # Calculate percentile thresholds
        lower_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_lower_percentile']/100)
        upper_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_upper_percentile']/100)
        
        # Select trusted clients
        trusted_mask = (l2_norms >= lower_threshold) & (l2_norms <= upper_threshold)
        trusted_indices = torch.where(trusted_mask)[0]
        
        # Safety check
        if len(trusted_indices) < self.alphaguard_params['min_trusted_clients']:
            logging.info("  Warning: Not enough trusted clients, using all clients")
            trusted_indices = torch.arange(num_clients, device=updates.device)
        
        # Compute reference using specified strategy (默认 median)
        strategy = self.alphaguard_params.get('gref_strategy', 'median')
        g_ref = self._compute_g_ref(updates, trusted_indices, strategy)
        
        logging.info("Stage 1 Results:")
        logging.info("  Trusted clients: %d/%d" % (len(trusted_indices), num_clients))
        logging.info("  L2 threshold: [%.4f, %.4f]" % (lower_threshold, upper_threshold))
        logging.info("  g_ref strategy: %s" % strategy)
        logging.info("  Reference norm: %.4f" % torch.norm(g_ref))
        
        return g_ref, trusted_indices

    def _alphaguard_stage2(self, updates, g_ref):
        """Stage 2: Alpha vector computation"""
        logging.info("------------ Stage 2: Alpha Vector Computation ------------")
        
        # Numerical stability - use larger epsilon
        epsilon = 1e-3  # Increase epsilon for better stability
        g_ref_safe = torch.where(torch.abs(g_ref) < epsilon,
                               torch.sign(g_ref) * epsilon,
                               g_ref)
        
        # Compute alpha vectors
        alpha_vectors = updates / g_ref_safe
        
        # Replace NaN and Inf values
        alpha_vectors = torch.nan_to_num(alpha_vectors, nan=0.0, posinf=1.0, neginf=-1.0)
        
        alpha_norms = torch.norm(alpha_vectors, dim=1)
        logging.info("Alpha vectors statistics:")
        logging.info("  Shape:      %s" % str(alpha_vectors.shape))
        logging.info("  Norm min:   %.4f" % alpha_norms.min())
        logging.info("  Norm max:   %.4f" % alpha_norms.max())
        logging.info("  Norm mean:  %.4f" % alpha_norms.mean())
        logging.info("  Norm std:   %.4f" % alpha_norms.std())
        logging.info("  Epsilon:    %.6f" % epsilon)
        
        return alpha_vectors

    def _alphaguard_stage3(self, updates, alpha_vectors):
        """Stage 3: Robust detection and penalty"""
        logging.info("------------ Stage 3: Robust Detection and Penalty ------------")
        
        num_clients = len(alpha_vectors)
        model_dim = alpha_vectors.shape[1]
        
        logging.info("Processing %d clients with model dimension %d" % (num_clients, model_dim))
        logging.info("Top-K percentage: %.1f%%" % (self.alphaguard_params['top_k_percentile']*100))
        
        # Compute Top-K values
        k = max(1, int(self.alphaguard_params['top_k_percentile'] * model_dim))
        abs_alpha = torch.abs(alpha_vectors)
        top_k_vals, _ = torch.topk(abs_alpha, k, dim=1)
        top_k_mean = torch.mean(top_k_vals, dim=1)
        
        # Handle NaN/Inf in top_k_mean
        top_k_mean = torch.nan_to_num(top_k_mean, nan=1.0, posinf=10.0, neginf=0.0)
        
        logging.info("Top-K mean statistics:")
        logging.info("  Min:  %.4f" % top_k_mean.min())
        logging.info("  Max:  %.4f" % top_k_mean.max())
        logging.info("  Mean: %.4f" % top_k_mean.mean())
        logging.info("  Std:  %.4f" % top_k_mean.std())
        
        # Show top-k values for each client
        if hasattr(self, '_current_client_ids'):
            logging.info("Top-K values by client:")
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
                logging.info("  Client %d%s: %.4f" % (client_id, mal_str, top_k_mean[i]))
        
        # Compute robust reference (V19 innovation)
        exclude_count = max(1, int(self.alphaguard_params['robust_alpha_exclude_ratio'] * num_clients))
        
        # Safety check for small number of clients
        if num_clients <= 2 * exclude_count:
            exclude_count = max(1, num_clients // 4)
        
        sorted_vals, _ = torch.sort(top_k_mean)
        
        if len(sorted_vals) > 2 * exclude_count:
            trimmed_vals = sorted_vals[exclude_count:-exclude_count]
        else:
            trimmed_vals = sorted_vals
        
        robust_reference = torch.mean(trimmed_vals)
        
        # Safety check for robust reference
        if torch.isnan(robust_reference) or torch.isinf(robust_reference) or robust_reference == 0:
            robust_reference = torch.median(top_k_mean)
            logging.info("  Using median as fallback reference: %.4f" % robust_reference)
        
        # Compute bidirectional penalty
        normal_lower = robust_reference * self.alphaguard_params['normal_range_lower_ratio']
        normal_upper = robust_reference * self.alphaguard_params['normal_range_upper_ratio']
        
        logging.info("Robust reference analysis:")
        logging.info("  Reference: %.4f" % robust_reference)
        logging.info("  Lower bound: %.4f" % normal_lower)
        logging.info("  Upper bound: %.4f" % normal_upper)
        
        deviation_scores = torch.zeros_like(top_k_mean)
        mask_lower = top_k_mean < normal_lower
        mask_upper = top_k_mean > normal_upper
        
        # Compute deviation with safety checks
        if normal_lower > 0:
            deviation_scores[mask_lower] = (normal_lower - top_k_mean[mask_lower]) / (normal_lower + 1e-6)
        if normal_upper > 0:
            deviation_scores[mask_upper] = (top_k_mean[mask_upper] - normal_upper) / (normal_upper + 1e-6)
        
        # Clamp deviation scores to reasonable range
        deviation_scores = torch.clamp(deviation_scores, 0, 10)
        
        logging.info("Deviation scores:")
        logging.info("  Min:  %.4f" % deviation_scores.min())
        logging.info("  Max:  %.4f" % deviation_scores.max())
        logging.info("  Mean: %.4f" % deviation_scores.mean())
        logging.info("  Clients below lower: %d" % mask_lower.sum())
        logging.info("  Clients above upper: %d" % mask_upper.sum())
        
        # Compute weights
        weights = torch.exp(-self.alphaguard_params['penalty_strength'] * deviation_scores)
        weights = torch.clamp(weights, min=self.alphaguard_params['min_weight_threshold'])
        
        logging.info("Weight Distribution:")
        logging.info("  Min weight:  %.4f" % weights.min())
        logging.info("  Max weight:  %.4f" % weights.max())
        logging.info("  Mean weight: %.4f" % weights.mean())
        logging.info("  Weight std:  %.4f" % weights.std())
        
        # Select clients to keep - bidirectional anomaly detection based on Top-K distribution
        # Calculate removal number: actual malicious clients selected + 1
        # First, count how many malicious clients are actually selected in this round
        if hasattr(self, '_current_client_ids'):
            num_malicious_selected = sum(1 for cid in self._current_client_ids if cid < self.args.num_corrupt)
        else:
            # Fallback: assume all clients with index < num_corrupt are malicious (not accurate but safe)
            num_malicious_selected = min(self.args.num_corrupt, num_clients)
        
        # Remove number = actual malicious clients + 1, but ensure at least 1 client remains
        num_to_remove = min(num_malicious_selected + 1, num_clients - 1)
        
        logging.info("Removal strategy (based on actual malicious participants):")
        logging.info("  Clients selected this round: %d" % num_clients)
        logging.info("  Malicious clients selected: %d" % num_malicious_selected)
        logging.info("  Num to remove: %d (malicious + 1, will keep at least %d)" % (num_to_remove, num_clients - num_to_remove))
        
        # Use robust statistics: median and MAD (Median Absolute Deviation)
        median_topk = torch.median(top_k_mean)
        mad = torch.median(torch.abs(top_k_mean - median_topk))
        
        # Fallback to IQR-based estimation if MAD is too small
        if mad < 1e-6:
            q75, q25 = torch.quantile(top_k_mean, 0.75), torch.quantile(top_k_mean, 0.25)
            iqr = q75 - q25
            mad = iqr / 1.35  # Convert IQR to MAD equivalent
            if mad < 1e-6:
                mad = torch.std(top_k_mean)  # Last resort fallback
        
        # Compute modified Z-scores (robust to outliers)
        modified_z_scores = 0.6745 * torch.abs(top_k_mean - median_topk) / (mad + 1e-6)
        
        logging.info("Bidirectional Anomaly Detection (Top-K based):")
        logging.info("  Median Top-K: %.4f" % median_topk)
        logging.info("  MAD: %.4f" % mad)
        logging.info("  Q25: %.4f, Q75: %.4f" % (torch.quantile(top_k_mean, 0.25), torch.quantile(top_k_mean, 0.75)))
        
        # Show anomaly scores for each client
        if hasattr(self, '_current_client_ids'):
            logging.info("Modified Z-scores by client:")
            for i, client_id in enumerate(self._current_client_ids):
                mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
                deviation = top_k_mean[i] - median_topk
                direction = "HIGH" if deviation > 0 else "LOW"
                logging.info("  Client %d%s: Top-K=%.4f, Deviation=%.4f (%s), Z-score=%.4f" % 
                           (client_id, mal_str, top_k_mean[i], abs(deviation), direction, modified_z_scores[i]))
        
        # Remove clients with highest modified Z-scores (true outliers from median)
        # This correctly handles both high and low outliers automatically
        # Example: [1,1,1,1,5,5,5,5,5,5,5] -> median=5, only 1's have high Z-scores
        
        _, sorted_indices_by_zscore = torch.sort(modified_z_scores)
        remaining_indices = sorted_indices_by_zscore[:-num_to_remove]  # Keep those with low Z-scores
        removed_indices = sorted_indices_by_zscore[-num_to_remove:]     # Remove those with high Z-scores
        
        # Analyze which removed clients had high vs low Top-K values
        removed_topk_values = top_k_mean[removed_indices]
        removed_above_median = (removed_topk_values > median_topk).sum().item()
        removed_below_median = (removed_topk_values < median_topk).sum().item()
        
        logging.info("Client Selection (MAD-based Outlier Removal):")
        logging.info("  Total removed: %d" % num_to_remove)
        logging.info("  Removed indices: %s" % str(removed_indices.tolist()))
        
        # 显示被移除的客户端ID（如果可用）
        if hasattr(self, '_current_client_ids'):
            removed_client_ids = [self._current_client_ids[idx.item()] if isinstance(idx, torch.Tensor) else self._current_client_ids[idx] 
                                 for idx in removed_indices]
            removed_client_ids_str = ", ".join([f"{cid}(MAL)" if cid < self.args.num_corrupt else f"{cid}" 
                                               for cid in removed_client_ids])
            logging.info("  Removed client IDs: [%s]" % removed_client_ids_str)
        
        logging.info("  Removed Top-K means: %s" % str([round(x, 4) for x in removed_topk_values.tolist()]))
        logging.info("  Removed Z-scores: %s" % str([round(x, 4) for x in modified_z_scores[removed_indices].tolist()]))
        logging.info("  Above median: %d, Below median: %d" % (removed_above_median, removed_below_median))
        logging.info("  Remaining clients: %d/%d" % (len(remaining_indices), num_clients))
        
        # 分析恶意客户端的检测情况
        # 需要根据真正的客户端ID来判断，而不是数组索引
        if hasattr(self, '_current_client_ids'):
            # 统计这一轮被选中的恶意和良性客户端数（已在前面计算过，这里复用）
            num_benign_selected = num_clients - num_malicious_selected
            
            # 将 tensor 转换为 Python list，确保索引正确
            removed_indices_list = removed_indices.tolist() if isinstance(removed_indices, torch.Tensor) else list(removed_indices)
            
            # 统计被移除的恶意和良性客户端数（使用真正的客户端ID）
            malicious_detected = sum(1 for idx in removed_indices_list if self._current_client_ids[idx] < self.args.num_corrupt)
            benign_removed = sum(1 for idx in removed_indices_list if self._current_client_ids[idx] >= self.args.num_corrupt)
            
            logging.info("Detection Analysis:")
            logging.info("  Malicious selected: %d/%d (this round)" % (num_malicious_selected, num_clients))
            logging.info("  Benign selected: %d/%d (this round)" % (num_benign_selected, num_clients))
            logging.info("  Malicious detected: %d/%d" % (malicious_detected, num_malicious_selected))
            logging.info("  Benign removed: %d/%d" % (benign_removed, num_benign_selected))
            
            if num_malicious_selected > 0:
                logging.info("  Detection rate: %.2f%%" % (malicious_detected/num_malicious_selected*100))
            else:
                logging.info("  Detection rate: N/A (no malicious clients selected)")
            
            if num_benign_selected > 0:
                logging.info("  False positive: %.2f%%" % (benign_removed/num_benign_selected*100))
            else:
                logging.info("  False positive: N/A (no benign clients selected)")
        else:
            # Fallback: 使用索引判断（不准确，但至少不会报错）
            malicious_detected = sum(1 for idx in removed_indices if idx < self.args.num_corrupt)
            benign_removed = sum(1 for idx in removed_indices if idx >= self.args.num_corrupt)
            num_benign = num_clients - self.args.num_corrupt
            
            logging.info("Detection Analysis (fallback):")
            logging.info("  Malicious detected: %d" % malicious_detected)
            logging.info("  Benign removed: %d" % benign_removed)
            if num_benign > 0:
                logging.info("  False positive: %.2f%%" % (benign_removed/num_benign*100))
            else:
                logging.info("  False positive: N/A (no benign clients selected)")
            
        logging.info("Final Statistics:")
        logging.info("  Reference value: %.4f" % robust_reference)
        logging.info("  Normal range: [%.4f, %.4f]" % (normal_lower, normal_upper))
        logging.info("  Selected clients: %d/%d" % (len(remaining_indices), num_clients))
        logging.info("  Weight range: [%.4f, %.4f]" % (weights[remaining_indices].min(), weights[remaining_indices].max()))
        
        return remaining_indices, weights

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        client_ids = []  # 保持客户端ID的顺序，与local_updates对应

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            client_ids.append(_id)  # 记录客户端ID，保持与local_updates的对应关系

        # 为了兼容旧代码，仍然构建malicious_id和benign_id列表
        benign_id = [cid for cid in client_ids if cid >= self.args.num_corrupt]
        malicious_id = [cid for cid in client_ids if cid < self.args.num_corrupt]

        chosen_clients = client_ids  # 使用原始顺序，而不是重新排序
        num_chosen_clients = len(client_ids)
        inter_model_updates = torch.stack(local_updates, dim=0)

        tda_list = []
        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

            mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
            tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


        ######## MZ-score calculation ########
        mpsa_std = np.std(mpsa_list)
        mpsa_med = np.median(mpsa_list)

        mzscore_mpsa = []
        for i in range(len(mpsa_list)):
            mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
        tda_std = np.std(tda_list)
        tda_med = np.median(tda_list)
        mzscore_tda = []
        for i in range(len(tda_list)):
            mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########

        benign_idx1 = set([i for i in range(num_chosen_clients)])
        benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
        benign_idx2 = set([i for i in range(num_chosen_clients)])
        benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

        benign_set = benign_idx2.intersection(benign_idx1)
        
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        # del grad_norm
        
        benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

        # 修复：使用真正的客户端ID来判断，而不是数组索引
        # benign_idx 是数组索引，需要映射到真正的客户端ID
        correct = 0
        for idx in benign_idx:
            client_id = chosen_clients[idx]  # 获取真正的客户端ID
            if client_id >= self.args.num_corrupt:  # 良性客户端
                correct += 1

        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                client_id = chosen_clients[idx]  # 获取真正的客户端ID
                if client_id < self.args.num_corrupt:  # 恶意客户端被误判为良性
                    wrong += 1
            FPR = wrong / len(malicious_id)

        # 显示被选中的客户端ID（而不是数组索引）
        selected_client_ids = [chosen_clients[idx] for idx in benign_idx]
        logging.info('benign client IDs:     %s' % str(benign_id))
        logging.info('selected client IDs:   %s' % str(selected_client_ids))
        logging.info('selected array indices: %s' % str(benign_idx))

        logging.info('FPR:       %.4f'  % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {}
        for idx in benign_idx:
            current_dict[chosen_clients[idx]] = benign_updates[idx]

        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    # def compute_robustLR(self, agent_updates_dict):

    #     agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
    #     sm_of_signs = torch.abs(sum(agent_updates_sign))
    #     mask=torch.zeros_like(sm_of_signs)
    #     mask[sm_of_signs < self.args.theta] = 0
    #     mask[sm_of_signs >= self.args.theta] = 1
    #     sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
    #     sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
    #     return sm_of_signs.to(self.args.device), mask

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        # 🔧 FIX: 对于符号不一致的参数，使用小学习率或不更新，而不是负学习率
        # 原实现使用负学习率会导致模型朝反方向更新，准确率极低！
        sm_of_signs[sm_of_signs < self.args.theta] = 0  # 不更新（或用 0.1 * self.server_lr）
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        client_ids = []  # 保持客户端ID的顺序，与local_updates对应

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            client_ids.append(_id)  # 记录客户端ID，保持与local_updates的对应关系

        # 为了兼容旧代码，仍然构建malicious_id和benign_id列表
        benign_id = [cid for cid in client_ids if cid >= self.args.num_corrupt]
        malicious_id = [cid for cid in client_ids if cid < self.args.num_corrupt]

        chosen_clients = client_ids  # 使用原始顺序，而不是重新排序
        num_chosen_clients = len(client_ids)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        client_ids = []  # 保持客户端ID的顺序，与local_updates对应

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            client_ids.append(_id)  # 记录客户端ID，保持与local_updates的对应关系

        # 为了兼容旧代码，仍然构建malicious_id和benign_id列表
        benign_id = [cid for cid in client_ids if cid >= self.args.num_corrupt]
        malicious_id = [cid for cid in client_ids if cid < self.args.num_corrupt]

        names = client_ids  # 使用原始顺序，而不是重新排序
        num_chosen_clients = len(client_ids)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            # 修复：names[i] 现在与 client_updates[i] 对应
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))

        # 修复：确保 wv[i] 与 local_updates[i] 对应
        # wv 是基于 updates 计算的，而 updates 的顺序与 client_updates 一致
        # local_updates 的顺序也与 client_updates 一致（都是按字典顺序）
        weighted_updates = [update * wv[i] for i, update in enumerate(local_updates)]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model
    

    
    def _calculate_tpr_fpr(self, detected_labels, n, m):
        """
        计算TPR和FPR
        detected_labels: 检测标签数组（-1表示异常，0表示正常）
        n: 总客户端数
        m: 恶意客户端数
        """
        actual_labels = np.array([1 if i < m else 0 for i in range(n)])
        detected_labels = np.array(detected_labels)
        
        TP = np.sum((actual_labels == 1) & (detected_labels == -1))
        FP = np.sum((actual_labels == 0) & (detected_labels == -1))
        FN = np.sum((actual_labels == 1) & (detected_labels == 0))
        TN = np.sum((actual_labels == 0) & (detected_labels == 0))
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        return TPR, FPR
    
    def _compute_wasserstein_distance_matrix(self, l):
        """计算Wasserstein距离矩阵"""
        n = len(l)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = wasserstein_distance(l[i], l[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        logging.info(f"MARS: Wasserstein distance matrix computed, shape: {distance_matrix.shape}")
        return distance_matrix
    
    def _kmeans_wasserstein_clustering(self, l, k=2):
        """使用Wasserstein距离进行K-means聚类"""
        distance_matrix = self._compute_wasserstein_distance_matrix(l)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(distance_matrix)
        return kmeans
    
    def _compute_cluster_centers(self, l, labels, k):
        """计算聚类中心"""
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(l[i])
        
        centers = []
        for cluster in clusters:
            if len(cluster) > 0:
                centers.append(np.mean(cluster, axis=0))
            else:
                # 空聚类，使用零向量
                centers.append(np.zeros_like(l[0]))
        return centers
    
    def _detect_anomaly(self, l, kmeans, threshold):
        """检测异常聚类"""
        labels = kmeans.labels_
        centers = self._compute_cluster_centers(l, labels, k=2)
        
        center_distance = wasserstein_distance(centers[0], centers[1])
        logging.info(f"MARS: Center distance: {center_distance:.4f}, threshold: {threshold}")
        
        if center_distance <= threshold:
            logging.info("MARS: No anomalies detected (center distance below threshold)")
            return np.zeros_like(labels)
        else:
            center_norms = [np.linalg.norm(center, ord=1) for center in centers]
            logging.info(f"MARS: Center norms: {[round(x, 4) for x in center_norms]}")
            
            anomaly_cluster = np.argmax(center_norms)
            anomaly_labels = np.where(labels == anomaly_cluster, -1, 0)
            
            num_anomalies = np.sum(anomaly_labels == -1)
            logging.info(f"MARS: Detected {num_anomalies} anomalous clients in cluster {anomaly_cluster}")
            
            return anomaly_labels
    
    def _cluster_and_detect_anomalies(self, l, threshold):
        """聚类并检测异常"""
        kmeans = self._kmeans_wasserstein_clustering(l)
        anomaly_labels = self._detect_anomaly(l, kmeans, threshold)
        return anomaly_labels
    
    def _compute_cbe(self, net):
        """
        计算Channel-wise Batch normalization Eigenvector (CBE)
        修改版本：当track_running_stats=False时，直接使用Conv+BN权重计算特征
        """
        # 确保模型在评估模式
        net.eval()
        
        params = net.state_dict()
        all_lips = []
        u = 0.05
        
        # 获取所有模块并构建名称到模块的映射
        named_modules = list(net.named_modules())
        
        # 找到所有Conv2d层，建立映射关系
        conv_layers = {}
        for name, m in named_modules:
            if isinstance(m, nn.Conv2d):
                conv_layers[name] = m
        
        # 遍历所有BatchNorm2d层
        for name, m in named_modules:
            if isinstance(m, nn.BatchNorm2d):
                # 查找对应的Conv2d层（通常是紧邻的前一层）
                # 尝试多种命名模式
                conv = None
                
                # 模式1: 如果BatchNorm名为 "bn1"，对应Conv为 "conv1"
                if name.startswith('bn'):
                    conv_name = name.replace('bn', 'conv')
                    conv = conv_layers.get(conv_name)
                
                # 模式2: 如果在Sequential中，找到前一个Conv层
                if conv is None:
                    # 分析名称结构，找到父模块中的Conv层
                    parts = name.split('.')
                    if len(parts) > 1:
                        # 尝试找同一个Sequential中的Conv层
                        for conv_name, conv_layer in conv_layers.items():
                            conv_parts = conv_name.split('.')
                            # 检查是否在同一个父模块下
                            if len(conv_parts) == len(parts):
                                if conv_parts[:-1] == parts[:-1]:
                                    # 同一父模块，尝试找到对应的Conv
                                    conv = conv_layer
                                    break
                
                # 如果还是找不到，跳过这个BatchNorm层
                if conv is None:
                    logging.warning(f"MARS CBE: Could not find Conv2d for BatchNorm {name}, skipping")
                    continue
                
                # 检查weight是否为None
                if m.weight is None:
                    logging.warning(f"MARS CBE: BatchNorm {name} has no weight, skipping")
                    continue
                
                # 方法改进：当track_running_stats=False时，使用BN的gamma权重作为标准化系数
                # 这样可以捕捉到模型训练后的真实权重差异
                weight = m.weight.abs()  # gamma参数
                
                channel_lips = []
                for idx in range(weight.shape[0]):
                    # 直接使用Conv权重和BN的gamma权重计算
                    w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * weight[idx]
                    # 使用SVD计算最大奇异值
                    try:
                        channel_lips.append(torch.svd(w.cpu())[1].max())
                    except:
                        # 如果SVD失败，使用L2范数作为近似
                        channel_lips.append(torch.norm(w.cpu(), p=2))
                
                channel_lips = torch.Tensor(channel_lips)
                threshold = torch.quantile(channel_lips, 1 - u)
                top_k_lips = channel_lips[channel_lips >= threshold]
                all_lips.append(top_k_lips)
        
        # 将所有Top-k Lipschitz常数拼接成一维向量
        if len(all_lips) > 0:
            all_lips_vector = torch.cat(all_lips)
        else:
            # 如果没有找到任何BatchNorm层，返回一个默认向量
            logging.warning("MARS CBE: No BatchNorm layers found, using default vector")
            all_lips_vector = torch.ones(100)  # 返回默认向量
        
        return np.array(all_lips_vector)
    
