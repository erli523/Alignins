# import copy

# import torch
# from torch.nn.utils import parameters_to_vector
# import numpy as np
# import logging
# from utils import vector_to_model, vector_to_name_param

# import sklearn.metrics.pairwise as smp
# from geom_median.torch import compute_geometric_median 


# class Aggregation():
#     def __init__(self, agent_data_sizes, n_params, args):
#         self.agent_data_sizes = agent_data_sizes
#         self.args = args
#         self.server_lr = args.server_lr
#         self.n_params = n_params
        
#         # AlphaGuard parameters (optimized for this project)
#         self.alphaguard_params = {
#             "epsilon": 1e-3,                      # Numerical stability
#             "l2_filter_lower_percentile": 25,     # Filter bottom 25%
#             "l2_filter_upper_percentile": 75,     # Filter top 25%
#             "min_trusted_clients": 8,             # At least 8 clients (40% of 20)
#             "top_k_percentile": 0.01,             # Top 1% of parameters (more focused)
#             "robust_alpha_exclude_ratio": 0.15,   # Exclude extreme 15%
#             "normal_range_lower_ratio": 0.9,      # Tighter lower bound (was 0.3)
#             "normal_range_upper_ratio": 1.1,      # Tighter upper bound (was 3.0)
#             "penalty_strength": 2.0,              # Stronger penalty (was 1.5)
#             "min_weight_threshold": 0.01,         # Lower minimum weight (was 0.05)
#             "alpha_outlier_num": min(args.num_corrupt + 2, args.num_agents // 2)  # Adaptive removal
#         }
        
#         if self.args.aggr == 'foolsgold':
#             self.memory_dict = dict()
#             self.wv_history = []
        
         
#     def aggregate_updates(self, global_model, agent_updates_dict):


#         lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
#         if self.args.aggr != "rlr":
#             lr_vector = lr_vector
#         else:
#             lr_vector, _ = self.compute_robustLR(agent_updates_dict)
#         # mask = torch.ones_like(agent_updates_dict[0])
#         aggregated_updates = 0
#         cur_global_params = parameters_to_vector(
#             [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
#         if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
#             aggregated_updates = self.agg_avg(agent_updates_dict)

#         elif self.args.aggr == 'alignins':
#             aggregated_updates = self.agg_alignins(agent_updates_dict, cur_global_params)
#         elif self.args.aggr == 'mmetric':
#             aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
#         elif self.args.aggr == 'foolsgold':
#             aggregated_updates = self.agg_foolsgold(agent_updates_dict)
#         elif self.args.aggr == 'signguard':
#             aggregated_updates = self.agg_signguard(agent_updates_dict)
#         elif self.args.aggr == "mkrum":
#             aggregated_updates = self.agg_mkrum(agent_updates_dict)
#         elif self.args.aggr == "rfa":
#             aggregated_updates = self.agg_rfa(agent_updates_dict)
#         elif self.args.aggr == "alphaguard":
#             aggregated_updates = self.agg_alphaguard(agent_updates_dict)
#         neurotoxin_mask = {}
#         updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
#         for name in updates_dict:
#             updates = updates_dict[name].abs().view(-1)
#             gradients_length = torch.numel(updates)
#             _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
#             mask_flat = torch.zeros(gradients_length)
#             mask_flat[indices.cpu()] = 1
#             neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

#         cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
#         new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
#         vector_to_model(new_global_params, global_model)
#         return updates_dict, neurotoxin_mask

#     def agg_rfa(self, agent_updates_dict):
#         local_updates = []

#         for _id, update in agent_updates_dict.items():
#             local_updates.append(update)

#         n = len(local_updates)
#         temp_updates = torch.stack(local_updates, dim=0)
#         weights = torch.ones(n).to(self.args.device)  
#         gw = compute_geometric_median(local_updates, weights).median
#         for i in range(2):
#             weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
#             gw = compute_geometric_median(local_updates, weights).median

#         aggregated_model = gw
#         return aggregated_model

#     def agg_alphaguard(self, agent_updates_dict):
#         """AlphaGuard defense method implementation"""
#         logging.info("------------ AlphaGuard Defense ------------")
#         logging.info("Configuration:")
#         logging.info("  Total clients: %d" % len(agent_updates_dict))
#         logging.info("  Malicious clients: %d" % self.args.num_corrupt)
#         logging.info("  Poison fraction: %.2f" % self.args.poison_frac)
        
#         # Convert updates to tensor format and keep track of client IDs
#         updates_list = []
#         client_ids = []
#         for _id, update in agent_updates_dict.items():
#             updates_list.append(update)
#             client_ids.append(_id)
#         updates_tensor = torch.stack(updates_list, dim=0)
        
#         # Store client_ids for later use
#         self._current_client_ids = client_ids
        
#         # Stage 1: L2 robust reference selection
#         g_ref, trusted_indices = self._alphaguard_stage1(updates_tensor)
        
#         # Stage 2: Alpha vector computation
#         alpha_vectors = self._alphaguard_stage2(updates_tensor, g_ref)
        
#         # Stage 3: Robust detection and penalty
#         remaining_indices, weights = self._alphaguard_stage3(updates_tensor, alpha_vectors)
        
#         # Final aggregation with weighted average
#         selected_updates = updates_tensor[remaining_indices]
#         selected_weights = weights[remaining_indices].to(self.args.device)
        
#         # Normalize weights
#         selected_weights = selected_weights / selected_weights.sum()
#         selected_weights = selected_weights.view(-1, 1)
        
#         aggregated_update = torch.sum(selected_updates * selected_weights, dim=0)
        
#         logging.info("Final aggregation:")
#         logging.info("  Selected clients: %d/%d" % (len(remaining_indices), len(updates_tensor)))
#         logging.info("  Weight sum: %.4f" % selected_weights.sum())
        
#         return aggregated_update

#     def _alphaguard_stage1(self, updates):
#         """Stage 1: L2 robust reference selection"""
#         num_clients = len(updates)
        
#         logging.info("------------ Stage 1: L2 Robust Reference Selection ------------")
#         logging.info("Processing %d client updates" % num_clients)
        
#         # Compute L2 norms
#         l2_norms = torch.norm(updates, dim=1)
        
#         logging.info("L2 norms statistics:")
#         logging.info("  Min:  %.4f" % l2_norms.min())
#         logging.info("  Max:  %.4f" % l2_norms.max())
#         logging.info("  Mean: %.4f" % l2_norms.mean())
#         logging.info("  Std:  %.4f" % l2_norms.std())
        
#         # Calculate percentile thresholds
#         lower_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_lower_percentile']/100)
#         upper_threshold = torch.quantile(l2_norms, self.alphaguard_params['l2_filter_upper_percentile']/100)
        
#         # Select trusted clients
#         trusted_mask = (l2_norms >= lower_threshold) & (l2_norms <= upper_threshold)
#         trusted_indices = torch.where(trusted_mask)[0]
        
#         # Safety check
#         if len(trusted_indices) < self.alphaguard_params['min_trusted_clients']:
#             logging.info("  Warning: Not enough trusted clients, using all clients")
#             trusted_indices = torch.arange(num_clients, device=updates.device)
        
#         # Compute reference using median
#         g_ref = torch.median(updates[trusted_indices], dim=0)[0]
        
#         logging.info("Stage 1 Results:")
#         logging.info("  Trusted clients: %d/%d" % (len(trusted_indices), num_clients))
#         logging.info("  L2 threshold: [%.4f, %.4f]" % (lower_threshold, upper_threshold))
#         logging.info("  Reference norm: %.4f" % torch.norm(g_ref))
        
#         return g_ref, trusted_indices

#     def _alphaguard_stage2(self, updates, g_ref):
#         """Stage 2: Alpha vector computation"""
#         logging.info("------------ Stage 2: Alpha Vector Computation ------------")
        
#         # Numerical stability - use larger epsilon
#         epsilon = 1e-3  # Increase epsilon for better stability
#         g_ref_safe = torch.where(torch.abs(g_ref) < epsilon,
#                                torch.sign(g_ref) * epsilon,
#                                g_ref)
        
#         # Compute alpha vectors
#         alpha_vectors = updates / g_ref_safe
        
#         # Replace NaN and Inf values
#         alpha_vectors = torch.nan_to_num(alpha_vectors, nan=0.0, posinf=1.0, neginf=-1.0)
        
#         alpha_norms = torch.norm(alpha_vectors, dim=1)
#         logging.info("Alpha vectors statistics:")
#         logging.info("  Shape:      %s" % str(alpha_vectors.shape))
#         logging.info("  Norm min:   %.4f" % alpha_norms.min())
#         logging.info("  Norm max:   %.4f" % alpha_norms.max())
#         logging.info("  Norm mean:  %.4f" % alpha_norms.mean())
#         logging.info("  Norm std:   %.4f" % alpha_norms.std())
#         logging.info("  Epsilon:    %.6f" % epsilon)
        
#         return alpha_vectors

#     def _alphaguard_stage3(self, updates, alpha_vectors):
#         """Stage 3: Robust detection and penalty"""
#         logging.info("------------ Stage 3: Robust Detection and Penalty ------------")
        
#         num_clients = len(alpha_vectors)
#         model_dim = alpha_vectors.shape[1]
        
#         logging.info("Processing %d clients with model dimension %d" % (num_clients, model_dim))
#         logging.info("Top-K percentage: %.1f%%" % (self.alphaguard_params['top_k_percentile']*100))
        
#         # Compute Top-K values
#         k = max(1, int(self.alphaguard_params['top_k_percentile'] * model_dim))
#         abs_alpha = torch.abs(alpha_vectors)
#         top_k_vals, _ = torch.topk(abs_alpha, k, dim=1)
#         top_k_mean = torch.mean(top_k_vals, dim=1)
        
#         # Handle NaN/Inf in top_k_mean
#         top_k_mean = torch.nan_to_num(top_k_mean, nan=1.0, posinf=10.0, neginf=0.0)
        
#         logging.info("Top-K mean statistics:")
#         logging.info("  Min:  %.4f" % top_k_mean.min())
#         logging.info("  Max:  %.4f" % top_k_mean.max())
#         logging.info("  Mean: %.4f" % top_k_mean.mean())
#         logging.info("  Std:  %.4f" % top_k_mean.std())
        
#         # Show top-k values for each client
#         if hasattr(self, '_current_client_ids'):
#             logging.info("Top-K values by client:")
#             for i, client_id in enumerate(self._current_client_ids):
#                 mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
#                 logging.info("  Client %d%s: %.4f" % (client_id, mal_str, top_k_mean[i]))
        
#         # Compute robust reference (V19 innovation)
#         exclude_count = max(1, int(self.alphaguard_params['robust_alpha_exclude_ratio'] * num_clients))
        
#         # Safety check for small number of clients
#         if num_clients <= 2 * exclude_count:
#             exclude_count = max(1, num_clients // 4)
        
#         sorted_vals, _ = torch.sort(top_k_mean)
        
#         if len(sorted_vals) > 2 * exclude_count:
#             trimmed_vals = sorted_vals[exclude_count:-exclude_count]
#         else:
#             trimmed_vals = sorted_vals
        
#         robust_reference = torch.mean(trimmed_vals)
        
#         # Safety check for robust reference
#         if torch.isnan(robust_reference) or torch.isinf(robust_reference) or robust_reference == 0:
#             robust_reference = torch.median(top_k_mean)
#             logging.info("  Using median as fallback reference: %.4f" % robust_reference)
        
#         # Compute bidirectional penalty
#         normal_lower = robust_reference * self.alphaguard_params['normal_range_lower_ratio']
#         normal_upper = robust_reference * self.alphaguard_params['normal_range_upper_ratio']
        
#         logging.info("Robust reference analysis:")
#         logging.info("  Reference: %.4f" % robust_reference)
#         logging.info("  Lower bound: %.4f" % normal_lower)
#         logging.info("  Upper bound: %.4f" % normal_upper)
        
#         deviation_scores = torch.zeros_like(top_k_mean)
#         mask_lower = top_k_mean < normal_lower
#         mask_upper = top_k_mean > normal_upper
        
#         # Compute deviation with safety checks
#         if normal_lower > 0:
#             deviation_scores[mask_lower] = (normal_lower - top_k_mean[mask_lower]) / (normal_lower + 1e-6)
#         if normal_upper > 0:
#             deviation_scores[mask_upper] = (top_k_mean[mask_upper] - normal_upper) / (normal_upper + 1e-6)
        
#         # Clamp deviation scores to reasonable range
#         deviation_scores = torch.clamp(deviation_scores, 0, 10)
        
#         logging.info("Deviation scores:")
#         logging.info("  Min:  %.4f" % deviation_scores.min())
#         logging.info("  Max:  %.4f" % deviation_scores.max())
#         logging.info("  Mean: %.4f" % deviation_scores.mean())
#         logging.info("  Clients below lower: %d" % mask_lower.sum())
#         logging.info("  Clients above upper: %d" % mask_upper.sum())
        
#         # Compute weights
#         weights = torch.exp(-self.alphaguard_params['penalty_strength'] * deviation_scores)
#         weights = torch.clamp(weights, min=self.alphaguard_params['min_weight_threshold'])
        
#         logging.info("Weight Distribution:")
#         logging.info("  Min weight:  %.4f" % weights.min())
#         logging.info("  Max weight:  %.4f" % weights.max())
#         logging.info("  Mean weight: %.4f" % weights.mean())
#         logging.info("  Weight std:  %.4f" % weights.std())
        
#         # Select clients to keep - bidirectional anomaly detection based on Top-K distribution
#         num_to_remove = min(self.alphaguard_params['alpha_outlier_num'], num_clients - 1)
        
#         # Use robust statistics: median and MAD (Median Absolute Deviation)
#         median_topk = torch.median(top_k_mean)
#         mad = torch.median(torch.abs(top_k_mean - median_topk))
        
#         # Fallback to IQR-based estimation if MAD is too small
#         if mad < 1e-6:
#             q75, q25 = torch.quantile(top_k_mean, 0.75), torch.quantile(top_k_mean, 0.25)
#             iqr = q75 - q25
#             mad = iqr / 1.35  # Convert IQR to MAD equivalent
#             if mad < 1e-6:
#                 mad = torch.std(top_k_mean)  # Last resort fallback
        
#         # Compute modified Z-scores (robust to outliers)
#         modified_z_scores = 0.6745 * torch.abs(top_k_mean - median_topk) / (mad + 1e-6)
        
#         logging.info("Bidirectional Anomaly Detection (Top-K based):")
#         logging.info("  Median Top-K: %.4f" % median_topk)
#         logging.info("  MAD: %.4f" % mad)
#         logging.info("  Q25: %.4f, Q75: %.4f" % (torch.quantile(top_k_mean, 0.25), torch.quantile(top_k_mean, 0.75)))
        
#         # Show anomaly scores for each client
#         if hasattr(self, '_current_client_ids'):
#             logging.info("Modified Z-scores by client:")
#             for i, client_id in enumerate(self._current_client_ids):
#                 mal_str = " (MAL)" if client_id < self.args.num_corrupt else ""
#                 deviation = top_k_mean[i] - median_topk
#                 direction = "HIGH" if deviation > 0 else "LOW"
#                 logging.info("  Client %d%s: Top-K=%.4f, Deviation=%.4f (%s), Z-score=%.4f" % 
#                            (client_id, mal_str, top_k_mean[i], abs(deviation), direction, modified_z_scores[i]))
        
#         # Remove clients with highest modified Z-scores (bidirectional outliers)
#         _, sorted_indices = torch.sort(modified_z_scores)
#         remaining_indices = sorted_indices[:-num_to_remove]
#         removed_indices = sorted_indices[-num_to_remove:]
        
#         logging.info("Client Selection:")
#         logging.info("  Removed clients: %d" % num_to_remove)
#         logging.info("  Removed indices: %s" % str(removed_indices.tolist()))
#         logging.info("  Top-K means (removed): %s" % str([round(x, 4) for x in top_k_mean[removed_indices].tolist()]))
#         logging.info("  Z-scores (removed): %s" % str([round(x, 4) for x in modified_z_scores[removed_indices].tolist()]))
#         logging.info("  Remaining clients: %d/%d" % (len(remaining_indices), num_clients))
        
#         # 分析恶意客户端的检测情况
#         malicious_detected = sum(1 for idx in removed_indices if idx < self.args.num_corrupt)
#         benign_removed = sum(1 for idx in removed_indices if idx >= self.args.num_corrupt)
#         logging.info("Detection Analysis:")
#         logging.info("  Malicious detected: %d/%d" % (malicious_detected, self.args.num_corrupt))
#         logging.info("  Benign removed: %d/%d" % (benign_removed, num_clients-self.args.num_corrupt))
#         if self.args.num_corrupt > 0:
#             logging.info("  Detection rate: %.2f%%" % (malicious_detected/self.args.num_corrupt*100))
#             logging.info("  False positive: %.2f%%" % (benign_removed/(num_clients-self.args.num_corrupt)*100))
        
#         logging.info("Final Statistics:")
#         logging.info("  Reference value: %.4f" % robust_reference)
#         logging.info("  Normal range: [%.4f, %.4f]" % (normal_lower, normal_upper))
#         logging.info("  Selected clients: %d/%d" % (len(remaining_indices), num_clients))
#         logging.info("  Weight range: [%.4f, %.4f]" % (weights[remaining_indices].min(), weights[remaining_indices].max()))
        
#         return remaining_indices, weights

#     def agg_alignins(self, agent_updates_dict, flat_global_model):
#         local_updates = []
#         benign_id = []
#         malicious_id = []

#         for _id, update in agent_updates_dict.items():
#             local_updates.append(update)
#             if _id < self.args.num_corrupt:
#                 malicious_id.append(_id)
#             else:
#                 benign_id.append(_id)

#         chosen_clients = malicious_id + benign_id
#         num_chosen_clients = len(malicious_id + benign_id)
#         inter_model_updates = torch.stack(local_updates, dim=0)

#         tda_list = []
#         mpsa_list = []
#         major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
#         cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
#         for i in range(len(inter_model_updates)):
#             _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), int(len(inter_model_updates[i]) * self.args.sparsity))

#             mpsa_list.append((torch.sum(torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(inter_model_updates[i][init_indices])).item())
    
#             tda_list.append(cos(inter_model_updates[i], flat_global_model).item())


#         logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
#         logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])


#         ######## MZ-score calculation ########
#         mpsa_std = np.std(mpsa_list)
#         mpsa_med = np.median(mpsa_list)

#         mzscore_mpsa = []
#         for i in range(len(mpsa_list)):
#             mzscore_mpsa.append(np.abs(mpsa_list[i] - mpsa_med) / mpsa_std)

#         logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])
        
#         tda_std = np.std(tda_list)
#         tda_med = np.median(tda_list)
#         mzscore_tda = []
#         for i in range(len(tda_list)):
#             mzscore_tda.append(np.abs(tda_list[i] - tda_med) / tda_std)

#         logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

#         ######## Anomaly detection with MZ score ########

#         benign_idx1 = set([i for i in range(num_chosen_clients)])
#         benign_idx1 = benign_idx1.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s)]))
#         benign_idx2 = set([i for i in range(num_chosen_clients)])
#         benign_idx2 = benign_idx2.intersection(set([int(i) for i in np.argwhere(np.array(mzscore_tda) < self.args.lambda_c)]))

#         benign_set = benign_idx2.intersection(benign_idx1)
        
#         benign_idx = list(benign_set)
#         if len(benign_idx) == 0:
#             return torch.zeros_like(local_updates[0])

#         benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

#         ######## Post-filtering model clipping ########
        
#         updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
#         norm_clip = updates_norm.median(dim=0)[0].item()
#         benign_updates = torch.stack(local_updates, dim=0)
#         updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
#         updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
#         # del grad_norm
        
#         benign_updates = (benign_updates/updates_norm)*updates_norm_clipped

#         correct = 0
#         for idx in benign_idx:
#             if idx >= len(malicious_id):
#                 correct += 1

#         TPR = correct / len(benign_id)

#         if len(malicious_id) == 0:
#             FPR = 0
#         else:
#             wrong = 0
#             for idx in benign_idx:
#                 if idx < len(malicious_id):
#                     wrong += 1
#             FPR = wrong / len(malicious_id)

#         logging.info('benign update index:   %s' % str(benign_id))
#         logging.info('selected update index: %s' % str(benign_idx))

#         logging.info('FPR:       %.4f'  % FPR)
#         logging.info('TPR:       %.4f' % TPR)

#         current_dict = {}
#         for idx in benign_idx:
#             current_dict[chosen_clients[idx]] = benign_updates[idx]

#         aggregated_update = self.agg_avg(current_dict)
#         return aggregated_update

#     def agg_avg(self, agent_updates_dict):
#         """ classic fed avg """

#         sm_updates, total_data = 0, 0
#         for _id, update in agent_updates_dict.items():
#             n_agent_data = self.agent_data_sizes[_id]
#             sm_updates +=  n_agent_data * update
#             total_data += n_agent_data
#         return  sm_updates / total_data

    
#     def agg_mkrum(self, agent_updates_dict):
#         krum_param_m = 10
#         def _compute_krum_score( vec_grad_list, byzantine_client_num):
#             krum_scores = []
#             num_client = len(vec_grad_list)
#             for i in range(0, num_client):
#                 dists = []
#                 for j in range(0, num_client):
#                     if i != j:
#                         dists.append(
#                             torch.norm(vec_grad_list[i]- vec_grad_list[j])
#                             .item() ** 2
#                         )
#                 dists.sort()  # ascending
#                 score = dists[0: num_client - byzantine_client_num - 2]
#                 krum_scores.append(sum(score))
#             return krum_scores

#         benign_id = []
#         malicious_id = []

#         for _id, update in agent_updates_dict.items():
#             # local_updates.append(update)
#             if _id < self.args.num_corrupt:
#                 malicious_id.append(_id)
#             else:
#                 benign_id.append(_id)

#         # Compute list of scores
#         __nbworkers = len(agent_updates_dict)
#         krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
#         score_index = torch.argsort(
#             torch.Tensor(krum_scores)
#         ).tolist()  # indices; ascending
#         score_index = score_index[0: krum_param_m]

#         print('%d clients are selected' % len(score_index))
#         return_updates = [agent_updates_dict[i] for i in score_index]


#         return sum(return_updates)/len(return_updates)

#     def compute_robustLR(self, agent_updates_dict):

#         agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
#         sm_of_signs = torch.abs(sum(agent_updates_sign))
#         mask=torch.zeros_like(sm_of_signs)
#         mask[sm_of_signs < self.args.theta] = 0
#         mask[sm_of_signs >= self.args.theta] = 1
#         sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
#         sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
#         return sm_of_signs.to(self.args.device), mask

#     def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
#         local_updates = []
#         benign_id = []
#         malicious_id = []

#         for _id, update in agent_updates_dict.items():
#             local_updates.append(update)
#             if _id < self.args.num_corrupt:
#                 malicious_id.append(_id)
#             else:
#                 benign_id.append(_id)

#         chosen_clients = malicious_id + benign_id
#         num_chosen_clients = len(malicious_id + benign_id)

#         vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

#         cos_dis = [0.0] * len(vectorize_nets)
#         length_dis = [0.0] * len(vectorize_nets)
#         manhattan_dis = [0.0] * len(vectorize_nets)
#         for i, g_i in enumerate(vectorize_nets):
#             for j in range(len(vectorize_nets)):
#                 if i != j:
#                     g_j = vectorize_nets[j]

#                     cosine_distance = float(
#                         (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
#                     manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
#                     length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

#                     cos_dis[i] += cosine_distance
#                     length_dis[i] += length_distance
#                     manhattan_dis[i] += manhattan_distance

#         tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

#         cov_matrix = np.cov(tri_distance.T)
#         inv_matrix = np.linalg.inv(cov_matrix)

#         ma_distances = []
#         for i, g_i in enumerate(vectorize_nets):
#             t = tri_distance[i]
#             ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
#             ma_distances.append(ma_dis)

#         scores = ma_distances
#         print(scores)

#         p = 0.3
#         p_num = p*len(scores)
#         topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

#         print(topk_ind)
#         current_dict = {}

#         for idx in topk_ind:
#             current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

#         update = self.agg_avg(current_dict)

#         return update
   
#     def agg_foolsgold(self, agent_updates_dict):
#         def foolsgold(updates):
#             """
#             :param updates:
#             :return: compute similatiry and return weightings
#             """
#             n_clients = updates.shape[0]
#             cs = smp.cosine_similarity(updates) - np.eye(n_clients)

#             maxcs = np.max(cs, axis=1)
#             # pardoning
#             for i in range(n_clients):
#                 for j in range(n_clients):
#                     if i == j:
#                         continue
#                     if maxcs[i] < maxcs[j]:
#                         cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
#             wv = 1 - (np.max(cs, axis=1))

#             wv[wv > 1] = 1
#             wv[wv < 0] = 0

#             alpha = np.max(cs, axis=1)

#             # Rescale so that max value is wv
#             wv = wv / np.max(wv)
#             wv[(wv == 1)] = .99

#             # Logit function
#             wv = (np.log(wv / (1 - wv)) + 0.5)
#             wv[(np.isinf(wv) + wv > 1)] = 1
#             wv[(wv < 0)] = 0

#             # wv is the weight
#             return wv, alpha

#         local_updates = []
#         benign_id = []
#         malicious_id = []

#         for _id, update in agent_updates_dict.items():
#             local_updates.append(update)
#             if _id < self.args.num_corrupt:
#                 malicious_id.append(_id)
#             else:
#                 benign_id.append(_id)

#         names = malicious_id + benign_id
#         num_chosen_clients = len(malicious_id + benign_id)

#         client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
#         update_len = np.array(client_updates[0].shape).prod()
#         # print("client_updates size", client_models[0].parameters())
#         # update_len = len(client_updates)
#         # if self.memory is None:
#         #     self.memory = np.zeros((self.num_clients, update_len))
#         if len(names) < len(client_updates):
#             names = np.append([-1], names)  # put in adv

#         num_clients = num_chosen_clients
#         memory = np.zeros((num_clients, update_len))
#         updates = np.zeros((num_clients, update_len))

#         for i in range(len(client_updates)):
#             # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
#             updates[i] = np.reshape(client_updates[i], (update_len))
#             if names[i] in self.memory_dict.keys():
#                 self.memory_dict[names[i]] += updates[i]
#             else:
#                 self.memory_dict[names[i]] = copy.deepcopy(updates[i])
#             memory[i] = self.memory_dict[names[i]]
#         # self.memory += updates
#         use_memory = False

#         if use_memory:
#             wv, alpha = foolsgold(None)  # Use FG
#         else:
#             wv, alpha = foolsgold(updates)  # Use FG
#         # logger.info(f'[foolsgold agg] wv: {wv}')
#         self.wv_history.append(wv)

#         print(len(client_updates), len(wv))


#         weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

#         aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

#         print(aggregated_model.shape)

#         return aggregated_model

import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median 


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        # AlphaGuard parameters (optimized for this project)
        self.alphaguard_params = {
            "epsilon": 1e-3,                      # Numerical stability
            "l2_filter_lower_percentile": 25,     # Filter bottom 25%
            "l2_filter_upper_percentile": 75,     # Filter top 25%
            "min_trusted_clients": 8,             # At least 8 clients (40% of 20)
            "top_k_percentile": 0.001,             # Top 1% of parameters (more focused)
            "robust_alpha_exclude_ratio": 0.15,   # Exclude extreme 15%
            "normal_range_lower_ratio": 0.9,      # Tighter lower bound (was 0.3)
            "normal_range_upper_ratio": 1.1,      # Tighter upper bound (was 3.0)
            "penalty_strength": 2.0,              # Stronger penalty (was 1.5)
            "min_weight_threshold": 0.01,         # Lower minimum weight (was 0.05)
            "alpha_outlier_num": min(args.num_corrupt + 2, args.num_agents // 2)  # Adaptive removal
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
        weights = torch.ones(n).to(self.args.device)  
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(temp_updates-gw, dim=1)))
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
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
        
        logging.info("Final aggregation:")
        logging.info("  Selected clients: %d/%d" % (len(remaining_indices), len(updates_tensor)))
        logging.info("  Weight sum: %.4f" % selected_weights.sum())
        
        return aggregated_update

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
        
        # Compute reference using median
        g_ref = torch.median(updates[trusted_indices], dim=0)[0]
        
        logging.info("Stage 1 Results:")
        logging.info("  Trusted clients: %d/%d" % (len(trusted_indices), num_clients))
        logging.info("  L2 threshold: [%.4f, %.4f]" % (lower_threshold, upper_threshold))
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
        num_to_remove = min(self.alphaguard_params['alpha_outlier_num'], num_clients - 1)
        
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
        logging.info("  Removed Top-K means: %s" % str([round(x, 4) for x in removed_topk_values.tolist()]))
        logging.info("  Removed Z-scores: %s" % str([round(x, 4) for x in modified_z_scores[removed_indices].tolist()]))
        logging.info("  Above median: %d, Below median: %d" % (removed_above_median, removed_below_median))
        logging.info("  Remaining clients: %d/%d" % (len(remaining_indices), num_clients))
        
        # 分析恶意客户端的检测情况
        malicious_detected = sum(1 for idx in removed_indices if idx < self.args.num_corrupt)
        benign_removed = sum(1 for idx in removed_indices if idx >= self.args.num_corrupt)
        
        logging.info("Detection Analysis:")
        logging.info("  Malicious detected: %d/%d" % (malicious_detected, self.args.num_corrupt))
        logging.info("  Benign removed: %d/%d" % (benign_removed, num_clients-self.args.num_corrupt))
        if self.args.num_corrupt > 0:
            logging.info("  Detection rate: %.2f%%" % (malicious_detected/self.args.num_corrupt*100))
            logging.info("  False positive: %.2f%%" % (benign_removed/(num_clients-self.args.num_corrupt)*100))
        
        logging.info("Final Statistics:")
        logging.info("  Reference value: %.4f" % robust_reference)
        logging.info("  Normal range: [%.4f, %.4f]" % (normal_lower, normal_upper))
        logging.info("  Selected clients: %d/%d" % (len(remaining_indices), num_clients))
        logging.info("  Weight range: [%.4f, %.4f]" % (weights[remaining_indices].min(), weights[remaining_indices].max()))
        
        return remaining_indices, weights

    def agg_alignins(self, agent_updates_dict, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)
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

        correct = 0
        for idx in benign_idx:
            if idx >= len(malicious_id):
                correct += 1

        TPR = correct / len(benign_id)

        if len(malicious_id) == 0:
            FPR = 0
        else:
            wrong = 0
            for idx in benign_idx:
                if idx < len(malicious_id):
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))

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

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

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
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

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


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model