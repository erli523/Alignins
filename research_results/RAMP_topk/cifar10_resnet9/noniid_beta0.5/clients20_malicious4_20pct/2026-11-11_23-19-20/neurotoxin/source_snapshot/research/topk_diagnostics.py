"""Read-only, pre-aggregation RAMP coordinate diagnostics (no update repair)."""
import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from research.matched_random import LayerMatchedRandom


def csv_writer(path, fields):
    handle = Path(path).open('w', newline='', encoding='utf-8')
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    handle.flush()
    return handle, writer


def overlap(a, b):
    common = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - common
    return common, common / union if union else float('nan')


class TopKObserver:
    def __init__(self, output, options):
        self.output = Path(output)
        self.options = options
        self.round = 0
        self.layout = None
        self.previous = {}
        self.handles = []
        (self.output / 'tables').mkdir()
        (self.output / 'indices').mkdir()
        self.client_fields = [
            'round', 'client_id', 'role', 'reference_candidate', 'param_l2',
            'k', 'intersection_count', 'raw_rel_overlap', 'raw_rel_jaccard',
            'promoted_count', 'promoted_fraction', 'relative_score',
            'previous_round', 'raw_temporal_jaccard', 'rel_temporal_jaccard',
            'promoted_temporal_jaccard', 'promoted_previous_survival',
            'raw_boundary_tie_count', 'rel_boundary_tie_count',
        ]
        self.random_summary_fields = [
            'random_temporal_jaccard_mean', 'random_temporal_jaccard_std',
            'random_temporal_jaccard_q025', 'random_temporal_jaccard_q975',
            'temporal_jaccard_excess',
        ]
        base_client_fields = self.client_fields.copy()
        self.client_fields += self.random_summary_fields
        self.base_client_fields = base_client_fields
        self.random_fields = ['round', 'client_id', 'role', 'previous_round', 'repeat',
                              'promoted_count', 'observed_temporal_jaccard',
                              'random_temporal_jaccard']
        self.layer_fields = ['round', 'client_id', 'role', 'layer', 'numel',
                             'set_name', 'count', 'set_size', 'share', 'enrichment']
        self.round_fields = [
            'round', 'clients', 'parameter_count', 'k', 'reference_client_ids',
            'reference_malicious_fraction', 'reference_fallback',
            'reference_l2_low', 'reference_l2_high',
            'reference_abs_below_epsilon_fraction', 'reference_abs_zero_fraction',
            'mean_raw_rel_overlap', 'mean_promoted_fraction', 'analysis_seconds',
        ]
        self.writers = {}
        for name, fields in [('client_round', self.client_fields),
                             ('layer_round', self.layer_fields),
                             ('round', self.round_fields),
                             ('random_control', self.random_fields)]:
            handle, writer = csv_writer(self.output / 'tables' / (name + '.csv'), fields)
            self.handles.append(handle)
            self.writers[name] = writer

    def initialize_layout(self, model):
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        full_offset, analysis_offset = 0, 0
        self.layout = []
        for name, value in model.state_dict().items():
            size = value.numel()
            if name in trainable:
                self.layout.append(dict(name=name, shape=list(value.shape), numel=size,
                                        full_start=full_offset, analysis_start=analysis_offset))
                analysis_offset += size
            full_offset += size
        self.full_dimension = full_offset
        self.dimension = analysis_offset
        if not self.dimension:
            raise ValueError('No trainable parameters available for diagnostics')
        self.ends = np.cumsum([row['numel'] for row in self.layout])
        self.random_control = LayerMatchedRandom(
            self.layout, repeats=self.options.random_repeats, seed=self.options.random_seed)

    def trainable_vector(self, update):
        if update.ndim != 1 or update.numel() != self.full_dimension:
            raise ValueError('Update/state_dict layout mismatch')
        if self.dimension == self.full_dimension:
            return update.detach()
        return torch.cat([update[r['full_start']:r['full_start'] + r['numel']]
                          for r in self.layout]).detach()

    @torch.no_grad()
    def observe(self, model, updates, args):
        if self.layout is None:
            self.initialize_layout(model)
        self.round += 1
        started = time.perf_counter()
        ids = [int(client_id) for client_id in updates]
        if not ids:
            raise ValueError('No participating clients')
        matrix = torch.stack([self.trainable_vector(updates[c]) for c in ids])
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError('Non-finite update: cannot produce valid Top-K diagnostics')
        norms = torch.linalg.vector_norm(matrix, dim=1)
        low = torch.quantile(norms, self.options.l2_lower / 100)
        high = torch.quantile(norms, self.options.l2_upper / 100)
        candidate = torch.where((norms >= low) & (norms <= high))[0]
        fallback = candidate.numel() < min(self.options.min_reference, len(ids))
        if fallback:
            candidate = torch.arange(len(ids), device=matrix.device)
        selected = matrix.index_select(0, candidate)
        # Match existing AlphaGuard's lower-median convention, including even N.
        reference = (selected.median(dim=0).values if self.options.reference == 'median'
                     else selected.mean(dim=0))
        denominator = reference.abs() + self.options.epsilon
        k = max(1, int(self.dimension * self.options.topk_ratio))
        candidate_rows = set(candidate.cpu().tolist())
        candidate_ids = [ids[i] for i in sorted(candidate_rows)]
        snapshots = {'client_ids': np.asarray(ids, dtype=np.int32)}
        all_raw, all_rel = [], []
        overlaps, promoted_fractions = [], []
        for row, cid in enumerate(ids):
            role = 'malicious' if cid < args.num_corrupt else 'benign'
            raw_values = matrix[row].abs()
            relative_values = raw_values / denominator
            if not bool(torch.isfinite(relative_values).all()):
                raise ValueError('Non-finite relative feature; check epsilon and update scale')
            raw_top, raw_index = torch.topk(raw_values, k, sorted=False)
            rel_top, rel_index = torch.topk(relative_values, k, sorted=False)
            raw = np.sort(raw_index.cpu().numpy().astype(np.int32))
            rel = np.sort(rel_index.cpu().numpy().astype(np.int32))
            promoted = np.setdiff1d(rel, raw, assume_unique=True)
            common, jaccard = overlap(raw, rel)
            overlaps.append(common / k)
            promoted_fractions.append(promoted.size / k)
            previous = self.previous.get(cid)
            temporal = [float('nan')] * 4
            if previous is not None:
                temporal[0] = overlap(raw, previous['raw'])[1]
                temporal[1] = overlap(rel, previous['rel'])[1]
                surviving, temporal[2] = overlap(promoted, previous['promoted'])
                temporal[3] = (surviving / previous['promoted'].size
                               if previous['promoted'].size else float('nan'))
            random_summary, random_values = self.random_control.compare(
                cid, self.round, raw, promoted, temporal[2])
            for repeat, value in enumerate(random_values):
                self.writers['random_control'].writerow(dict(zip(self.random_fields, [
                    self.round, cid, role, previous['round'] if previous else '', repeat,
                    promoted.size, temporal[2], value,
                ])))
            record = dict(zip(self.base_client_fields, [
                self.round, cid, role, int(row in candidate_rows), float(norms[row]),
                k, common, common / k, jaccard, promoted.size, promoted.size / k,
                float(rel_top.mean()), previous['round'] if previous else '',
                *temporal, int((raw_values == raw_top.min()).sum()),
                int((relative_values == rel_top.min()).sum()),
            ]))
            record.update(random_summary)
            self.writers['client_round'].writerow(record)
            for set_name, indices in [('raw', raw), ('relative', rel), ('promoted', promoted)]:
                counts = np.bincount(np.searchsorted(self.ends, indices, side='right'),
                                     minlength=len(self.layout))
                for layer, count in zip(self.layout, counts):
                    total = indices.size
                    share = int(count) / total if total else float('nan')
                    self.writers['layer_round'].writerow(dict(zip(self.layer_fields, [
                        self.round, cid, role, layer['name'], layer['numel'], set_name,
                        int(count), total, share,
                        share / (layer['numel'] / self.dimension),
                    ])))
            self.previous[cid] = dict(round=self.round, raw=raw, rel=rel, promoted=promoted)
            all_raw.append(raw)
            all_rel.append(rel)
        if self.options.save_indices == 'all' or (
                self.options.save_indices == 'selected' and
                (self.round == 1 or self.round % self.options.snapshot_every == 0
                 or self.round == args.rounds)):
            snapshots.update(raw=np.stack(all_raw), relative=np.stack(all_rel),
                             parameter_count=np.asarray(self.dimension), k=np.asarray(k))
            np.savez_compressed(self.output / 'indices' / f'global_{self.round:03d}.npz',
                                **snapshots)
        if matrix.is_cuda:
            torch.cuda.synchronize(matrix.device)
        elapsed = time.perf_counter() - started
        epsilon_fraction = float((reference.abs() < self.options.epsilon).float().mean())
        self.writers['round'].writerow(dict(zip(self.round_fields, [
            self.round, len(ids), self.dimension, k, '|'.join(map(str, candidate_ids)),
            sum(cid < args.num_corrupt for cid in candidate_ids) / len(candidate_ids),
            int(fallback), float(low), float(high), epsilon_fraction,
            float((reference == 0).float().mean()), np.mean(overlaps),
            np.mean(promoted_fractions), elapsed,
        ])))
        for handle in self.handles:
            handle.flush()
        logging.info('RAMP Top-K | Global %d | clients=%s | K=%d | overlap=%.2f%% | '
                     'promoted=%.2f%% | reference=%s | abs(ref)<eps=%.2f%% | %.2fs',
                     self.round, ids, k, 100 * np.mean(overlaps),
                     100 * np.mean(promoted_fractions), candidate_ids,
                     100 * epsilon_fraction, elapsed)
        # No input tensor, model state, RNG state or native aggregation is modified.

    def close(self):
        for handle in self.handles:
            handle.close()


class MetricsHandler(logging.Handler):
    """Extract per-global-round metrics; ignore the trailing best-results recap."""
    def __init__(self, output):
        super().__init__()
        self.csv_handle, self.writer = csv_writer(Path(output) / 'tables' / 'metrics.csv',
                                             ['round', 'clean_acc', 'asr', 'backdoor_acc'])
        self.round = 0
        self.best_recap = False
        self.pending = {}

    def emit(self, record):
        message = record.getMessage().strip()
        if message.startswith('--------round '):
            self.round = int(message.split()[1])
            self.pending = {'round': self.round}
        if message == 'Best results:':
            self.best_recap = True
        if self.best_recap or not self.round:
            return
        for prefix, key in [('Clean ACC:', 'clean_acc'), ('Attack Success Ratio:', 'asr'),
                            ('Backdoor ACC:', 'backdoor_acc')]:
            if message.startswith(prefix):
                self.pending[key] = float(message.split(':', 1)[1])
        if all(k in self.pending for k in ('round', 'clean_acc', 'asr', 'backdoor_acc')):
            self.writer.writerow(self.pending)
            self.csv_handle.flush()
            self.pending = {'round': self.round}

    def close(self):
        if hasattr(self, 'csv_handle') and not self.csv_handle.closed:
            self.csv_handle.close()
        super().close()
