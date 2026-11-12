"""Focused scientific invariants and completed smoke-batch audit, on the server."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from research.oracle_core import (GROUPS, apply_replacement, layer_random, layout_of,
                                  locate, matched_clean_target_rate, paired_local_train, rng_state, restore_rng, tensor_hash)


def core_checks():
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    import utils
    from aggregation import Aggregation
    from agent import Agent
    import random
    torch.manual_seed(12); np.random.seed(12); random.seed(12)
    layout = [dict(name='a', start=0, numel=8), dict(name='b', start=8, numel=12)]
    promoted = np.array([2, 3, 12, 13, 14])
    raw = np.array([0, 1, 8, 9])
    before_rng = rng_state()
    sampled = layer_random(layout, raw, promoted, 123, 2, 0)
    after_rng = rng_state()
    assert before_rng[0] == after_rng[0]
    assert np.array_equal(before_rng[1][1], after_rng[1][1])
    assert torch.equal(before_rng[2], after_rng[2])
    assert len(np.intersect1d(sampled, raw)) == 0 and len(np.unique(sampled)) == 5
    assert np.count_nonzero(sampled < 8) == 2 and np.count_nonzero(sampled >= 8) == 3
    assert np.array_equal(sampled, layer_random(layout, raw, promoted, 123, 2, 0))
    attack, clean = torch.arange(20, dtype=torch.float32), torch.arange(20, dtype=torch.float32) + 100
    assert apply_replacement(attack, clean, promoted, GROUPS[0]) is attack
    repaired = apply_replacement(attack, clean, promoted, GROUPS[1])
    assert torch.equal(repaired[promoted], clean[promoted])
    other = np.setdiff1d(np.arange(20), promoted)
    assert torch.equal(repaired[other], attack[other]) and torch.equal(attack, torch.arange(20, dtype=torch.float32))
    assert torch.equal(apply_replacement(attack, clean, [], GROUPS[3]), clean)
    zero = apply_replacement(attack, clean, [], GROUPS[1])
    assert torch.equal(zero, attack)
    # Known median and exact full-dimensional update application through native FedAvg.
    updates = {i: torch.ones(20) * i for i in range(20)}
    located, reference = locate(updates, layout, .2, .001)
    assert reference['reference_client_ids'] == list(range(5, 15))
    assert all(x['raw'].size == 4 for x in located.values())
    model = torch.nn.Linear(2, 2, bias=False)
    args = SimpleNamespace(server_lr=.5, device='cpu', aggr='avg', num_corrupt=1, num_agents=2, dense_ratio=.25)
    agg = Aggregation({0: 1, 1: 3}, 4, args)
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    agg.aggregate_updates(model, {0: torch.ones(4)*2, 1: torch.ones(4)*4})
    actual = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(actual, initial + .5 * 3.5)
    # Real native Agent training: same data must produce identical attack/clean updates;
    # replay must leave global RNG and model exactly as attack-only native training.
    class MiniDataset:
        def __init__(self):
            self.data = torch.arange(64*2, dtype=torch.float32).reshape(64, 2)/128
            self.targets = torch.arange(64) % 2
        def __len__(self): return 64
        def __getitem__(self, index): return self.data[index], self.targets[index]
    dataset = MiniDataset()
    args = SimpleNamespace(num_corrupt=1, bs=16, num_workers=0, device='cpu', client_lr=.1,
                           lr_decay=.99, wd=.0001, momentum=0., local_ep=2, attack='badnet', cease_poison=10000)
    agent = Agent.__new__(Agent)
    agent.id, agent.args, agent.is_malicious = 0, args, 1
    agent.data_idxs, agent.clean_backup_dataset = list(range(64)), dataset
    agent.train_dataset = utils.DatasetSplit(dataset, agent.data_idxs)
    agent.train_loader = DataLoader(agent.train_dataset, batch_size=16, shuffle=True, drop_last=True)
    model = torch.nn.Linear(2, 2)
    initial_state = {n: p.clone() for n, p in model.state_dict().items()}
    initial_rng = rng_state()
    expected = Agent.local_train(agent, model, torch.nn.CrossEntropyLoss(), 1, neurotoxin_mask={}).clone()
    expected_rng = rng_state()
    expected_model = {n: p.clone() for n, p in model.state_dict().items()}
    model.load_state_dict(initial_state); restore_rng(initial_rng)
    actual, paired = paired_local_train(Agent.local_train, agent, model, torch.nn.CrossEntropyLoss(), 1,
                                       dict(neurotoxin_mask={}))
    actual_rng = rng_state()
    assert torch.equal(expected, actual) and torch.equal(actual.cpu(), paired)
    assert all(torch.equal(expected_model[n], p) for n, p in model.state_dict().items())
    assert expected_rng[0] == actual_rng[0] and torch.equal(expected_rng[2], actual_rng[2])
    assert np.array_equal(expected_rng[1][1], actual_rng[1][1])
    if expected_rng[3] is not None:
        assert all(torch.equal(a, b) for a, b in zip(expected_rng[3], actual_rng[3]))
    args.target_class = 1
    model.eval()
    clean_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    baseline_rng = rng_state()
    rate = matched_clean_target_rate(model, clean_loader, args)
    baseline_after = rng_state()
    assert torch.equal(baseline_rng[2], baseline_after[2]) and baseline_rng[0] == baseline_after[0]
    assert 0 <= rate <= 1
    print('PASS: layer budgets, isolated random sampler, sparse/full/empty replacement, native weighted FedAvg, native paired replay and RNG/model preservation')


def batch_checks(batch, zero_poison):
    from research.oracle_report import compare_batch
    meta = []
    hashes = []
    for group in GROUPS:
        path = batch / group
        m = json.loads((path / 'run.json').read_text())
        assert m['status'] == 'completed' and m['observed_rounds'] == m['settings']['rounds']
        with (path / 'tables/client_round.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == m['settings']['rounds'] * m['settings']['clients']
        assert all(int(r['selected_count']) == 0 and float(r['replacement_l2']) == 0 for r in rows if r['role'] == 'benign')
        if group == GROUPS[0]:
            assert all(int(r['selected_count']) == 0 and float(r['replacement_l2']) == 0 for r in rows)
        if group == GROUPS[3]:
            assert all(r['clean_update_sha256'] == r['applied_update_sha256'] for r in rows if r['role'] == 'malicious' and r['clean_update_sha256'])
        if group in GROUPS[1:3]:
            assert all(r['promoted_count'] == r['selected_count'] for r in rows if r['role'] == 'malicious')
        with (path / 'tables/layer_round.csv').open() as handle:
            assert all(r['promoted_count'] == r['random_count'] for r in csv.DictReader(handle))
        with (path / 'tables/metrics.csv').open() as handle:
            metrics = list(csv.DictReader(handle))
        assert len(metrics) == m['settings']['rounds']
        assert all(int(r['participants']) == m['settings']['clients'] for r in metrics)
        assert all(0 <= float(r[key]) <= 1 for r in metrics for key in ['clean_acc', 'asr', 'clean_target_rate', 'triggered_original_acc'])
        assert (path / 'snapshots' / f'global_{m["settings"]["rounds"]:03d}.pt').exists()
        if zero_poison:
            assert all(float(r['attack_clean_l2']) == 0 for r in rows if r['role'] == 'malicious')
        meta.append(m); hashes.append(m['final_model_sha256'])
    for field in ['initial_model_sha256', 'partition_sha256', 'poison_plan_sha256',
                  'first_round_attack_sha256', 'first_round_clean_sha256']:
        assert all(m[field] == meta[0][field] for m in meta), field
    if zero_poison:
        assert len(set(hashes)) == 1, 'No-poison repair should produce identical four trajectories'
    else:
        assert len(set(hashes)) > 1, 'Poisoned interventions should affect the training trajectory'
    compare_batch(batch)
    print('PASS: completed four-group batch, counts/roles/budgets, all clients retained, same initial/data/poison/paired updates, report generation' +
          ('; zero-poison trajectories identical' if zero_poison else '; poisoned trajectories differ'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--zero-poison', action='store_true')
    options = parser.parse_args()
    if options.batch:
        batch_checks(options.batch, options.zero_poison)
    else:
        core_checks()
