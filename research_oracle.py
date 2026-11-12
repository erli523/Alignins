#!/usr/bin/env python3
"""Four independent Oracle repair trajectories; native Alignins files stay intact."""
import argparse
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEFAULT_BATCH = ROOT / 'research_results/RAMP_oracle/cifar10_resnet9/noniid_beta0.5/clients20_malicious4_20pct/badnet/oracle_v1'
GROUPS = ('A_no_repair', 'B_promoted_repair', 'C_random_repair', 'D_full_clean')


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def native_args(options):
    return ['--data', 'cifar10', '--model', 'resnet9', '--attack', 'badnet', '--aggr', 'avg',
            '--num_agents', str(options.clients), '--num_corrupt', str(options.corrupt),
            '--agent_frac', '1.0', '--poison_frac', str(options.poison_frac),
            '--non_iid', '--beta', '0.5', '--rounds', str(options.rounds),
            '--local_ep', str(options.local_ep), '--client_lr', '0.1', '--server_lr', '1.0',
            '--lr_decay', '0.99', '--bs', '64', '--momentum', '0.0', '--wd', '0.0001',
            '--target_class', '7', '--pattern_type', 'plus', '--snap', '1', '--num_workers', '0',
            '--device', options.device]


class Tee:
    def __init__(self, original, handle):
        self.original, self.handle = original, handle
    def write(self, message):
        self.original.write(message); self.handle.write(message); self.handle.flush()
        return len(message)
    def flush(self):
        self.original.flush()
        if not self.handle.closed:
            self.handle.flush()
    def isatty(self):
        return False


def train(options):
    import numpy as np
    import torch
    from research.oracle_core import (apply_replacement, cpu_model_state, layer_random,
                                      layout_of, locate, matched_clean_target_rate, paired_local_train, tensor_hash)
    from research.oracle_report import make_report, compare_batch
    if options.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(options.device))
        torch.empty(1, device=options.device)
    output = options.output.resolve()
    batch = output.parent
    settings = {k: v for k, v in vars(options).items() if k not in ('output', 'group', 'device', 'mode', 'batch_root', 'lane')}
    settings['native_arguments_except_device'] = native_args(options)[:-2]
    sources = [ROOT / 'research_oracle.py', ROOT / 'run_ramp_oracle_gpu0.sh', ROOT / 'run_ramp_oracle_gpu1.sh',
               ROOT / 'RAMP_ORACLE_README.md', ROOT / 'research/oracle_core.py', ROOT / 'research/oracle_report.py',
               *sorted((ROOT / 'src').glob('*.py'))]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()}
    signature = hashlib.sha256(json.dumps(dict(settings=settings, sources=source_hashes), sort_keys=True).encode()).hexdigest()
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / '.setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = batch / 'batch.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['signature'] != signature:
                raise ValueError('Batch settings/source differ. Use a new BATCH_NAME or RESULT_ROOT.')
        else:
            save_json(manifest, dict(signature=signature, settings=settings, created_at=time.strftime('%Y-%m-%d %H:%M:%S %z')))
        # One atomic claim; never overwrite or silently rerun an existing group.
        output.mkdir(exist_ok=False)
    metadata = dict(status='initializing', group=options.group, settings=settings, signature=signature,
                    device=options.device, started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                    seed=0, source_sha256=source_hashes, native_argv=native_args(options),
                    formula='abs(delta)/(abs(reference)+epsilon)',
                    aggregation='native data-size weighted FedAvg; all participating clients; no filtering',
                    oracle='true malicious IDs select repair targets; paired clean updates provide replacement values',
                    random_control='one independent layer/size-matched draw per client per round, outside raw Top-K',
                    clean_target_baseline='separate non-target clean pass with ASR-matched ordering/batches; RNG preserved',
                    poison_fraction_semantics='fraction of non-target-class eligible local examples',
                    snapshot_rounds=sorted({options.rounds, *options.snapshot_rounds}), observed_rounds=0)
    for directory in ('tables', 'figures', 'snapshots', 'source_snapshot'):
        (output / directory).mkdir()
    for source in sources:
        if source.exists():
            destination = output / 'source_snapshot' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    save_json(output / 'run.json', metadata)
    out0, err0 = sys.stdout, sys.stderr
    console = (output / 'console.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(out0, console), Tee(err0, console)
    handles, writers = {}, {}
    schemas = dict(
        metrics=['round', 'clean_acc', 'asr', 'clean_target_rate', 'triggered_original_acc', 'participants',
                 'malicious_participants', 'repaired_clients', 'selected_coordinates', 'aggregate_delta_l2', 'elapsed_seconds'],
        client_round=['round', 'client_id', 'role', 'reference_candidate', 'param_l2', 'relative_score',
                      'k', 'promoted_count', 'selected_count', 'random_promoted_overlap', 'aggregation_weight',
                      'attack_clean_l2', 'replacement_l2', 'oracle_error_energy_removed_fraction',
                      'attack_update_sha256', 'clean_update_sha256', 'applied_update_sha256'],
        layer_round=['round', 'client_id', 'layer', 'numel', 'promoted_count', 'random_count'])
    for name, fields in schemas.items():
        handles[name] = (output / 'tables' / f'{name}.csv').open('w', newline='', encoding='utf-8')
        writers[name] = csv.DictWriter(handles[name], fieldnames=fields)
        writers[name].writeheader(); handles[name].flush()
    started, failure = time.perf_counter(), None
    try:
        sys.path.insert(0, str(ROOT / 'src'))
        import utils
        from agent import Agent
        from aggregation import Aggregation
        from torch.nn.utils import parameters_to_vector
        original_init, original_train = Agent.__init__, Agent.local_train
        original_aggregate, original_datasets = Aggregation.aggregate_updates, utils.get_datasets
        context = dict(round=0, clean={}, layout=None, partition={}, poison={}, evaluations={}, round_stats={})

        def setup_logging(args):
            if args.aggr != 'avg' or args.attack != 'badnet' or args.num_workers != 0:
                raise ValueError('This study requires avg, badnet and num_workers=0')
            metadata.update(training=vars(args).copy(), status='running', torch_version=torch.__version__,
                            numpy_version=np.__version__, gpu_name=torch.cuda.get_device_name(args.device)
                            if str(args.device).startswith('cuda') else 'cpu')
            save_json(output / 'run.json', metadata)
            logger = logging.getLogger()
            for handler in list(logger.handlers):
                logger.removeHandler(handler); handler.close()
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(handler)
            logging.info('Oracle group=%s; output=%s', options.group, output)
            logging.info('NO CLIENT FILTERING. All clients enter data-size weighted FedAvg.')
            logging.info('Every round intervention; oracle malicious IDs; paired clean values; settings=%s', args)
            return str(output)

        def limited_datasets(data):
            train_data, test_data = original_datasets(data)
            for dataset, limit in ((train_data, options.train_limit), (test_data, options.test_limit)):
                if limit:
                    dataset.data = dataset.data[:limit].copy()
                    dataset.targets = dataset.targets[:limit].clone()
            return train_data, test_data

        def init_agent(agent, *args, **kwargs):
            original_init(agent, *args, **kwargs)
            indices = list(map(int, agent.train_dataset.idxs))
            context['partition'][agent.id] = indices
            if agent.id < agent.args.num_corrupt:
                clean_targets = agent.clean_backup_dataset.targets
                attack_targets = agent.train_dataset.dataset.targets
                changed = [i for i in indices if int(clean_targets[i]) != int(attack_targets[i])]
                context['poison'][agent.id] = changed
            if len(context['partition']) == agent.args.num_agents:
                metadata['partition_sha256'] = hashlib.sha256(json.dumps(context['partition'], sort_keys=True).encode()).hexdigest()
                metadata['poison_plan_sha256'] = hashlib.sha256(json.dumps(context['poison'], sort_keys=True).encode()).hexdigest()
                save_json(output / 'run.json', metadata)
                save_json(output / 'tables' / 'data_partition.json', dict(partition=context['partition'], poisoned_indices=context['poison']))

        def local_train(agent, model, criterion, round=None, **kwargs):
            if context['layout'] is None:
                context['layout'], context['dimension'] = layout_of(model)
                initial = parameters_to_vector(list(model.state_dict().values()))
                metadata.update(parameter_layout=context['layout'], parameter_count=context['dimension'],
                                initial_model_sha256=tensor_hash(initial))
                save_json(output / 'run.json', metadata)
            if agent.id < agent.args.num_corrupt:
                logging.info('Global %d | Client %d | paired attack and clean local training', round, agent.id)
                attack, clean = paired_local_train(original_train, agent, model, criterion, round, kwargs)
                context['clean'][agent.id] = clean
                return attack
            return original_train(agent, model, criterion, round, **kwargs)

        @torch.no_grad()
        def aggregate(aggregator, model, updates):
            context['round'] += 1
            rnd, args = context['round'], aggregator.args
            ids, layout = list(updates), context['layout']
            malicious = [c for c in ids if c < args.num_corrupt]
            benign = [c for c in ids if c >= args.num_corrupt]
            if set(context['clean']) != set(malicious):
                raise ValueError('Paired clean updates missing or stale')
            located, reference = locate(updates, layout, options.topk_ratio, options.epsilon, options.reference)
            logging.info('Global %d | participants=%s | malicious=%s | benign=%s', rnd, ids, malicious, benign)
            logging.info('Global %d | reference=%s | k=%d | fallback=%s | group=%s | filtering=OFF',
                         rnd, reference['reference_client_ids'], reference['k'], reference['reference_fallback'], options.group)
            total_data = sum(aggregator.agent_data_sizes[c] for c in ids)
            applied, indices_saved, selected_total = dict(updates), {}, 0
            snapshot = rnd in metadata['snapshot_rounds']
            if rnd == 1:
                metadata['first_round_attack_sha256'] = {str(c): tensor_hash(updates[c]) for c in ids}
                metadata['first_round_clean_sha256'] = {str(c): tensor_hash(context['clean'][c]) for c in malicious}
            for cid in ids:
                info = located[cid]
                record = dict(round=rnd, client_id=cid, role='malicious' if cid in malicious else 'benign',
                              reference_candidate=info['reference_candidate'], param_l2=info['param_l2'],
                              relative_score=info['relative_score'], k=reference['k'], promoted_count=info['promoted'].size,
                              selected_count=0, random_promoted_overlap='',
                              aggregation_weight=aggregator.agent_data_sizes[cid]/total_data,
                              attack_clean_l2='', replacement_l2=0., oracle_error_energy_removed_fraction='',
                              attack_update_sha256=tensor_hash(updates[cid]) if snapshot or rnd == 1 else '',
                              clean_update_sha256='', applied_update_sha256='')
                if cid in malicious:
                    clean = context['clean'][cid].to(updates[cid].device)
                    promoted = info['promoted']
                    random_set = layer_random(layout, info['raw'], promoted, options.random_seed, rnd, cid)
                    for layer in layout:
                        start, end = layer['start'], layer['start'] + layer['numel']
                        p_count = int(np.count_nonzero((promoted >= start) & (promoted < end)))
                        r_count = int(np.count_nonzero((random_set >= start) & (random_set < end)))
                        if p_count != r_count:
                            raise AssertionError('Random and promoted layer budgets differ')
                        writers['layer_round'].writerow(dict(round=rnd, client_id=cid, layer=layer['name'],
                                                              numel=layer['numel'], promoted_count=p_count, random_count=r_count))
                    if options.group == GROUPS[1]:
                        selected = promoted
                    elif options.group == GROUPS[2]:
                        selected = random_set
                    else:
                        selected = np.empty(0, np.int64)
                    applied[cid] = apply_replacement(updates[cid], clean, selected, options.group)
                    if options.group == GROUPS[3]:
                        count = context['dimension']
                    else:
                        count = selected.size
                    error = updates[cid] - clean
                    change = applied[cid] - updates[cid]
                    energy = float(error.square().sum())
                    removed_energy = float(change.square().sum())
                    record.update(selected_count=count, random_promoted_overlap=int(np.intersect1d(random_set, promoted).size),
                                  attack_clean_l2=float(error.norm()), replacement_l2=float(change.norm()),
                                  oracle_error_energy_removed_fraction=removed_energy/energy if energy else 0.,
                                  clean_update_sha256=tensor_hash(clean) if snapshot or rnd == 1 else '',
                                  applied_update_sha256=tensor_hash(applied[cid]) if snapshot or rnd == 1 else '')
                    selected_total += count
                    indices_saved[str(cid)] = dict(raw=torch.as_tensor(info['raw'].copy()),
                                                  promoted=torch.as_tensor(promoted.copy()),
                                                  random=torch.as_tensor(random_set.copy()),
                                                  selected=torch.as_tensor(selected.copy()),
                                                  full_replacement=options.group == GROUPS[3])
                logging.info('Global %d | Client %d %-9s | L2=%.6g | relative_score=%.6g | reference=%s | promoted=%d | replaced=%d | weight=%.6f | KEPT',
                             rnd, cid, record['role'], record['param_l2'], record['relative_score'], record['reference_candidate'],
                             record['promoted_count'], record['selected_count'], record['aggregation_weight'])
                writers['client_round'].writerow(record)
            if snapshot:
                payload = dict(round=rnd, group=options.group, global_before=cpu_model_state(model),
                               attack_updates={c: updates[c].detach().cpu() for c in malicious},
                               clean_updates=context['clean'].copy(), indices=indices_saved,
                               reference=reference, layout=layout,
                               client_weights={c: aggregator.agent_data_sizes[c]/total_data for c in ids},
                               note='Only malicious updates saved; final group model is final_model.pt')
                torch.save(payload, output / 'snapshots' / f'global_{rnd:03d}.pt')
            # Exactly the native weighted FedAvg; no AlphaGuard keep mask or weight changes.
            before = parameters_to_vector(list(model.state_dict().values())).detach().clone()
            result = original_aggregate(aggregator, model, applied)
            delta_norm = float((parameters_to_vector(list(model.state_dict().values())).detach() - before).norm())
            context['round_stats'] = dict(participants=len(ids), malicious_participants=len(malicious),
                                          repaired_clients=len(malicious) if options.group != GROUPS[0] else 0,
                                          selected_coordinates=selected_total, aggregate_delta_l2=delta_norm)
            context['clean'].clear()
            for handle in handles.values():
                handle.flush()
            return result

        @torch.no_grad()
        def evaluate(model, criterion, loader, args, round, num_classes=10, sent140_flag=False):
            model.eval()
            correct, total, target_pred, non_target_total = 0, 0, 0, 0
            # Preserve the three native evaluation passes and their RNG consumption.
            # Do not restore RNG here: native DataLoader iterator seeds consume it too.
            for inputs, labels in loader:
                labels = labels.to(args.device)
                predicted = model(inputs.to(args.device)).argmax(dim=1)
                correct += int((predicted == labels).sum())
                total += labels.numel()
                non_target = labels != args.target_class
                target_pred += int(((predicted == args.target_class) & non_target).sum())
                non_target_total += int(non_target.sum())
            accuracy = correct / total
            entries = context['evaluations'].setdefault(round, [])
            if not entries:
                target_rate = matched_clean_target_rate(model, loader, args)
            else:
                target_rate = float('nan')
            entries.append((accuracy, target_rate))
            if len(entries) == 3:
                record = dict(round=round, clean_acc=entries[0][0], asr=entries[1][0],
                              clean_target_rate=entries[0][1], triggered_original_acc=entries[2][0],
                              elapsed_seconds=time.perf_counter()-started, **context['round_stats'])
                writers['metrics'].writerow(record); handles['metrics'].flush()
                metadata['observed_rounds'] = round
                save_json(output / 'run.json', metadata)
                logging.info('Global %d | clean target prediction rate=%.6f | repaired_clients=%d | replaced_coordinates=%d',
                             round, record['clean_target_rate'], record['repaired_clients'], record['selected_coordinates'])
                del context['evaluations'][round]
            elif len(entries) > 3:
                raise AssertionError('Unexpected native evaluation call count')
            return accuracy

        utils.setup_logging, utils.get_datasets, utils.get_loss_n_accuracy = setup_logging, limited_datasets, evaluate
        Agent.__init__, Agent.local_train, Aggregation.aggregate_updates = init_agent, local_train, aggregate
        os.chdir(ROOT / 'src')
        sys.argv = [str(ROOT / 'src/federated.py'), *native_args(options)]
        namespace = runpy.run_path(str(ROOT / 'src/federated.py'), run_name='__main__')
        if metadata['observed_rounds'] != options.rounds:
            raise AssertionError('Requested rounds were not fully evaluated')
        final_state = cpu_model_state(namespace['global_model'])
        torch.save(dict(round=options.rounds, group=options.group, state_dict=final_state), output / 'final_model.pt')
        metadata.update(status='completed', final_model_sha256=tensor_hash(torch.cat([v.flatten() for v in final_state.values()])))
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc(); failure = exc
    finally:
        for handle in handles.values():
            handle.close()
        metadata.update(finished_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), elapsed_seconds=time.perf_counter()-started)
        save_json(output / 'run.json', metadata)
        try:
            make_report(output)
            metadata['report_status'] = 'generated'
            save_json(output / 'run.json', metadata)
            compare_batch(batch)
        except Exception as exc:
            metadata['report_status'] = f'failed: {exc}'
            save_json(output / 'run.json', metadata)
            traceback.print_exc()
            if failure is None:
                failure = exc
        print(f'Oracle status={metadata["status"]}; results={output}', flush=True)
        sys.stdout, sys.stderr = out0, err0
        console.close()
    if failure is not None:
        raise SystemExit(130 if isinstance(failure, KeyboardInterrupt) else 1)


def common(parser):
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local-ep', type=int, default=2)
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--corrupt', type=int, default=4)
    parser.add_argument('--poison-frac', type=float, default=.3)
    parser.add_argument('--topk-ratio', type=float, default=.001)
    parser.add_argument('--epsilon', type=float, default=.001)
    parser.add_argument('--reference', choices=['median', 'mean'], default='median')
    parser.add_argument('--random-seed', type=int, default=20261112)
    parser.add_argument('--snapshot-rounds', type=int, nargs='*', default=[])
    parser.add_argument('--train-limit', type=int, default=0, help='Validation only; 0 uses full dataset')
    parser.add_argument('--test-limit', type=int, default=0, help='Validation only; 0 uses full test set')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    one = sub.add_parser('train'); common(one)
    one.add_argument('--group', choices=GROUPS, required=True)
    one.add_argument('--output', type=Path, required=True)
    lane = sub.add_parser('lane'); common(lane)
    lane.add_argument('--lane', type=int, choices=[0, 1], required=True)
    lane.add_argument('--batch-root', type=Path, default=DEFAULT_BATCH)
    report = sub.add_parser('compare'); report.add_argument('batch_root', type=Path)
    options = parser.parse_args()
    if options.mode == 'compare':
        from research.oracle_report import make_report, compare_batch
        for group in GROUPS:
            if (options.batch_root / group / 'run.json').exists():
                make_report(options.batch_root / group)
        compare_batch(options.batch_root)
        return
    if (options.rounds < 1 or options.local_ep < 1 or not 0 <= options.corrupt < options.clients
            or not 0 <= options.poison_frac <= 1 or not 0 < options.topk_ratio <= 1
            or not math.isfinite(options.epsilon) or options.epsilon <= 0
            or options.random_seed < 0 or options.train_limit < 0 or options.test_limit < 0
            or any(r < 1 or r > options.rounds for r in options.snapshot_rounds)):
        parser.error('Invalid experiment settings')
    if options.mode == 'train':
        train(options)
        return
    groups = GROUPS[:2] if options.lane == 0 else GROUPS[2:]
    # Reject an existing target before starting either group, not half way through a lane.
    for group in groups:
        if (options.batch_root / group).exists():
            parser.error(f'Refuse to overwrite {options.batch_root / group}; choose a new BATCH_NAME / RESULT_ROOT')
    print(f'Lane {options.lane}; device={options.device}; groups={groups}; batch={options.batch_root}', flush=True)
    for group in groups:
        command = [sys.executable, '-u', str(ROOT / 'research_oracle.py'), 'train',
                   '--group', group, '--output', str(options.batch_root / group)]
        for key in ('device', 'rounds', 'local_ep', 'clients', 'corrupt', 'poison_frac', 'topk_ratio',
                    'epsilon', 'reference', 'random_seed', 'train_limit', 'test_limit'):
            command += ['--' + key.replace('_', '-'), str(getattr(options, key))]
        if options.snapshot_rounds:
            command += ['--snapshot-rounds', *map(str, options.snapshot_rounds)]
        subprocess.run(command, check=True, cwd=ROOT)
    print(f'Lane finished. Open {options.batch_root / "README.md"}', flush=True)


if __name__ == '__main__':
    main()
