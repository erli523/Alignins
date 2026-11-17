"""Lean multi-attack runner using the existing RAMP AG Alpha-Zero implementation.

Run from the project root: python src/ramp_attack_eval.py --help
No source copies, model checkpoints, or coordinate dumps are written.
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
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import ramp_ag_alpha_zero as core
from agent import Agent

ATTACKS = ['badnet', 'DBA', 'neurotoxin', 'pgd', 'dynamic_trigger', 'altermin', 'model_replacement']
METRICS = ['round', 'clean_acc', 'asr', 'clean_target_rate', 'triggered_original_acc',
           'elapsed_seconds', 'round_seconds', 'selected_coordinates', 'aggregate_delta_l2']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['resnet9','vgg9'],default='resnet9')
    p.add_argument('--beta',type=float,default=0.7)
    p.add_argument('--data', choices=['cifar10', 'cifar100'], default='cifar10')
    p.add_argument('--attack', choices=ATTACKS, default='DBA')
    p.add_argument('--config', choices=['A', 'B'], default='B')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounds', type=int)
    p.add_argument('--client_lr', type=float)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max_batches', type=int, default=0, help='Validation only; 0 = full client dataset')
    p.add_argument('--test_limit', type=int, default=0, help='Validation only; 0 = full test set')
    p.add_argument('--trigger_outter_epochs', type=int, default=50)
    p.add_argument('--dm_adv_model_count', type=int, default=3)
    p.add_argument('--dm_adv_epochs', type=int, default=2)
    p.add_argument('--trigger_optim_interval', type=int, default=5)
    p.add_argument('--trigger_lr', type=float, default=0.1)
    a = p.parse_args(argv)
    a.rounds = a.rounds if a.rounds is not None else (100 if a.data == 'cifar10' else 150)
    a.client_lr = a.client_lr if a.client_lr is not None else (0.1 if a.data == 'cifar10' else 0.05)
    if a.rounds < 1 or a.client_lr <= 0 or a.threads < 1 or a.max_batches < 0 or a.test_limit < 0:
        p.error('Invalid training settings')
    if a.trigger_outter_epochs < 1 or a.trigger_optim_interval < 1 or a.dm_adv_model_count < 0 or a.dm_adv_epochs < 1:
        p.error('Invalid dynamic trigger settings')
    a.aggr, a.num_target = 'avg', 10 if a.data == 'cifar10' else 100
    a.num_agents, a.num_corrupt, a.agent_frac = 20, 4, 1.0
    a.poison_frac, a.local_ep, a.bs = 0.3, 2, 64
    a.server_lr, a.lr_decay, a.momentum, a.wd = 1.0, 0.99, 0.0, 1e-4
    a.non_iid, a.target_class, a.pattern_type = True, 7, 'plus'
    a.ag_topk_ratio, a.topk_ratio = 0.001, 0.05 if a.config == 'A' else 0.04
    a.penalty_strength = 5.0 if a.config == 'A' else 7.0
    a.alpha_threshold, a.epsilon = 1.0, 0.001
    a.reference, a.l2_lower, a.l2_upper, a.min_reference = 'median', 25., 75., 8
    a.zero_scope, a.cease_poison, a.num_workers = 'alphaguard_detected', 100000, 0
    a.dense_ratio = 0.25
    a.trigger_size, a.dm_adv_K, a.noise_loss_lambda = 5, 10, 0.1
    a.altermin_attack_mode, a.altermin_source_label, a.altermin_target_label = 'targeted', 3, None
    a.altermin_poisoned_cnt, a.altermin_boosting_factor, a.altermin_rho = 100, 1.5, 1e-4
    a.altermin_benign_epochs, a.altermin_malicious_epochs = a.local_ep, 1
    # Preserve federated.py's CLI defaults: effective update scaling = 50 * .02 = 1.
    a.model_replacement_scaling_factor, a.model_replacement_alpha = 50., 0.5
    a.model_replacement_constrain_weight, a.model_replacement_target_label = 0.02, None
    # Native agent.py: CIFAR10 uses .1 * ||w||; CIFAR100 uses 1 * ||w||.
    a.pgd_radius_multiplier = 0.1 if a.data == 'cifar10' else 1.0
    a.output = a.output.resolve()
    return a


class AttackEvalSet(Dataset):
    """Native static trigger, current client-0 dynamic trigger, or clean source images."""
    def __init__(self, dataset, indices, args, agents, original_labels=False):
        self.dataset, self.indices, self.args, self.agents = dataset, indices, args, agents
        self.original_labels = original_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.args.attack == 'dynamic_trigger':
            attacker = self.agents[0].dynamic_trigger_attacker
            t, m = attacker.trigger.detach().cpu()[0], attacker.mask.detach().cpu()[0]
            image = t * m + image * (1 - m)
        return image, label if self.original_labels else self.args.target_class


def evaluation_loaders(test, args, agents):
    indices = (test.targets != args.target_class).nonzero().flatten().tolist()
    def loader(ds):
        return DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)
    clean_non_target = loader(Subset(test, indices))
    if args.attack in ['dynamic_trigger', 'altermin']:
        if args.attack == 'altermin':
            indices = (test.targets == args.altermin_source_label).nonzero().flatten().tolist()
        asr = loader(AttackEvalSet(test, indices, args, agents))
        original = loader(AttackEvalSet(test, indices, args, agents, original_labels=True))
    else:
        attacked, original_data = copy.deepcopy(test), copy.deepcopy(test)
        core.utils.poison_dataset(attacked, args, indices, poison_all=True)
        core.utils.poison_dataset(original_data, args, indices, poison_all=True, modify_label=False)
        asr, original = loader(core.utils.DatasetSplit(attacked, indices)), loader(core.utils.DatasetSplit(original_data, indices))
    return dict(clean=loader(test), clean_non_target=clean_non_target, asr=asr, triggered=original)


def init_altermin(agent, args):
    """Use only this client's clean local training examples; never use evaluation data."""
    indices = [int(i) for i in agent.data_idxs
               if int(agent.clean_backup_dataset.targets[i]) == args.altermin_source_label]
    if not indices:
        raise ValueError(f'AlterMin client {agent.id} has no local source-class training examples')
    dataset = core.utils.DatasetSplit(agent.clean_backup_dataset, indices)
    agent.altermin_attacker.init_poisoned_loader(dataset)
    return len(agent.altermin_attacker.poisoned_loader.dataset)


def neurotoxin_mask(delta, model, ratio):
    """Native per-layer bottom-|update| mask from the previous aggregate; round 1 is unmasked."""
    masks, offset = {}, 0
    for name, parameter in model.state_dict().items():
        values = delta[offset:offset + parameter.numel()].abs()
        mask = torch.zeros_like(values)
        indices = torch.topk(-values, int(values.numel() * ratio)).indices
        mask[indices] = 1
        masks[name] = mask.reshape(parameter.shape)
        offset += parameter.numel()
    return masks


def csv_write(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def update_outputs(output, rows, args, status):
    milestones = []
    for end in sorted({100, 150, 200, args.rounds}):
        match = [r for r in rows if r['round'] == end]
        if not match:
            continue
        last = match[0]
        recent = [r for r in rows if max(1, end-9) <= r['round'] <= end]
        milestones.append(dict(round=end, clean_acc_pct=100*last['clean_acc'], asr_pct=100*last['asr'],
                               window_start=recent[0]['round'], window_count=len(recent),
                               clean_acc_last10_pct=100*np.mean([r['clean_acc'] for r in recent]),
                               asr_last10_pct=100*np.mean([r['asr'] for r in recent])))
    fields = ['round','clean_acc_pct','asr_pct','window_start','window_count','clean_acc_last10_pct','asr_last10_pct']
    csv_write(output/'milestones.csv', milestones, fields)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(11,4), layout='constrained')
    for ax,key,title in zip(axes,['clean_acc','asr'],['Clean ACC','ASR']):
        ax.plot([r['round'] for r in rows],[100*r[key] for r in rows])
        for end in [100,150,200]:
            if end <= args.rounds:
                ax.axvline(end,color='gray',linestyle='--',alpha=.35)
        ax.set(title=title,xlabel='Global round',ylabel='%',ylim=(0,100))
        ax.grid(alpha=.2)
    fig.suptitle(f'{args.data} {args.model} beta={args.beta} | {args.attack} | {args.config} zero={args.topk_ratio:g}, penalty={args.penalty_strength:g} | {status}')
    fig.savefig(output/'curves.png',dpi=160)
    plt.close(fig)


def train(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='initializing', observed_rounds=0, settings=vars(args).copy(),
                    started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')},
                    attack_notes=dict(
                        neurotoxin='Previous defended aggregate, layerwise smallest 25%; first round has no mask.',
                        pgd='Native end-of-local-epoch full-update projection; radius multiplier is data-specific.',
                        dynamic_trigger='Native optimized trigger in normalized input space; evaluate current client-0 trigger, as native federated.py.',
                        altermin='Local TRAIN source class 3 to target 7, max100 examples/client; 2 benign epochs plus 1 malicious phase; no test data used for training.',
                        model_replacement='Native CLI defaults: gamma50 * constrain_weight0.02 = effective scaling1; alpha0.5. Not a strong scaling attack.'),
                    aggregation='Existing RAMPAGAlphaZero; original detection; detected-client coordinate masking; native penalties normalized over all clients.',
                    storage='Parameters, log, metrics, milestones, curves only; hashes are provenance, no source copies.')
    core.save_json(output/'config.json', metadata)
    started = time.perf_counter()
    rows, failure = [], None
    original_out, original_err = sys.stdout, sys.stderr
    with (output/'training.log').open('w',encoding='utf-8',buffering=1) as log:
        sys.stdout = core.Tee(original_out, log)
        sys.stderr = core.Tee(original_err, log)
        root = logging.getLogger()
        root.handlers.clear()
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            os.chdir(Path(__file__).parent)
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            if args.device.startswith('cuda'):
                torch.cuda.set_device(torch.device(args.device))
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
            logging.info('RAMP attack evaluation | settings=%s',args)
            train_data, test_data = core.utils.get_datasets(args.data)
            if args.test_limit:
                test_data.data = test_data.data[:args.test_limit].copy()
                test_data.targets = test_data.targets[:args.test_limit].clone()
            groups = core.utils.distribute_data_dirichlet(train_data,args)
            if min(map(len,groups.values())) < args.bs:
                raise ValueError('Client dataset smaller than one batch')
            metadata['partition_sha256'] = hashlib.sha256(json.dumps({int(k):list(map(int,v)) for k,v in groups.items()},sort_keys=True).encode()).hexdigest()
            # Construct evaluation datasets before model initialization, as in the BadNet study.
            loaders = None if args.attack in ['dynamic_trigger','altermin'] else evaluation_loaders(test_data,args,[])
            if args.model == 'vgg9':
                from vgg import VGG
                model = VGG('VGG9',num_classes=args.num_target)
                # Match ResNet9's buffer-free BN policy; no running-state coordinates enter B.
                for layer in model.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
                        layer.track_running_stats=False
                        layer.running_mean=None; layer.running_var=None; layer.num_batches_tracked=None
                model=model.to(args.device)
            else:
                model = core.models.get_model(args.data,args).to(args.device)
            metadata['batchnorm_policy']='track_running_stats=False for both models; B processes trainable updates only'
            logging.info('Model=%s | BN running statistics disabled | beta=%s',args.model,args.beta)
            metadata['initial_model_sha256'] = core.tensor_hash(core.model_vector(model))
            if model(torch.zeros(2,3,32,32,device=args.device)).shape != (2,args.num_target):
                raise ValueError('Wrong classifier output shape')
            # Shape check consumes no RNG with this buffer-free ResNet9.
            method = core.RAMPAGAlphaZero(args,model)
            agents, sizes, attack_counts = [], {}, {}
            for cid in range(args.num_agents):
                agent = Agent(cid,args,train_data,groups[cid],backdoor_train_dataset=None)
                agent.is_malicious = int(cid < args.num_corrupt)
                if args.attack == 'altermin' and agent.is_malicious:
                    attack_counts[cid] = init_altermin(agent,args)
                if args.max_batches:
                    agent.train_loader = DataLoader(Subset(agent.train_dataset, range(min(len(agent.train_dataset),args.max_batches*args.bs))),
                                                    batch_size=args.bs,shuffle=True,num_workers=0,drop_last=True)
                agents.append(agent); sizes[cid] = agent.n_data
            if loaders is None:
                loaders = evaluation_loaders(test_data,args,agents)
            if args.attack == 'dynamic_trigger':
                # Diagnostic trigger validation uses local training data; held-out ASR never drives optimization.
                for agent in agents[:args.num_corrupt]:
                    agent.dynamic_trigger_attacker.test_loader = None
            metadata.update(status='running', local_data_sizes=sizes, altermin_local_attack_samples=attack_counts,
                            parameter_count=method.dimension)
            core.save_json(output/'config.json',metadata)
            criterion, mask = torch.nn.CrossEntropyLoss().to(args.device), {}
            with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as metrics:
                writer = csv.DictWriter(metrics,fieldnames=METRICS); writer.writeheader(); metrics.flush()
                for rnd in range(1,args.rounds+1):
                    round_start = time.perf_counter()
                    baseline = core.model_vector(model).clone()
                    ids = sorted(np.random.choice(args.num_agents, args.num_agents, replace=False).tolist())
                    logging.info('Global %d | participants=%s | malicious=%s | benign=%s | local_lr=%.8g',rnd,ids,
                                 [i for i in ids if i<args.num_corrupt],[i for i in ids if i>=args.num_corrupt],args.client_lr*args.lr_decay**rnd)
                    updates = {}
                    for cid in ids:
                        updates[cid] = agents[cid].local_train(model,criterion,rnd,neurotoxin_mask=mask)
                        if not torch.isfinite(updates[cid]).all():
                            raise ValueError(f'Nonfinite update client={cid}')
                        core.utils.vector_to_model(baseline.clone(),model)
                    if args.attack == 'dynamic_trigger':
                        metadata['trigger_changed'] = [not torch.equal(a.dynamic_trigger_attacker.trigger,a.dynamic_trigger_attacker.trigger0) for a in agents[:args.num_corrupt]]
                    details = method.aggregate(model,updates,sizes,rnd)
                    if args.attack == 'neurotoxin':
                        mask = neurotoxin_mask((core.model_vector(model)-baseline)/args.server_lr,model,args.dense_ratio)
                        logging.info('Global %d | Neurotoxin next-round active coordinates=%d',rnd,sum(int(m.sum()) for m in mask.values()))
                    scores = core.evaluate(model,loaders,args)
                    if not all(math.isfinite(v) for v in scores.values()):
                        raise ValueError('Nonfinite evaluation metric')
                    row = dict(round=rnd,**scores,elapsed_seconds=time.perf_counter()-started,
                               round_seconds=time.perf_counter()-round_start,
                               selected_coordinates=details['selected_coordinates'],aggregate_delta_l2=details['aggregate_delta_l2'])
                    rows.append(row);writer.writerow(row);metrics.flush()
                    metadata['observed_rounds'] = rnd
                    core.save_json(output/'config.json',metadata)
                    logging.info('Global %d | Clean ACC=%.6f | ASR=%.6f | seconds=%.2f',rnd,scores['clean_acc'],scores['asr'],row['round_seconds'])
                    if rnd in [100,150,200]:
                        update_outputs(output,rows,args,'running')
                    del updates,details,baseline
            metadata['status'] = 'completed'
        except BaseException as error:
            failure = error
            metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
            traceback.print_exc()
        finally:
            metadata.update(elapsed_seconds=time.perf_counter()-started,finished_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
            core.save_json(output/'config.json',metadata)
            if rows:
                try:
                    update_outputs(output,rows,args,metadata['status'])
                except Exception:
                    traceback.print_exc()
                    if failure is None:
                        failure = RuntimeError('Report generation failed')
                        metadata.update(status='failed',error=str(failure))
                        core.save_json(output/'config.json',metadata)
            root.removeHandler(handler)
            sys.stdout,sys.stderr = original_out,original_err
    if failure:
        raise failure
    print(f'Completed: {output}',flush=True)


if __name__ == '__main__':
    train(parse_args())
