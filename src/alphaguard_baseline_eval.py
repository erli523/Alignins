"""Original project AlphaGuard, CIFAR100/ResNet9, BadNet or PGD.

Uses the unchanged Aggregation.aggregate_updates and Agent.local_train.
Only reporting is added; no RAMP masking or source/model snapshots.
"""
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import ramp_attack_eval as helpers
from aggregation import Aggregation

core = helpers.core


class ReportedAlphaGuard(Aggregation):
    """Observe native selection and penalty weights, without changing returned values."""
    def _alphaguard_stage3(self, updates, alpha):
        keep, penalties = super()._alphaguard_stage3(updates, alpha)
        self.last_keep = keep.detach().clone()
        self.last_penalties = penalties.detach().clone()
        kept = set(keep.cpu().tolist())
        denominator = penalties[keep].sum()
        for index, cid in enumerate(self._current_client_ids):
            weight = float(penalties[index] / denominator) if index in kept else 0.
            logging.info('Native final weight | Client %d | role=%s | kept=%s | penalty=%.8g | normalized_weight=%.8g',
                         cid,'malicious' if cid < self.args.num_corrupt else 'benign',index in kept,float(penalties[index]),weight)
        return keep, penalties


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attack',choices=['badnet','pgd'],required=True)
    p.add_argument('--client_lr',type=float,choices=[0.1,0.05],required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--bs',type=int,default=64)
    p.add_argument('--beta',type=float,default=0.5)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max_batches',type=int,default=0,help='Smoke validation only; 0 = full local data')
    p.add_argument('--test_limit',type=int,default=0,help='Smoke validation only; 0 = all test data')
    a = p.parse_args(argv)
    if a.bs<1:p.error('--bs must be positive')
    if not np.isfinite(a.beta) or a.beta<=0:p.error('--beta must be finite and positive')
    batch_size,beta=a.bs,a.beta
    a = helpers.parse_args(['--data','cifar100','--attack',a.attack,'--client_lr',str(a.client_lr),
                           '--device',a.device,'--rounds',str(a.rounds),'--output',str(a.output),
                           '--max_batches',str(a.max_batches),'--test_limit',str(a.test_limit)])
    for key in ['config','ag_topk_ratio','topk_ratio','penalty_strength','alpha_threshold','epsilon',
                'reference','l2_lower','l2_upper','min_reference','zero_scope']:
        delattr(a,key)
    a.aggr, a.ag_topk, a.ag_gref_strategy = 'alphaguard',0.001,'median'
    a.bs=batch_size
    a.beta=beta
    return a


def reports(output, rows, args, status):
    stages=[]
    for end in sorted({100,150,200,args.rounds}):
        exact=[r for r in rows if r['round']==end]
        window=[r for r in rows if max(1,end-9)<=r['round']<=end]
        if not exact:continue
        stages.append(dict(round=end,clean_acc_pct=100*exact[0]['clean_acc'],asr_pct=100*exact[0]['asr'],
                           window_start=window[0]['round'],window_count=len(window),
                           clean_acc_last10_pct=100*np.mean([r['clean_acc'] for r in window]),
                           asr_last10_pct=100*np.mean([r['asr'] for r in window])))
    helpers.csv_write(output/'milestones.csv',stages,['round','clean_acc_pct','asr_pct','window_start','window_count','clean_acc_last10_pct','asr_last10_pct'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key,title in zip(axes,['clean_acc','asr'],['Clean ACC','ASR']):
        ax.plot([r['round'] for r in rows],[100*r[key] for r in rows])
        for end in [100,150,200]:
            if end<=args.rounds:ax.axvline(end,color='gray',linestyle='--',alpha=.3)
        ax.set(title=title,xlabel='Global round',ylabel='%',ylim=(0,100));ax.grid(alpha=.2)
    fig.suptitle(f'Original AlphaGuard | CIFAR100 ResNet9 | {args.attack} | LR={args.client_lr}, beta={args.beta}, batch={args.bs} | {status}')
    fig.savefig(output/'curves.png',dpi=160);plt.close(fig)


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    meta=dict(status='initializing',observed_rounds=0,settings=vars(args).copy(),
              started_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),defense='Original Aggregation.aggregate_updates, aggr=alphaguard',
              aggregation='Native exclusion, then native penalty-weight normalization over retained clients; no coordinate zeroing; native penalty_strength=2.',
              pgd='Native CIFAR100 radius=||initial_global_model||; end-of-local-epoch projection.',
              source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                             for name in ['aggregation.py','agent.py','models.py','resnet.py','utils.py','alphaguard_baseline_eval.py','ramp_attack_eval.py','ramp_ag_alpha_zero.py']})
    core.save_json(output/'config.json',meta)
    started=time.perf_counter();rows=[];failure=None
    stdout,stderr=sys.stdout,sys.stderr
    with (output/'training.log').open('w',encoding='utf-8',buffering=1) as log:
        sys.stdout,sys.stderr=core.Tee(stdout,log),core.Tee(stderr,log)
        logger=logging.getLogger();logger.handlers.clear();logger.setLevel(logging.INFO)
        handler=logging.StreamHandler(sys.stderr);handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'));logger.addHandler(handler)
        try:
            os.chdir(Path(__file__).parent)
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            logging.info('Original AlphaGuard baseline | settings=%s',args)
            train,test=core.utils.get_datasets(args.data)
            if args.test_limit:
                test.data=test.data[:args.test_limit].copy();test.targets=test.targets[:args.test_limit].clone()
            groups=core.utils.distribute_data_dirichlet(train,args)
            if min(map(len,groups.values()))<args.bs:raise ValueError('Client dataset too small')
            meta['partition_sha256']=hashlib.sha256(json.dumps({int(k):list(map(int,v)) for k,v in groups.items()},sort_keys=True).encode()).hexdigest()
            loaders=helpers.evaluation_loaders(test,args,[])
            model=core.models.get_model(args.data,args).to(args.device)
            meta['initial_model_sha256']=core.tensor_hash(core.model_vector(model))
            agents,sizes=[],{}
            for cid in range(args.num_agents):
                agent=helpers.Agent(cid,args,train,groups[cid],backdoor_train_dataset=None)
                agent.is_malicious=int(cid<args.num_corrupt)
                if args.max_batches:
                    agent.train_loader=DataLoader(Subset(agent.train_dataset,range(min(len(agent.train_dataset),args.max_batches*args.bs))),batch_size=args.bs,shuffle=True,drop_last=True,num_workers=0)
                agents.append(agent);sizes[cid]=agent.n_data
            method=ReportedAlphaGuard(sizes,core.model_vector(model).numel(),args)
            meta.update(status='running',local_data_sizes=sizes,alphaguard_parameters=method.alphaguard_params)
            core.save_json(output/'config.json',meta)
            criterion=torch.nn.CrossEntropyLoss().to(args.device)
            fields=['round','clean_acc','asr','clean_target_rate','triggered_original_acc','elapsed_seconds','round_seconds','retained_clients','aggregate_delta_l2']
            with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();file.flush()
                for rnd in range(1,args.rounds+1):
                    tick=time.perf_counter();baseline=core.model_vector(model).clone()
                    ids=sorted(np.random.choice(args.num_agents,args.num_agents,replace=False).tolist())
                    logging.info('Global %d | participants=%s | malicious=%s | benign=%s | local_lr=%.8g',rnd,ids,ids[:args.num_corrupt],ids[args.num_corrupt:],args.client_lr*args.lr_decay**rnd)
                    updates={}
                    for cid in ids:
                        updates[cid]=agents[cid].local_train(model,criterion,rnd,neurotoxin_mask={})
                        if not torch.isfinite(updates[cid]).all():raise ValueError(f'Nonfinite update client {cid}')
                        core.utils.vector_to_model(baseline.clone(),model)
                    # The original server aggregation, including native server_lr and selection.
                    _,unused_mask=method.aggregate_updates(model,updates)
                    del unused_mask
                    if not torch.isfinite(core.model_vector(model)).all():raise ValueError('Nonfinite global model')
                    scores=core.evaluate(model,loaders,args)
                    row=dict(round=rnd,**scores,elapsed_seconds=time.perf_counter()-started,round_seconds=time.perf_counter()-tick,
                             retained_clients=len(method.last_keep),aggregate_delta_l2=float((core.model_vector(model)-baseline).norm()))
                    rows.append(row);writer.writerow(row);file.flush()
                    logging.info('Global %d | Clean ACC=%.6f | ASR=%.6f | seconds=%.2f',rnd,scores['clean_acc'],scores['asr'],row['round_seconds'])
                    meta['observed_rounds']=rnd;core.save_json(output/'config.json',meta)
                    if rnd in [100,150,200]:reports(output,rows,args,'running')
                    del updates,baseline
            meta['status']='completed'
        except BaseException as e:
            failure=e;meta.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=f'{type(e).__name__}: {e}');traceback.print_exc()
        finally:
            meta.update(elapsed_seconds=time.perf_counter()-started,finished_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
            core.save_json(output/'config.json',meta)
            if rows:
                try:reports(output,rows,args,meta['status'])
                except Exception:
                    traceback.print_exc()
                    if failure is None:
                        failure=RuntimeError('Report generation failed')
                        meta.update(status='failed',error=str(failure));core.save_json(output/'config.json',meta)
            logger.removeHandler(handler);sys.stdout,sys.stderr=stdout,stderr
    if failure:raise failure


if __name__=='__main__':train(parse_args())
