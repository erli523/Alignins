"""Four queues, two GPUs, lean attack-evaluation reports. Run launch/report/status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'research_results/RAMP_attack_eval/cifar10_cifar100_AB_20261116'
ATTACKS = ['DBA','neurotoxin','pgd','model_replacement','altermin','dynamic_trigger']
CONFIGS = {'A':dict(topk_ratio=.05,penalty_strength=5), 'B':dict(topk_ratio=.04,penalty_strength=7)}
QUEUES = [('cifar10','A',0,0),('cifar10','B',1,0),('cifar100','A',0,1),('cifar100','B',1,1)]


def atomic_json(path,value):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)


def jobs():
    result=[]
    for data,config,gpu,slot in QUEUES:
        for attack in (['badnet']+ATTACKS if data=='cifar100' else ATTACKS):
            result.append(dict(data=data,config=config,attack=attack,gpu=gpu,slot=slot,
                               rounds=100 if data=='cifar10' else 200,client_lr=.1 if data=='cifar10' else .05,
                               relative_path=f'{data}/{attack}/{config}_zero{5 if config=="A" else 4}pct_penalty{5 if config=="A" else 7}'))
    old=ROOT/'research_results/RAMP_grid/zero1to5_penalty2to8_20261112/runs'
    for c,z,p in [('A',5,5),('B',4,7)]:
        result.append(dict(data='cifar10',config=c,attack='badnet',rounds=100,client_lr=.1,
                           reused=True,source=str(old/f'zero{z}pct_penalty{p}')))
    return result


def prepare(batch):
    batch.mkdir(parents=True,exist_ok=True)
    manifest=batch/'batch.json'
    if manifest.exists():return json.loads(manifest.read_text())
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}
    value=dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=jobs(),source_sha256=hashes,
               configs=CONFIGS,notes='26 new runs plus 2 reused CIFAR10 BadNet results. No model/source snapshots. Initial LR decays by .99**round, following native Agent.',
               queues=QUEUES)
    atomic_json(manifest,value)
    return value


def inspect_job(batch,job):
    path=Path(job['source']) if job.get('reused') else batch/job['relative_path']
    meta_path=path/('run.json' if job.get('reused') else 'config.json')
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    metrics_path=path/('tables/metrics.csv' if job.get('reused') else 'metrics.csv')
    if metrics_path.exists():
        with metrics_path.open() as f:rows=list(csv.DictReader(f))
    else:rows=[]
    rounds=max((int(r['round']) for r in rows),default=0)
    complete=meta.get('status')=='completed' and rounds==job['rounds'] and len(rows)==job['rounds']
    status=('reused_completed' if job.get('reused') else 'completed') if complete else meta.get('status','pending')
    return path,meta,rows,rounds,status


def report(batch):
    manifest=prepare(batch)
    with (batch/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        table,stages=[],[]
        for job in manifest['jobs']:
            path,meta,rows,rnd,status=inspect_job(batch,job)
            recent=rows[-10:]
            record=dict(data=job['data'],attack=job['attack'],config=job['config'],
                        zero_pct=100*CONFIGS[job['config']]['topk_ratio'],penalty_strength=CONFIGS[job['config']]['penalty_strength'],
                        status=status,rounds=rnd,target_rounds=job['rounds'],
                        acc_recent10_pct=sum(100*float(r['clean_acc']) for r in recent)/len(recent) if recent else '',
                        asr_recent10_pct=sum(100*float(r['asr']) for r in recent)/len(recent) if recent else '',path=str(path))
            table.append(record)
            for end in [100,150,200]:
                if end>job['rounds']:continue
                window=[r for r in rows if end-9<=int(r['round'])<=end]
                exact=[r for r in rows if int(r['round'])==end]
                if not exact or len(window)!=10:continue
                stages.append(dict(data=job['data'],attack=job['attack'],config=job['config'],round=end,
                                   clean_acc_pct=100*float(exact[0]['clean_acc']),asr_pct=100*float(exact[0]['asr']),
                                   clean_acc_last10_pct=sum(100*float(r['clean_acc']) for r in window)/10,
                                   asr_last10_pct=sum(100*float(r['asr']) for r in window)/10))
        for name,records,fields in [('summary.csv',table,list(table[0])),('milestones.csv',stages,['data','attack','config','round','clean_acc_pct','asr_pct','clean_acc_last10_pct','asr_last10_pct'])]:
            tmp=batch/(name+'.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
            os.replace(tmp,batch/name)
        counts={s:sum(r['status']==s for r in table) for s in sorted(set(r['status'] for r in table))}
        lines=['# RAMP attack evaluation','',f'Updated: {time.strftime("%Y-%m-%d %H:%M:%S %z")}',f'Status: {counts}',
               '', 'A: zero Top-K 5%, penalty 5. B: zero Top-K 4%, penalty 7. Detection Top-K 0.1%, coordinate threshold 1.',
               'CIFAR10: LR 0.1, 100 rounds. CIFAR100: LR 0.05, 200 rounds; milestones 100/150/200. LR decay .99 per round.',
               '20 clients, 4 corrupt, local_ep=2, batch=64, Dirichlet beta=.5, poison_frac=.3 where applicable, seed=0.',
               'AlterMin: LOCAL TRAIN source class3 -> target7, max100 samples, benign2 + malicious1 phases; no held-out training.',
               'Model replacement: native CLI gamma50 * constrain_weight.02 = effective scaling1, alpha.5; this is NOT a strong amplification setting.',
               'PGD: preserve native radius .1*||w|| on CIFAR10, 1*||w|| on CIFAR100, end-of-epoch projection.',
               'Neurotoxin: layerwise bottom25% of previous defended aggregate; no mask in first round.',
               'Dynamic trigger: native defaults (size5, lr.1, outer50, interval5, adv_models3, adv_epochs2); native normalized-space patch; evaluate client0 current trigger.',
               'ASR: non-target triggered test inputs except AlterMin, which evaluates clean source3 -> target7. ASR values across these definitions should not be treated as identical attack tasks.',
               'CIFAR10 BadNet A/B reuse the completed grid; original source paths are linked below.',
               'No model, source snapshot, or coordinate dumps. Each run has config.json, training.log, metrics.csv, milestones.csv, curves.png.',
               '', '| Dataset | Attack | Config | Status | Rounds | ACC recent10 % | ASR recent10 % | Results |',
               '|---|---|---|---|---|---|---|---|']
        for r in table:
            fmt=lambda v:f'{v:.3f}' if isinstance(v,float) else str(v)
            target=Path(r['path'])
            try:link=str(target.relative_to(batch))
            except ValueError:link=str(target)
            lines.append(f'| {r["data"]} | {r["attack"]} | {r["config"]} | {r["status"]} | {r["rounds"]}/{r["target_rounds"]} | {fmt(r["acc_recent10_pct"])} | {fmt(r["asr_recent10_pct"])} | [open]({link}) |')
        lines+=['','Recent10 refers to the latest available rounds, not necessarily the final 10; check status and rounds.',
                'milestones.csv stores exact-round metrics AND preceding 10-round means at 100/150/200.',
                '', '## Run / inspect', '```bash',f'python research/ramp_attack_batch.py status --batch {batch}',
                f'python research/ramp_attack_batch.py report --batch {batch}','```']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        (batch/'figures').mkdir(exist_ok=True)
        order=['badnet']+ATTACKS
        for data in ['cifar10','cifar100']:
            ends=[100] if data=='cifar10' else [100,150,200]
            for end in ends:
                values=[r for r in stages if r['data']==data and r['round']==end]
                if not values:continue
                fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
                for ax,key,title in zip(axes,['clean_acc_last10_pct','asr_last10_pct'],['Clean ACC','ASR']):
                    x=np.arange(len(order))
                    for j,c in enumerate(['A','B']):
                        lookup={r['attack']:r[key] for r in values if r['config']==c}
                        ax.bar(x+(j-.5)*.36,[lookup.get(a,np.nan) for a in order],width=.36,label=c)
                    ax.set(xticks=x,xticklabels=order,ylabel='%',ylim=(0,100),title=f'{title}: rounds {end-9}-{end}')
                    ax.tick_params(axis='x',labelrotation=30);ax.legend();ax.grid(axis='y',alpha=.2)
                fig.suptitle(f'{data} | A zero5% penalty5 vs B zero4% penalty7')
                name=f'{data}_round{end}.png';fig.savefig(batch/'figures'/name,dpi=160);plt.close(fig)
                lines+=['',f'![{data} round{end}](figures/{name})']
        (batch/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return counts


def verify_sources(manifest):
    for relative,digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Source changed during batch: {relative}; investigate before restarting')


def run_queue(batch,queue,manifest):
    data,config,gpu,slot=QUEUES[queue-1]
    lock_dir=ROOT/'research_results/.gpu_locks';lock_dir.mkdir(parents=True,exist_ok=True)
    with (batch/f'.queue{queue}.lock').open('a') as qlock:
        try:fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError(f'Queue {queue} already running')
        for job in manifest['jobs']:
            if job.get('reused') or (job['data'],job['config'])!=(data,config):continue
            path,meta,rows,rnd,status=inspect_job(batch,job)
            if status=='completed':continue
            if path.exists():
                raise RuntimeError(f'Existing incomplete run preserved: {path}. Do not overwrite it.')
            verify_sources(manifest)
            with (lock_dir/f'cuda_{gpu}.lock').open('a') as glock:
                # Shared allows two light jobs; exclusive serializes expensive dynamic-trigger jobs.
                fcntl.flock(glock,fcntl.LOCK_EX if job['attack']=='dynamic_trigger' else fcntl.LOCK_SH)
                path.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-u',str(ROOT/'src/ramp_attack_eval.py'),'--data',data,'--attack',job['attack'],
                     '--config',config,'--device',f'cuda:{gpu}','--output',str(path)]
                print(f'START queue{queue} cuda:{gpu} {data} {job["attack"]} {config}',flush=True)
                env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                # The child owns the sole training.log, so no duplicate console log is retained.
                code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                print(f'FINISH queue{queue} {data} {job["attack"]} {config} exit={code}',flush=True)
            report(batch)
            if code:raise RuntimeError(f'Run failed; inspect {path}/training.log')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','launch','queue','report','status'])
    p.add_argument('--batch',type=Path,default=DEFAULT)
    p.add_argument('--queue',type=int,choices=[1,2,3,4])
    p.add_argument('--queues',type=int,choices=[2,4],default=4,help='4=two light runs/GPU; 2=serialize queues per GPU')
    args=p.parse_args();batch=args.batch.resolve();manifest=prepare(batch)
    if args.action in ['prepare','report','status']:
        print(report(batch),flush=True)
        if args.action=='status':
            for job in manifest['jobs']:
                path,meta,rows,rnd,status=inspect_job(batch,job)
                print(job['data'],job['attack'],job['config'],status,f'{rnd}/{job["rounds"]}')
        return
    if args.action=='queue':
        if args.queue is None:p.error('--queue is required')
        run_queue(batch,args.queue,manifest);return
    with (batch/'.launcher.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A batch launcher is already running')
        failures=[]
        def group(qs):
            for q in qs:
                try:run_queue(batch,q,manifest)
                except Exception as e:
                    failures.append(str(e));print('QUEUE FAILED:',str(e),flush=True)
        groups=[[1],[2],[3],[4]] if args.queues==4 else [[1,3],[2,4]]
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            list(pool.map(group,groups))
        print('Batch finished. Failures:',failures,flush=True);report(batch)
        if failures:sys.exit(1)


if __name__=='__main__':main()
