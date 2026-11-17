"""Legacy AlphaGuard VGG9: three beta settings, BadNet and PGD, three concurrent slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'research_results/AlphaGuard_original/cifar100_vgg9_legacy_beta0.5_0.7_0.9_badnet_pgd_20261117'
LOCK=threading.Lock()


def event(message):
    with LOCK:print(time.strftime('%Y-%m-%d %H:%M:%S %z'),message,flush=True)


def jobs():
    return [dict(attack=attack,beta=beta,bs=64,client_lr=.1,gpu=gpu,slot=slot,
                 path=f'beta{str(beta).replace(".","p")}/{attack}')
            for slot,beta,gpu in [(0,.5,0),(1,.7,1),(2,.9,0)] for attack in ['badnet','pgd']]


def prepare(batch):
    batch.mkdir(parents=True,exist_ok=True)
    f=batch/'batch.json'
    if f.exists():return json.loads(f.read_text())
    sources=['aggregation.py','agent.py','models.py','resnet.py','utils.py','ramp_ag_alpha_zero.py','ramp_attack_eval.py','alphaguard_vgg_eval.py','alphaguard_legacy_vgg.py','vgg.py']
    manifest=dict(jobs=jobs(),created_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  settings=dict(data='cifar100',model='vgg9',rounds=200,num_agents=20,num_corrupt=4,poison_frac=.3,non_iid=True,local_ep=2,lr_decay=.99,server_lr=1.,seed=0),
                  defense='Archived AlphaGuard methods from 2025-11-08-12-12: remove 6, median reference, top0.1%, penalty2. VGG9 state dimension 9282796; native PGD radius preserved.',
                  scheduling='Three slots: GPU0 beta0.5 and beta0.9; GPU1 beta0.7. Each slot runs BadNet then PGD. Shares GPUs with existing jobs as requested; existing processes and their locks are left untouched. This batch has independent slot locks.',
                  source_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in sources})
    f.write_text(json.dumps(manifest,indent=2));return manifest


def inspect(batch,job):
    d=batch/job['path'];f=d/'config.json';meta=json.loads(f.read_text()) if f.exists() else {}
    f=d/'metrics.csv';rows=list(csv.DictReader(f.open())) if f.exists() else []
    return d,meta,rows


def report(batch):
    manifest=prepare(batch)
    with (batch/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        summary,stages=[],[]
        for job in manifest['jobs']:
            d,meta,rows=inspect(batch,job);window=rows[-10:]
            summary.append(dict(attack=job['attack'],beta=job['beta'],bs=job['bs'],client_lr=job['client_lr'],status=meta.get('status','pending'),rounds=len(rows),
                                acc_recent10_pct=sum(100*float(r['clean_acc']) for r in window)/len(window) if window else '',
                                asr_recent10_pct=sum(100*float(r['asr']) for r in window)/len(window) if window else '',path=str(d)))
            for end in [100,150,200]:
                exact=[r for r in rows if int(r['round'])==end]
                window=[r for r in rows if end-9<=int(r['round'])<=end]
                if not exact or len(window)!=10:continue
                stages.append(dict(attack=job['attack'],beta=job['beta'],bs=job['bs'],client_lr=job['client_lr'],round=end,
                                   clean_acc_pct=100*float(exact[0]['clean_acc']),asr_pct=100*float(exact[0]['asr']),
                                   clean_acc_last10_pct=sum(100*float(r['clean_acc']) for r in window)/10,
                                   asr_last10_pct=sum(100*float(r['asr']) for r in window)/10))
        for name,records,fields in [('summary.csv',summary,list(summary[0])),('milestones.csv',stages,['attack','beta','bs','client_lr','round','clean_acc_pct','asr_pct','clean_acc_last10_pct','asr_last10_pct'])]:
            tmp=batch/(name+'.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
            os.replace(tmp,batch/name)
        lines=['# 旧版 AlphaGuard：CIFAR100 / VGG9','',f'更新：{time.strftime("%Y-%m-%d %H:%M:%S %z")}',
               '', 'beta=0.5、0.7、0.9，各跑 BadNet 和 PGD，共六组；初始学习率0.1，batch64。每组 200 轮；100/150/200 轮保存单轮指标和最近 10 轮均值。',
               '20 客户端，4 恶意，poison_frac=0.3，Non-IID，local_ep=2，SGD，momentum=0，weight_decay=1e-4，server_lr=1，seed=0。',
               'beta 是 Dirichlet 浓度参数；0.9 比 0.5 通常划分得更均匀。学习率按 client_lr * 0.99**round 衰减。',
               '旧实验备份恢复的 AlphaGuard：L2中间25%-75%取median，检测Top-K=0.001，排除6个，保留14个按penalty_strength=2的惩罚权重归一化。VGG9完整state_dict（含BN状态）参与更新与检测，无RAMP置零。',
               '三个并行槽位：GPU0跑beta0.5和0.9；GPU1跑beta0.7，各自先BadNet后PGD。本批次独立槽位锁；与旧任务共享GPU算力，不停止或修改旧任务。PGD沿用旧半径，逐轮记录投影检查和实际触发次数。',
               '每组仅保存 config.json、training.log、metrics.csv、milestones.csv、curves.png。没有代码副本、模型或坐标快照。',
               '', '| 攻击 | beta | batch | 初始 LR | 状态 | 轮数 | 最近 10 轮 ACC% | 最近 10 轮 ASR% |', '|---|---|---|---|---|---|---|---|']
        for r in summary:
            fmt=lambda v:f'{v:.3f}' if isinstance(v,float) else str(v)
            lines.append(f'| {r["attack"]} | {r["beta"]} | {r["bs"]} | {r["client_lr"]} | {r["status"]} | {r["rounds"]}/200 | {fmt(r["acc_recent10_pct"])} | {fmt(r["asr_recent10_pct"])} |')
        lines+=['','运行中表格是当前最近 10 轮，不能当作最终结果。阶段对比见 milestones.csv。',
                '', '```bash','python research/alphaguard_vgg_batch.py status','```']
        if stages:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            (batch/'figures').mkdir(exist_ok=True)
            for end in [100,150,200]:
                values={(r['attack'],r['beta'],r['bs'],r['client_lr']):r for r in stages if r['round']==end}
                if not values:continue
                fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
                labels=[f'{j["attack"]} beta={j["beta"]}' for j in manifest['jobs']]
                for ax,key,title in zip(axes,['clean_acc_last10_pct','asr_last10_pct'],['Clean ACC','ASR']):
                    ys=[values.get((j['attack'],j['beta'],j['bs'],j['client_lr']),{}).get(key,np.nan) for j in manifest['jobs']]
                    ax.bar(range(6),ys)
                    ax.set(xticks=range(6),xticklabels=labels,ylabel='%',ylim=(0,100),title=title)
                    ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
                fig.suptitle(f'Original AlphaGuard | CIFAR100 VGG9 BadNet / PGD | rounds {end-9}-{end}')
                name=f'milestone_{end}.png';fig.savefig(batch/'figures'/name,dpi=160);plt.close(fig)
                lines+=['',f'![Round {end}](figures/{name})']
        (batch/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def queue(batch,slot,manifest):
    lockdir=ROOT/'research_results/.gpu_locks';lockdir.mkdir(exist_ok=True,parents=True)
    for job in manifest['jobs']:
        if job['slot']!=slot:continue
        gpu=job['gpu']
        d,meta,rows=inspect(batch,job)
        if meta.get('status')=='completed' and len(rows)==200:continue
        if d.exists():raise RuntimeError(f'Incomplete output preserved: {d}')
        for name,digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Source changed: {name}')
        with (batch/f'.slot_{slot}.lock').open('a') as lock:
            event(f'WAIT cuda:{gpu} beta={job["beta"]} bs={job["bs"]} lr={job["client_lr"]}')
            fcntl.flock(lock,fcntl.LOCK_EX)
            d.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',str(ROOT/'src/alphaguard_vgg_eval.py'),'--attack',job['attack'],'--client_lr',str(job['client_lr']),
                 '--device',f'cuda:{gpu}','--output',str(d),'--beta',str(job['beta']),'--bs',str(job['bs'])]
            env=os.environ.copy();env.update(TZ='Asia/Shanghai',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            event(f'START cuda:{gpu} beta={job["beta"]} bs={job["bs"]} lr={job["client_lr"]}')
            code=subprocess.call(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            event(f'FINISH cuda:{gpu} beta={job["beta"]} bs={job["bs"]} lr={job["client_lr"]} exit={code}')
        report(batch)
        if code:raise RuntimeError(f'Failed run: {d}/training.log')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['launch','status','report','prepare']);p.add_argument('--batch',type=Path,default=DEFAULT)
    args=p.parse_args();batch=args.batch.resolve();manifest=prepare(batch)
    if args.action!='launch':
        for row in report(batch):print(row)
        return
    with (batch/'.launcher.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Baseline launcher already running')
        failures=[]
        def worker(slot):
            try:queue(batch,slot,manifest)
            except Exception as e:failures.append(str(e));event(str(e))
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending={pool.submit(worker,slot) for slot in [0,1,2]}
            while pending:
                done,pending=wait(pending,timeout=60,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                report(batch)
        event('All queues finished; failures='+str(failures))
        if failures:sys.exit(1)


if __name__=='__main__':main()
