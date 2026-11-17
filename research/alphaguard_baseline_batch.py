"""Original AlphaGuard: two client LRs x BadNet/PGD, one extra job per GPU."""
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
DEFAULT=ROOT/'research_results/AlphaGuard_original/cifar100_resnet9_badnet_pgd_lr0.1_0.05_20261116'
LOCK=threading.Lock()


def event(message):
    with LOCK:print(time.strftime('%Y-%m-%d %H:%M:%S %z'),message,flush=True)


def jobs():
    return [dict(attack=attack,client_lr=lr,gpu=gpu,path=f'{attack}/lr{str(lr).replace(".","p")}')
            for gpu,lr in [(0,.1),(1,.05)] for attack in ['badnet','pgd']]


def prepare(batch):
    batch.mkdir(parents=True,exist_ok=True)
    f=batch/'batch.json'
    if f.exists():return json.loads(f.read_text())
    sources=['aggregation.py','agent.py','models.py','resnet.py','utils.py','ramp_ag_alpha_zero.py','ramp_attack_eval.py','alphaguard_baseline_eval.py']
    manifest=dict(jobs=jobs(),created_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  settings=dict(data='cifar100',model='resnet9',rounds=200,num_agents=20,num_corrupt=4,poison_frac=.3,non_iid=True,beta=.5,local_ep=2,bs=64,lr_decay=.99,server_lr=1.,seed=0),
                  defense='Unchanged native Aggregation with aggr=alphaguard; native penalty_strength=2; no RAMP masking.',
                  scheduling='Two new jobs maximum, one extra per GPU. Shared GPU locks with the ongoing RAMP batch; may slow concurrent training. Respects exclusive Dynamic Trigger locks.',
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
            summary.append(dict(attack=job['attack'],client_lr=job['client_lr'],status=meta.get('status','pending'),rounds=len(rows),
                                acc_recent10_pct=sum(100*float(r['clean_acc']) for r in window)/len(window) if window else '',
                                asr_recent10_pct=sum(100*float(r['asr']) for r in window)/len(window) if window else '',path=str(d)))
            for end in [100,150,200]:
                exact=[r for r in rows if int(r['round'])==end]
                window=[r for r in rows if end-9<=int(r['round'])<=end]
                if not exact or len(window)!=10:continue
                stages.append(dict(attack=job['attack'],client_lr=job['client_lr'],round=end,
                                   clean_acc_pct=100*float(exact[0]['clean_acc']),asr_pct=100*float(exact[0]['asr']),
                                   clean_acc_last10_pct=sum(100*float(r['clean_acc']) for r in window)/10,
                                   asr_last10_pct=sum(100*float(r['asr']) for r in window)/10))
        for name,records,fields in [('summary.csv',summary,list(summary[0])),('milestones.csv',stages,['attack','client_lr','round','clean_acc_pct','asr_pct','clean_acc_last10_pct','asr_last10_pct'])]:
            tmp=batch/(name+'.tmp')
            with tmp.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
            os.replace(tmp,batch/name)
        lines=['# 原始 AlphaGuard：CIFAR100 / ResNet9','',f'更新：{time.strftime("%Y-%m-%d %H:%M:%S %z")}',
               '', 'BadNet、PGD × 客户端初始学习率 0.1、0.05，共四组。每组 200 轮；100/150/200 轮保存单轮指标和最近 10 轮均值。',
               '20 客户端，4 恶意，poison_frac=0.3，Non-IID beta=0.5，local_ep=2，batch=64，SGD，momentum=0，weight_decay=1e-4，server_lr=1，seed=0。',
               '学习率按 client_lr * 0.99**round 衰减。PGD 保留原项目 CIFAR100 半径 ||initial_global_model||，每个本地 epoch 末投影。',
               '防御直接调用原 aggregation.py 的 Aggregation.aggregate_updates；先排除异常客户端，再按原生惩罚权重在保留集合内归一化；原生 penalty_strength=2、检测 Top-K=0.001、参考 median。无 RAMP 置零。',
               '每张卡最多增加一个新任务；原有任务继续运行，但共享算力可能降低速度。遵守原批次 GPU 独占锁。',
               '每组仅保存 config.json、training.log、metrics.csv、milestones.csv、curves.png。没有代码副本、模型或坐标快照。',
               '', '| 攻击 | 初始 LR | 状态 | 轮数 | 最近 10 轮 ACC% | 最近 10 轮 ASR% |', '|---|---|---|---|---|---|']
        for r in summary:
            fmt=lambda v:f'{v:.3f}' if isinstance(v,float) else str(v)
            lines.append(f'| {r["attack"]} | {r["client_lr"]} | {r["status"]} | {r["rounds"]}/200 | {fmt(r["acc_recent10_pct"])} | {fmt(r["asr_recent10_pct"])} |')
        lines+=['','运行中表格是当前最近 10 轮，不能当作最终结果。阶段对比见 milestones.csv。',
                '', '```bash','python research/alphaguard_baseline_batch.py status','```']
        (batch/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def queue(batch,gpu,manifest):
    lockdir=ROOT/'research_results/.gpu_locks';lockdir.mkdir(exist_ok=True,parents=True)
    for job in manifest['jobs']:
        if job['gpu']!=gpu:continue
        d,meta,rows=inspect(batch,job)
        if meta.get('status')=='completed' and len(rows)==200:continue
        if d.exists():raise RuntimeError(f'Incomplete output preserved: {d}')
        for name,digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Source changed: {name}')
        with (lockdir/f'cuda_{gpu}.lock').open('a') as lock:
            event(f'WAIT cuda:{gpu} {job["attack"]} lr={job["client_lr"]}')
            fcntl.flock(lock,fcntl.LOCK_SH)
            d.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',str(ROOT/'src/alphaguard_baseline_eval.py'),'--attack',job['attack'],'--client_lr',str(job['client_lr']),
                 '--device',f'cuda:{gpu}','--output',str(d)]
            env=os.environ.copy();env.update(TZ='Asia/Shanghai',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            event(f'START cuda:{gpu} {job["attack"]} lr={job["client_lr"]}')
            code=subprocess.call(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            event(f'FINISH cuda:{gpu} {job["attack"]} lr={job["client_lr"]} exit={code}')
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
        def worker(gpu):
            try:queue(batch,gpu,manifest)
            except Exception as e:failures.append(str(e));event(str(e))
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(worker,gpu) for gpu in [0,1]}
            while pending:
                done,pending=wait(pending,timeout=60,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                report(batch)
        event('All queues finished; failures='+str(failures))
        if failures:sys.exit(1)


if __name__=='__main__':main()
