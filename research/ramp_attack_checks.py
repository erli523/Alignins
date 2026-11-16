"""Server smoke validation and measured 2-vs-4 process throughput."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CHECK=Path('/data/risegpu_ramp_attack_eval/validation_20261116')
CHECK.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
summary={'status':'running','smoke':[],'benchmark':{}}


def save():
    f=CHECK/'checks.json';tmp=CHECK/'checks.json.tmp'
    tmp.write_text(json.dumps(summary,indent=2));tmp.replace(f)


def run_one(data,attack,config,gpu,tag,batches=1,dynamic_fast=True):
    path=CHECK/tag
    cmd=[sys.executable,'-u',str(ROOT/'src/ramp_attack_eval.py'),'--data',data,'--attack',attack,
         '--config',config,'--device',f'cuda:{gpu}','--output',str(path),
         '--rounds','2','--max_batches',str(batches),'--test_limit','512']
    if attack=='dynamic_trigger' and dynamic_fast:
        cmd+=['--trigger_outter_epochs','2','--dm_adv_model_count','1','--dm_adv_epochs','1','--trigger_optim_interval','1']
    start=time.perf_counter()
    code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    meta=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
    result=dict(data=data,attack=attack,config=config,exit=code,status=meta.get('status'),rounds=meta.get('observed_rounds'),
                seconds=time.perf_counter()-start,path=str(path),error=meta.get('error'),trigger_changed=meta.get('trigger_changed'))
    print(json.dumps(result),flush=True)
    return result


try:
    save()
    attacks=['badnet','DBA','neurotoxin','pgd','model_replacement','altermin','dynamic_trigger']
    tasks=[(data,a,'A',gpu,f'smoke_{data}_{a}_A') for a in attacks for gpu,data in enumerate(['cifar10','cifar100'])]
    tasks += [(data,'neurotoxin','B',gpu,f'smoke_{data}_neurotoxin_B') for gpu,data in enumerate(['cifar10','cifar100'])]
    for start in range(0,len(tasks),2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda x:run_one(*x),tasks[start:start+2]))
        summary['smoke'].extend(results);save()
        if any(r['exit'] or r['status']!='completed' or r['rounds']!=2 for r in results):
            raise RuntimeError('Smoke failed; inspect training.log')
    for mode in [2,4]:
        tasks=[('cifar10','DBA','A',j%2,f'bench_{mode}_{j}',4) for j in range(4)]
        start=time.perf_counter()
        if mode==2:
            results=[]
            for i in [0,2]:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results.extend(pool.map(lambda x:run_one(*x),tasks[i:i+2]))
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda x:run_one(*x),tasks))
        summary['benchmark'][str(mode)]=dict(wall_seconds=time.perf_counter()-start,results=results)
        save()
        if any(r['exit'] for r in results):raise RuntimeError('Benchmark failed')
    summary['recommended_queues']=4 if summary['benchmark']['4']['wall_seconds']<summary['benchmark']['2']['wall_seconds'] else 2
    summary['status']='completed'
except BaseException as e:
    summary.update(status='failed',error=str(e))
finally:
    save()
print(json.dumps(summary,indent=2),flush=True)
