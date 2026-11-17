"""Four resource-aware queues: CIFAR100 beta0.7, B, two models, seven attacks."""
import argparse,csv,fcntl,hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'research_results/RAMP_B/cifar100_beta0.7_resnet9_vgg9_150rounds_20261117'
LOCK=threading.RLock()
ACTIVE={}
ATTACKS=['badnet','DBA','neurotoxin','pgd','altermin','model_replacement','dynamic_trigger']
def event(s):
 with LOCK:print(time.strftime('%Y-%m-%d %H:%M:%S %z'),s,flush=True)
def prepare(base):
 base.mkdir(parents=True,exist_ok=True);f=base/'batch.json'
 if f.exists():return json.loads(f.read_text())
 jobs=[]
 for attack in ATTACKS:
  for model in ['resnet9','vgg9']:
   i=len(jobs);slot=i%4
   jobs.append(dict(model=model,attack=attack,slot=slot,gpu=slot//2,path=f'{model}/{attack}'))
 files=['ramp_beta07_eval.py','ramp_attack_eval.py','ramp_ag_alpha_zero.py','agent.py','aggregation.py','models.py','resnet.py','vgg.py','utils.py','dynamic_trigger_attack.py','altermin_attack.py','model_replacement_attack.py']
 m=dict(jobs=jobs,settings=dict(data='cifar100',beta=.7,rounds=150,client_lr=.05,bs=64,local_ep=2,num_agents=20,num_corrupt=4,seed=0,config='B',zero_topk=.04,penalty_strength=7,detection_topk=.001,alpha_threshold=1),batchnorm_policy='Both models track_running_stats=False, trainable update coordinates only',source_sha256={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in files})
 f.write_text(json.dumps(m,indent=2));return m
def inspect(base,j):
 d=base/j['path'];f=d/'config.json';m=json.loads(f.read_text()) if f.exists() else {}
 f=d/'metrics.csv';rows=list(csv.DictReader(f.open())) if f.exists() else []
 return d,m,rows
def report(base):
 with LOCK:
  manifest=prepare(base);summary=[];stages=[]
  for j in manifest['jobs']:
   d,m,rows=inspect(base,j);window=rows[-10:]
   r=dict(model=j['model'],attack=j['attack'],gpu=j['gpu'],status=m.get('status','waiting'),rounds=len(rows),acc_recent10_pct=sum(float(x['clean_acc'])*100 for x in window)/len(window) if window else '',asr_recent10_pct=sum(float(x['asr'])*100 for x in window)/len(window) if window else '')
   summary.append(r)
   for end in [100,150]:
    window=[x for x in rows if end-9<=int(x['round'])<=end];exact=[x for x in rows if int(x['round'])==end]
    if len(window)==10 and exact:stages.append(dict(model=j['model'],attack=j['attack'],round=end,acc_pct=float(exact[0]['clean_acc'])*100,asr_pct=float(exact[0]['asr'])*100,acc_last10_pct=sum(float(x['clean_acc'])*100 for x in window)/10,asr_last10_pct=sum(float(x['asr'])*100 for x in window)/10))
  for name,records,fields in [('summary.csv',summary,list(summary[0])),('milestones.csv',stages,['model','attack','round','acc_pct','asr_pct','acc_last10_pct','asr_last10_pct'])]:
   with (base/(name+'.tmp')).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
   os.replace(base/(name+'.tmp'),base/name)
  lines=['# RAMP B / CIFAR100 / beta0.7','', 'ResNet9 and VGG9; 7 attacks each; 150 rounds; LR0.05; batch64; local_ep2; clients20/malicious4; seed0. Zero4%, penalty7, detection0.1%, threshold1.', '', 'Both models use BN track_running_stats=False. VGG9 running buffers are disabled to match the parameter-only B implementation. Attack-specific defaults are preserved; PGD uses the native radius; model replacement effective scaling is 50*0.02=1.', '', 'Four queues, two per GPU. Memory-aware admission reserves headroom and may wait for older tasks. No existing tasks are terminated. No source/model snapshots. Metrics are fractions, report tables are percentages.', '', '| Model | Attack | GPU | Status | Rounds | Recent10 ACC% | Recent10 ASR% |','|---|---|---|---|---|---|---|']
  for r in summary:
   fmt=lambda x:f'{x:.3f}' if isinstance(x,float) else str(x)
   lines.append('| '+' | '.join(fmt(r[k]) for k in ['model','attack','gpu','status','rounds','acc_recent10_pct','asr_recent10_pct'])+' |')
  if stages:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   import numpy as np
   (base/'figures').mkdir(exist_ok=True)
   for end in [100,150]:
    vals={(r['model'],r['attack']):r for r in stages if r['round']==end}
    if not vals:continue
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,key in zip(axes,['acc_last10_pct','asr_last10_pct']):
     for offset,model in [(-.2,'resnet9'),(.2,'vgg9')]:
      ax.bar(np.arange(7)+offset,[vals.get((model,a),{}).get(key,np.nan) for a in ATTACKS],width=.4,label=model)
     ax.set(xticks=range(7),xticklabels=ATTACKS,ylim=(0,100),title=key,ylabel='%');ax.tick_params(axis='x',rotation=45);ax.legend()
    fig.suptitle(f'B CIFAR100 beta0.7 rounds {end-9}-{end}');fig.savefig(base/'figures'/f'comparison_{end}.png',dpi=160);plt.close(fig)
    lines+=['',f'![Round {end}](figures/comparison_{end}.png)']
  (base/'README.md').write_text('\n'.join(lines)+'\n');return summary

def admit(gpu,reserve):
 info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(l.split(',')[0]):int(l.split(',')[1]) for l in info.splitlines()}
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
 used={int(l.split(',')[0]):int(l.split(',')[1]) for l in procs.splitlines() if l.strip() and l.split(',')[1].strip().isdigit()}
 reserved=sum(max(0,r-used.get(pid,0)) for pid,(g,r) in ACTIVE.items() if g==gpu)
 return free[gpu]-reserved>=reserve+1800

def worker(base,slot,manifest):
 failures=[]
 for j in manifest['jobs']:
  if j['slot']!=slot:continue
  d,m,rows=inspect(base,j)
  if m.get('status')=='completed' and len(rows)==150:continue
  if d.exists():event(f'PRESERVED incomplete {d}');failures.append(str(d));continue
  for n,h in manifest['source_sha256'].items():
   if hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed: '+n)
  reserve=(11000 if j['model']=='vgg9' else 9000) if j['attack']=='dynamic_trigger' else (6000 if j['model']=='vgg9' else 4500)
  event(f'WAIT GPU{j["gpu"]} {j["path"]} memory_reserve={reserve}MiB')
  while True:
   with LOCK:
    if admit(j['gpu'],reserve):
     d.parent.mkdir(parents=True,exist_ok=True)
     cmd=[sys.executable,'-u',str(ROOT/'src/ramp_beta07_eval.py'),'--data','cifar100','--model',j['model'],'--attack',j['attack'],'--config','B','--beta','0.7','--rounds','150','--client_lr','0.05','--device',f'cuda:{j["gpu"]}','--output',str(d)]
     env=os.environ.copy();env.update(TZ='Asia/Shanghai',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
     with (base/f'slot{slot}.log').open('a') as log:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
     ACTIVE[p.pid]=(j['gpu'],reserve);event(f'START {j["path"]} GPU{j["gpu"]} pid={p.pid}');break
   time.sleep(15)
  code=p.wait()
  with LOCK:ACTIVE.pop(p.pid,None)
  event(f'FINISH {j["path"]} exit={code}');report(base)
  if code:failures.append(str(d))
 return failures

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','prepare']);p.add_argument('--batch',type=Path,default=DEFAULT);a=p.parse_args();base=a.batch.resolve();m=prepare(base)
 if a.action!='launch':
  for r in report(base):print(r)
  return
 with (base/'.launcher.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  failures=[]
  with ThreadPoolExecutor(max_workers=4) as pool:
   pending={pool.submit(worker,base,s,m) for s in range(4)}
   while pending:
    done,pending=wait(pending,timeout=60,return_when=FIRST_COMPLETED)
    for t in done:failures.extend(t.result())
    report(base)
  event(f'ALL FINISHED failures={failures}')
  if failures:sys.exit(1)
if __name__=='__main__':main()
