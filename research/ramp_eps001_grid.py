"""Eight concurrent independent ascending-zero sequences, durable lean results."""
import argparse,csv,fcntl,json,os,queue,subprocess,sys,threading,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_results/RAMP_eps001_grid/20261123_100rounds_zero0.1to3_threshold2to0.7'
RATIOS=[.1,.3,.5,1.,2.,3.];THRESHOLDS=[2.,1.5,1.3,1.,.7]
LOCK=threading.RLock()
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def plan():
 return [dict(data=d,beta=b,attack=a,threshold=t,zero_pct=z,status='pending') for t in THRESHOLDS for d,b in [('cifar10',.5),('cifar100',.7)] for a in ['badnet','pgd'] for z in RATIOS]
def report(rows):
 with LOCK:
  atomic(BASE/'state.json',rows)
  keys=['data','beta','attack','threshold','zero_pct','status','gpu','pid','rounds','acc_pct','asr_pct','reason','path']
  with (BASE/'summary.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def figures(rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12})
 for data in ['cifar10','cifar100']:
  for attack in ['badnet','pgd']:
   fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
   for ax,key,title in zip(axs,['acc_pct','asr_pct'],['Final Clean ACC (%)','Final ASR (%)']):
    vals=np.full((6,5),np.nan);lookup={}
    for r in rows:
     if r['data']==data and r['attack']==attack:
      i,j=RATIOS.index(r['zero_pct']),THRESHOLDS.index(r['threshold']);lookup[i,j]=r
      if r['status']=='completed':vals[i,j]=r[key]
    cmap=plt.get_cmap('Blues' if key=='acc_pct' else 'YlOrRd').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(np.ma.masked_invalid(vals),vmin=0,vmax=100,cmap=cmap,aspect='auto');fig.colorbar(im,ax=ax,fraction=.045)
    for i in range(6):
     for j in range(5):
      r=lookup[i,j];v=vals[i,j];txt=f'{v:.2f}' if math.isfinite(v) else {'skipped':'SKIP','pending':'WAIT','running':'RUN','failed':'FAIL'}.get(r['status'],r['status'])
      ax.text(j,i,txt,ha='center',va='center',fontsize=10,color='white' if math.isfinite(v) and v>65 else '#222')
    ax.set(xticks=range(5),xticklabels=THRESHOLDS,yticks=range(6),yticklabels=[f'{z:g}%' for z in RATIOS],xlabel='Coordinate ratio threshold',ylabel='Zero Top-K',title=title)
   fig.suptitle(f'{data.upper()} / {attack} / ResNet9 / eps=0.001 / 100 rounds\nSKIP: smaller zero ratio already reached final ASR < 10%')
   fig.savefig(BASE/f'{data}_{attack}_final.png',dpi=250);plt.close(fig)
def launch():
 BASE.mkdir(parents=True,exist_ok=True)
 with (BASE/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=json.loads((BASE/'state.json').read_text()) if (BASE/'state.json').exists() else plan()
  if any(r['status']=='running' for r in rows):raise RuntimeError('Stale running jobs require inspection before resume')
  atomic(BASE/'batch.json',dict(controller_pid=os.getpid(),python=sys.executable,epsilon=0.001,workers=8,rounds=100,client_lr=.1,seed=0,stop='Final round ASR < .1; skip larger zero ratios only',zero_order=RATIOS,threshold_order=THRESHOLDS,storage='Per run config.json/training.log/metrics.csv; batch state/summary/log and four PNGs'))
  jobs=queue.Queue()
  for t in THRESHOLDS:
   for d in ['cifar10','cifar100']:
    for a in ['badnet','pgd']:jobs.put([r for r in rows if r['data']==d and r['attack']==a and r['threshold']==t])
  report(rows)
  def worker(slot):
   gpu=slot%2
   while True:
    try:seq=jobs.get_nowait()
    except queue.Empty:return
    reason=None
    for r in seq:
     if r['status']=='completed':
      if r['asr_pct']<10:reason=f"zero={r['zero_pct']:g}% final ASR={r['asr_pct']:.6g}% < 10%"
      continue
     if r['status']=='skipped':continue
     if reason:
      with LOCK:r.update(status='skipped',reason=reason);report(rows)
      continue
     dest=BASE/r['data']/r['attack']/f"threshold{r['threshold']:g}"/f"zero{r['zero_pct']:g}pct"
     if dest.exists():
      with LOCK:r.update(status='failed',reason='Existing unfinished run preserved; inspect before retry');report(rows)
      break
     dest.parent.mkdir(parents=True,exist_ok=True)
     cmd=[sys.executable,'-u',str(ROOT/'src/ramp_epsgrid_eval.py'),'--data',r['data'],'--attack',r['attack'],'--beta',str(r['beta']),'--model','resnet9','--rounds','100','--client_lr','0.1','--topk_ratio',str(r['zero_pct']/100),'--alpha_threshold',str(r['threshold']),'--epsilon','0.001','--device',f'cuda:{gpu}','--output',str(dest.resolve())]
     env=os.environ.copy();env.update(TZ='Asia/Shanghai',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
     with LOCK:
      c=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
      r.update(status='running',gpu=gpu,pid=c.pid,path=str(dest));report(rows)
      print('START',r['data'],r['attack'],'threshold',r['threshold'],'zero',r['zero_pct'],'GPU',gpu,'PID',c.pid,flush=True)
     rc=c.wait()
     try:
      config=json.loads((dest/'config.json').read_text());m=list(csv.DictReader((dest/'metrics.csv').open()))
      assert rc==0 and config['status']=='completed' and len(m)==100 and int(m[-1]['round'])==100
      acc,asr=100*float(m[-1]['clean_acc']),100*float(m[-1]['asr']);assert math.isfinite(acc) and math.isfinite(asr)
      with LOCK:r.update(status='completed',rounds=100,acc_pct=acc,asr_pct=asr);report(rows)
      print('DONE',r['data'],r['attack'],r['threshold'],r['zero_pct'],'ACC',acc,'ASR',asr,flush=True)
      if asr<10:reason=f"zero={r['zero_pct']:g}% final ASR={asr:.6g}% < 10%"
     except Exception as e:
      with LOCK:r.update(status='failed',reason=f'Exit {rc}: {type(e).__name__}: {e}');report(rows)
      print('FAILED',r,flush=True);break
    jobs.task_done()
  threads=[threading.Thread(target=worker,args=(slot,)) for slot in range(8)]
  for t in threads:t.start()
  while any(t.is_alive() for t in threads):
   with LOCK:snapshot=[dict(r) for r in rows]
   figures(snapshot);time.sleep(30)
  for t in threads:t.join()
  report(rows);figures(rows)
  print('FINISHED', {s:sum(r['status']==s for r in rows) for s in ['completed','skipped','failed','pending']},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','report']);a=p.parse_args()
 if a.action=='launch':launch()
 else:
  rows=json.loads((BASE/'state.json').read_text())
  if a.action=='report':figures(rows)
  print({s:sum(r['status']==s for r in rows) for s in ['completed','running','pending','skipped','failed']})
