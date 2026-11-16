"""35-point grid. Commands: launch | row N --device cuda:0 | report | plan."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT = Path('/home/risegpu/sunjiahao/Alignins')
BATCH = ROOT/'research_results/RAMP_grid/zero1to5_penalty2to8_20261112'
POINTS = [(k,p) for k in range(1,6) for p in range(2,9)]

def command(k,p,device,output=None):
    return [sys.executable,str(BATCH/'code/ramp_ag_alpha_zero.py'),
        '--data','cifar10','--model','resnet9','--attack','badnet',
        '--num_agents','20','--num_corrupt','4','--poison_frac','0.3',
        '--non_iid','--beta','0.5','--rounds','100','--local_ep','2',
        '--seed','0','--bs','64','--client_lr','0.1','--server_lr','1',
        '--lr_decay','0.99','--ag_topk_ratio','0.001','--topk_ratio',str(k/100),
        '--alpha_threshold','1','--penalty_strength',str(p),'--device',device,
        '--output',str(output or BATCH/f'runs/zero{k}pct_penalty{p}')]

def report():
    with (BATCH/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for k,p in POINTS:
            output=BATCH/f'runs/zero{k}pct_penalty{p}'
            r=dict(zero_pct=k,penalty_strength=p,status='pending',rounds=0,acc_last10='',asr_last10='',asr_peak_50_100='',path=str(output))
            try:
                if (output/'run.json').exists():
                    meta=json.loads((output/'run.json').read_text())
                    r.update(status=meta['status'],rounds=meta.get('observed_rounds',0))
                if r['status']=='completed' and r['rounds']==100:
                    with (output/'tables/metrics.csv').open() as f: data=list(csv.DictReader(f))
                    assert len(data)==100 and int(data[-1]['round'])==100
                    r['acc_last10']=100*sum(float(x['clean_acc']) for x in data[-10:])/10
                    r['asr_last10']=100*sum(float(x['asr']) for x in data[-10:])/10
                    r['asr_peak_50_100']=100*max(float(x['asr']) for x in data if int(x['round'])>=50)
            except Exception as e:
                r['status']='unreadable: '+str(e)
            rows.append(r)
        temporary=BATCH/'summary.csv.tmp'
        with temporary.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        temporary.replace(BATCH/'summary.csv')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        (BATCH/'figures').mkdir(exist_ok=True)
        for key,label in [('acc_last10','Clean ACC'),('asr_last10','ASR')]:
            values=np.full((5,7),np.nan)
            for r in rows:
                if r[key]!='':values[r['zero_pct']-1,r['penalty_strength']-2]=r[key]
            fig,ax=plt.subplots(figsize=(9,5))
            im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap='viridis' if key=='acc_last10' else 'magma')
            ax.set(xticks=range(7),xticklabels=range(2,9),yticks=range(5),yticklabels=[f'{k}%' for k in range(1,6)],xlabel='Penalty strength',ylabel='Zero Top-K',title=label+' (%): mean of rounds 91-100')
            for i in range(5):
                for j in range(7):ax.text(j,i,'—' if np.isnan(values[i,j]) else f'{values[i,j]:.2f}',ha='center',va='center',color='white' if np.isnan(values[i,j]) or values[i,j]<50 else 'black')
            fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(BATCH/f'figures/{key}.png',dpi=170);plt.close(fig)
        lines=['# RAMP grid: zero 1–5%, penalty 2–8','',f'Completed: {sum(r["status"]=="completed" for r in rows)}/35. Only completed 100-round runs enter heatmaps.','', '![ACC](figures/acc_last10.png)','![ASR](figures/asr_last10.png)','', '| Zero % | Penalty | Status | Rounds | ACC last 10 % | ASR last 10 % |','|---|---|---|---|---|---|']
        for r in rows:lines.append(f'| {r["zero_pct"]} | {r["penalty_strength"]} | {r["status"]} | {r["rounds"]} | {r["acc_last10"]} | {r["asr_last10"]} |')
        temp=BATCH/'README.md.tmp';temp.write_text('\n'.join(lines));temp.replace(BATCH/'README.md')

def run_point(k,p,device):
    locks=ROOT/'research_results/.gpu_locks';locks.mkdir(exist_ok=True)
    with (locks/(device.replace(':','_')+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        output=BATCH/f'runs/zero{k}pct_penalty{p}'
        if output.exists():
            try:m=json.loads((output/'run.json').read_text())
            except Exception:m={}
            if m.get('status')=='completed' and m.get('observed_rounds')==100:
                print('SKIP completed',output,flush=True);return True
            print('Preserving incomplete output; inspect before retry:',output,flush=True);return False
        print('START',device,output.name,flush=True)
        with (BATCH/f'driver_logs/{output.name}.log').open('w') as log:
            result=subprocess.run(command(k,p,device),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report()
        print('FINISH',device,output.name,'exit',result.returncode,flush=True)
        return result.returncode==0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','row','report','plan'])
    parser.add_argument('row',type=int,nargs='?',choices=range(1,6))
    parser.add_argument('--device',choices=['cuda:0','cuda:1'],default='cuda:0')
    parser.add_argument('--start-zero',type=int,choices=range(1,6),default=1,help='launch/plan: start at this zero percentage; earlier rows remain untouched')
    args=parser.parse_args()
    if args.action=='report':report();return
    if args.action=='plan':
        for k,p in POINTS:
            if k>=args.start_zero:print(' '.join(command(k,p,'cuda:'+str(POINTS.index((k,p))%2))))
        return
    if args.action=='row':
        if args.row is None:parser.error('row requires 1..5')
        ok=True
        for p in range(2,9):ok=run_point(args.row,p,args.device) and ok
        raise SystemExit(0 if ok else 1)
    # Single launcher only. Two workers dynamically share 35 jobs (18/17 approximately).
    with (BATCH/'launcher.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A grid launcher is already active')
        jobs=queue.Queue()
        for point in POINTS:
            if point[0]>=args.start_zero:jobs.put(point)
        errors=[]
        def worker(device):
            while True:
                try:k,p=jobs.get_nowait()
                except queue.Empty:return
                try:
                    if not run_point(k,p,device):errors.append((k,p))
                except Exception as e:
                    errors.append((k,p));print('ERROR',k,p,str(e),flush=True)
                finally:jobs.task_done()
        threads=[threading.Thread(target=worker,args=(f'cuda:{i}',)) for i in range(2)]
        for t in threads:t.start()
        for t in threads:t.join()
        report()
        print('Grid finished:',BATCH/'README.md','failed/incomplete:',errors,flush=True)
        raise SystemExit(bool(errors))

if __name__=='__main__':main()
