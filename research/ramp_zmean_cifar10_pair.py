"""Independent two-process BadNet/PGD mean-Z penalty experiment."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_results/RAMP_Zmean/cifar10_resnet9_beta0.5_lr0.1_zero4pct_threshold0.7_lambda1_150rounds_20261118'
JOBS=[('badnet',0),('pgd',1)]
def report():
 rows=[]
 for attack,gpu in JOBS:
  d=BASE/attack; f=d/'config.json'; m=json.loads(f.read_text()) if f.exists() else {}
  f=d/'metrics.csv'; rs=list(csv.DictReader(f.open())) if f.exists() else [];tail=rs[-10:]
  rows.append(dict(attack=attack,gpu=gpu,status=m.get('status','waiting'),rounds=len(rs),acc_last10_pct=sum(float(r['clean_acc'])*100 for r in tail)/len(tail) if tail else '',asr_last10_pct=sum(float(r['asr'])*100 for r in tail)/len(tail) if tail else ''))
 with (BASE/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=['# RAMP：自适应 Z 均值惩罚实验','','CIFAR10 / ResNet9 / Non-IID beta0.5 / 初始LR0.1，decay0.99 / 150轮 / 本地2epoch / batch64 / 20客户端，4恶意 / poison0.3 / seed0。','','BadNet：GPU0；PGD：GPU1。PGD沿用项目原生CIFAR10攻击半径与投影规则。','','参考、AlphaGuard Top-0.1%检测保持原样。以未检出异常客户端的Z均值为z_base；对所有客户端 p=exp(-max(0,Z-z_base))，lambda=1，无0.01下限，权重对全部参与者归一化，不乘数据量。仅对检出异常客户端执行相对比值Top-4%且严格>0.7的更新坐标置零；没有整客户端排除。','','每组文件：config.json、training.log、metrics.csv、milestones.csv、curves.png、client_weights.csv、z_baseline.csv。不保存源码快照、模型快照或数据集副本。client_weights.csv包含每个客户端的Z、z_base、超出程度、惩罚系数和最终权重。','','|攻击|GPU|状态|已完成轮数|最近10轮ACC%|最近10轮ASR%|','|---|---|---|---|---|---|']
 for r in rows:text.append('| '+' | '.join(str(v) if not isinstance(v,float) else f'{v:.4f}' for v in r.values())+' |')
 for attack,_ in JOBS:text+=['',f'## {attack}',f'![{attack}]({attack}/curves.png)']
 (BASE/'README.md').write_text('\n'.join(text)+'\n')
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status']);a=p.parse_args()
 if a.action=='status':print(json.dumps(report(),indent=2));return
 with (BASE/'.launcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  children=[]
  for attack,gpu in JOBS:
   assert not (BASE/attack).exists(),'Preserving existing results'
   cmd=[sys.executable,'-u',str(ROOT/'src/ramp_zmean_eval.py'),'--data','cifar10','--model','resnet9','--attack',attack,'--config','B','--beta','0.5','--client_lr','0.1','--rounds','150','--alpha_threshold','0.7','--device',f'cuda:{gpu}','--output',str((BASE/attack).resolve())]
   env=os.environ.copy();env.update(TZ='Asia/Shanghai',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   with (BASE/f'{attack}_launcher.log').open('a') as f:c=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f)
   children.append(c);print('START',attack,'GPU',gpu,'PID',c.pid,flush=True)
  (BASE/'processes.json').write_text(json.dumps(dict(controller_pid=os.getpid(),children=[dict(attack=a,gpu=g,pid=c.pid) for (a,g),c in zip(JOBS,children)]),indent=2))
  while any(c.poll() is None for c in children):report();time.sleep(30)
  report();print('FINISHED exitcodes',[c.returncode for c in children],flush=True)
  if any(c.returncode for c in children):sys.exit(1)
if __name__=='__main__':main()
