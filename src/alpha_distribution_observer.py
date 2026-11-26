"""Read-only, exact per-coordinate distribution observer; no RNG or model writes."""
import csv,json,logging
from pathlib import Path
import numpy as np

QUANTILES=[0,.01,.05,.5,.9,.95,.99,.999,1]
QNAMES=['min','p01','p05','median','p90','p95','p99','p999','max']
EDGES=np.array([-np.inf,-100,-10,-5,-2,-1,-.8,-.7,-.5,-.1,0,.1,.25,.5,.7,.8,1,1.5,2,3,5,10,20,50,100,np.inf])

class Observer:
 def __init__(self,output):
  self.output=Path(output);self.handles={};self.writers={}
 def emit(self,name,row):
  if name not in self.writers:
   f=(self.output/(name+'.csv')).open('w',newline='');self.handles[name]=f
   self.writers[name]=csv.DictWriter(f,fieldnames=list(row));self.writers[name].writeheader()
  self.writers[name].writerow(row)
 def record(self,rnd,method,updates):
  ref=method.observed_ref.detach().cpu().numpy()
  dim=ref.size;ids=method._current_client_ids;kept=set(method.last_keep.cpu().tolist())
  refids=[ids[i] for i in method.observed_reference_indices.cpu().tolist()]
  self.emit('reference',dict(round=rnd,dimension=dim,reference_clients=json.dumps(refids),reference_zero_count=int((ref==0).sum()),reference_small_count=int((np.abs(ref)<.001).sum()),epsilon=.001))
  denom=np.abs(ref)+.001
  for pos,cid in enumerate(ids):
   native=method.observed_alpha[pos].detach().cpu().numpy()
   delta=updates[cid].detach().cpu().numpy()
   for mode,x in [('native_signed_alpha',native),('native_abs_alpha',np.abs(native)),('zero_abs_ratio',np.abs(delta)/denom)]:
    assert x.size==dim and np.isfinite(x).all()
    q=np.quantile(x,QUANTILES)
    row=dict(round=rnd,client_id=cid,role='malicious' if cid<method.args.num_corrupt else 'benign',mode=mode,dimension=dim,reference=cid in refids,kept=pos in kept,**dict(zip(QNAMES,map(float,q))))
    row.update(negative_pct=float((x<0).mean()*100),zero_pct=float((x==0).mean()*100))
    for t in [.7,.8,1.,2.,5.,10.]:row['gt_'+str(t)+'_pct']=float((x>t).mean()*100)
    # Top percentages use absolute alpha for native alpha, positive ratios otherwise.
    rank=np.abs(x) if mode=='native_signed_alpha' else x
    ks=[max(1,int(dim*f)) for f in [.001,.01,.04]]
    partition=np.partition(rank,[dim-k for k in ks])
    for label,k in zip(['0p1pct','1pct','4pct'],ks):
     v=partition[-k:];row['top_'+label+'_count']=k;row['top_'+label+'_min']=float(v.min());row['top_'+label+'_max']=float(v.max());row['top_'+label+'_mean']=float(v.mean(dtype=np.float64))
    self.emit('coordinate_distributions',row)
    counts,_=np.histogram(x,bins=EDGES)
    assert int(counts.sum())==dim
    for lo,hi,count in zip(EDGES[:-1],EDGES[1:],counts):self.emit('histograms',dict(round=rnd,client_id=cid,role=row['role'],mode=mode,bin_lower=float(lo),bin_upper=float(hi),count=int(count),pct=100*int(count)/dim))
   if cid<method.args.num_corrupt:
    logging.info('Alpha distribution Global %d | Client %d malicious | native |alpha| P99=%.6g | zero-ratio P99=%.6g | kept=%s',rnd,cid,float(np.quantile(np.abs(native),.99)),float(np.quantile(np.abs(delta)/denom,.99)),pos in kept)
  for f in self.handles.values():f.flush()
  method.observed_alpha=None;method.observed_ref=None
 def plots(self):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  rows=list(csv.DictReader((self.output/'coordinate_distributions.csv').open()))
  for mode in ['native_abs_alpha','zero_abs_ratio']:
   fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
   for role,color in [('malicious','#cc4444'),('benign','#3377bb')]:
    data=[r for r in rows if r['mode']==mode and r['role']==role]
    ts=sorted({int(r['round']) for r in data})
    mean=lambda key:[np.mean([float(r[key]) for r in data if int(r['round'])==t]) for t in ts]
    for key,style in [('median','-'),('p99','--'),('p999',':')]:axes[0].plot(ts,mean(key),style,color=color,label=role+' '+key)
    for key,style in [('gt_0.7_pct','-'),('gt_1.0_pct','--'),('gt_2.0_pct',':')]:axes[1].plot(ts,mean(key),style,color=color,label=role+' '+key)
    for key,style in [('top_0p1pct_min','-'),('top_1pct_min','--'),('top_4pct_min',':')]:axes[2].plot(ts,mean(key),style,color=color,label=role+' '+key)
   for ax,title in zip(axes,['Coordinate quantiles (mean per client)','Coordinates strictly above thresholds (%)','Lower boundary of absolute Top-K']):ax.set(title=title,xlabel='Global round');ax.grid(alpha=.2);ax.legend(fontsize=7)
   fig.suptitle('Original AlphaGuard / CIFAR100 ResNet9 beta0.7 / '+mode)
   fig.savefig(self.output/(mode+'_trends.png'),dpi=160);plt.close(fig)
  (self.output/'README.md').write_text('''# Alpha coordinate distribution experiment

CIFAR100 / ResNet9 / BadNet / beta0.7 / LR0.1 / 150 rounds / clients20 malicious4 / batch64 local_ep2 seed0.
Current project native AlphaGuard runs unchanged; all client updates are observed before filtering. No RAMP zeroing. The detector and final aggregation use native code.

Files: coordinate_distributions.csv (per client, round and formula); histograms.csv (exact interval counts); reference.csv (reference IDs and zero/small denominators); metrics.csv; config.json; training.log; milestones.csv; curves.png.

Three separate formulas:
- native_signed_alpha: exact output of native stage2. Signed epsilon clipping is used; zero-reference NaN/Inf become 0/+1/-1 through native nan_to_num. Quantiles therefore describe the actual post-processing result used by the defense.
- native_abs_alpha: absolute value of the same native output, used by native Top-K scoring.
- zero_abs_ratio: abs(client update)/(abs(reference)+0.001), diagnostic only. Not applied to aggregation.

Quantiles are exact across every coordinate, not sampled. Intervals are [lower,upper). Counts include zero-valued coordinates. Strict > thresholds use all D coordinates as denominator. Histograms retain separate client identities and roles. Top-K boundaries use absolute values for signed alpha, and ratios otherwise. Trend plots average per-client quantiles, not pooled-coordinate quantiles.
Ground-truth roles are used for reporting only and do not restrict reference selection. reference.csv records exactly-zero references and small-reference counts to identify native numerical replacement effects.

![Native alpha](native_abs_alpha_trends.png)
![Zero ratio](zero_abs_ratio_trends.png)
''')
 def close(self):
  for f in self.handles.values():f.close()
