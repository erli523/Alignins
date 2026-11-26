"""Read-only analysis of the completed historical grid; no training changes."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/home/risegpu/sunjiahao/Alignins/research_results/RAMP_grid/zero1to5_penalty2to8_20261112')
OUT=BASE/'analysis_actual_zero'
OUT.mkdir(exist_ok=True)
summary=[]; rounds=[]; clients=[]
def truth(s): return str(s).lower()=='true'
def avg(rows,key): return sum(float(r[key]) for r in rows)/len(rows) if rows else 0.
for k in range(1,6):
 for p in range(2,9):
  run=BASE/f'runs/zero{k}pct_penalty{p}'
  meta=json.loads((run/'run.json').read_text())
  assert meta['status']=='completed' and meta['observed_rounds']==100
  dim=sum(x['numel'] for x in meta['layout'])
  data=list(csv.DictReader((run/'tables/clients.csv').open()))
  metrics={int(r['round']):r for r in csv.DictReader((run/'tables/metrics.csv').open())}
  group=[]
  for t in range(1,101):
   rows=[r for r in data if int(r['round'])==t]
   mal=[r for r in rows if r['role']=='malicious'];det=[r for r in mal if truth(r['detected_anomaly'])]
   assert len(rows)==20 and len(mal)==4
   row=dict(zero_pct=k,penalty=p,round=t,dimension=dim,
    malicious_actual_pct=avg(mal,'selected_count')/dim*100,
    malicious_eligible_pct=avg(mal,'above_threshold_count')/dim*100,
    detected_malicious_actual_pct=avg(det,'selected_count')/dim*100 if det else float('nan'),
    malicious_min_pct=min(int(r['selected_count']) for r in mal)/dim*100,
    malicious_max_pct=max(int(r['selected_count']) for r in mal)/dim*100,
    detected_malicious_count=len(det),
    malicious_cap_used_pct=sum(int(r['selected_count']) for r in mal)/sum(int(r['topk_count']) for r in mal)*100,
    all_client_actual_pct=avg(rows,'selected_count')/dim*100,
    benign_actual_pct=avg([r for r in rows if r['role']=='benign'],'selected_count')/dim*100,
    malicious_aggregation_weight=sum(float(r['aggregation_weight']) for r in mal),
    clean_acc_pct=float(metrics[t]['clean_acc'])*100,asr_pct=float(metrics[t]['asr'])*100)
   rounds.append(row);group.append(row)
   if t==100:
    for r in rows:
     clients.append(dict(zero_pct=k,penalty=p,round=t,dimension=dim,client_id=r['client_id'],role=r['role'],detected=r['detected_anomaly'],topk_count=r['topk_count'],eligible_count=r['above_threshold_count'],zeroed_count=r['selected_count'],actual_pct=int(r['selected_count'])/dim*100,aggregation_weight=r['aggregation_weight']))
  final=group[-1];tail=group[-10:]
  s=dict(zero_pct=k,penalty=p,dimension=dim,nominal_topk_count=int(rows[0]['topk_count']))
  for key in ['malicious_actual_pct','malicious_eligible_pct','detected_malicious_actual_pct','malicious_min_pct','malicious_max_pct','detected_malicious_count','malicious_cap_used_pct','all_client_actual_pct','benign_actual_pct','malicious_aggregation_weight','clean_acc_pct','asr_pct']:
   s[key+'_final']=final[key];s[key+'_last10']=avg(tail,key)
  summary.append(s)
for name,data in [('summary',summary),('per_round',rounds),('final_clients',clients)]:
 with (OUT/f'{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(15,5.8),layout='constrained')
for ax,suffix,title in zip(axes,['final','last10'],['Final round (100)','Mean of rounds 91-100']):
 a=np.array([[next(s for s in summary if s['zero_pct']==k and s['penalty']==p)['malicious_actual_pct_'+suffix] for p in range(2,9)] for k in range(1,6)])
 im=ax.imshow(a,vmin=0,vmax=5,cmap='YlGnBu',aspect='auto')
 for i in range(5):
  for j in range(7):ax.text(j,i,f'{a[i,j]:.2f}%',ha='center',va='center',color='white' if a[i,j]>3 else 'black')
 ax.set(xticks=range(7),xticklabels=range(2,9),yticks=range(5),yticklabels=[f'{k}%' for k in range(1,6)],xlabel='Penalty strength',ylabel='Configured zero Top-K',title=title)
fig.colorbar(im,ax=axes,label='Actual zeroed coordinates / model dimension (%)')
fig.suptitle('Historical grid: actual zeroing per malicious client (mean of 4 clients)\nCIFAR10 / ResNet9 / beta=0.5 / alpha threshold=1')
fig.savefig(OUT/'actual_zero_heatmaps.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
colors=plt.cm.viridis(np.linspace(.05,.95,7))
for ax,k in zip(axes.flat,range(1,6)):
 for color,p in zip(colors,range(2,9)):
  rs=[r for r in rounds if r['zero_pct']==k and r['penalty']==p]
  ax.plot([r['round'] for r in rs],[r['malicious_actual_pct'] for r in rs],color=color,label=f'Penalty {p}',lw=1.5)
 ax.axhline(k,color='black',ls='--',lw=1,label='Configured cap')
 ax.set(title=f'Configured Top-K = {k}%',xlabel='Global round',ylabel='Actual zeroed (%)',ylim=(0,k*1.1));ax.grid(alpha=.2)
axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
fig.suptitle('Actual zeroing throughout training: mean across all 4 malicious clients\nUndetected malicious clients contribute 0; percentages use one model dimension')
fig.savefig(OUT/'actual_zero_trajectories.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
for ax,p in zip(axes,[5,7]):
 for k,color in [(1,'#4477aa'),(4,'#cc6677')]:
  rs=[r for r in rounds if r['zero_pct']==k and r['penalty']==p]
  ax.plot([r['round'] for r in rs],[r['malicious_actual_pct'] for r in rs],color=color,label=f'Top-{k}%: actually zeroed')
  ax.plot([r['round'] for r in rs],[r['malicious_eligible_pct'] for r in rs],color=color,ls=':',label=f'Top-{k}%: meets coordinate threshold')
  ax.axhline(k,color=color,ls='--',alpha=.5)
 ax.set(title=f'Penalty strength {p}',xlabel='Global round',ylabel='Mean of 4 malicious clients (%)',ylim=(0,4.5));ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle('A / B comparison: coordinate threshold eligibility vs actual intervention\nSolid = applied after client detection; dotted = eligible regardless of detection; dashed = cap')
fig.savefig(OUT/'AB_zeroing_explained.png',dpi=180);plt.close(fig)

lines=['# Historical grid: actual coordinate zeroing','','Source: '+str(BASE),'','35 completed runs; CIFAR10 / ResNet9 / BadNet / beta0.5 / 100 rounds / threshold1. This is NOT the later CIFAR100 beta0.7 batch.','','## Definitions','','- D = sum of recorded trainable layout dimensions (not inferred from rounded Top-K).','- Main statistic = sum of zeroed coordinate counts over 4 malicious clients / (4 * D) * 100%. An undetected malicious client contributes zero.','- Eligible statistic counts Top-K coordinates meeting the coordinate threshold, before client detection.','- Conditional statistic averages only detected malicious clients.','- Counts are client-coordinate pairs; no union of global model coordinates is inferred. Zeroing local updates does not force corresponding global model parameters to zero.','- Final = round100; last10 = rounds91-100 mean. Coordinate count alone does not measure update energy removed or the effect of penalty weights.','','![Actual ratios](actual_zero_heatmaps.png)','','![Training trajectories](actual_zero_trajectories.png)','','![A B explanation](AB_zeroing_explained.png)','','| Top-K | Penalty | Final actual % | Last10 actual % | Last10 cap used % | Last10 detected malicious /4 | Last10 ACC % | Last10 ASR % |','|---|---|---|---|---|---|---|---|']
for s in summary:
 lines.append('| '+ ' | '.join([str(s['zero_pct'])+'%',str(s['penalty'])]+[f"{s[key]:.3f}" for key in ['malicious_actual_pct_final','malicious_actual_pct_last10','malicious_cap_used_pct_last10','detected_malicious_count_last10','clean_acc_pct_last10','asr_pct_last10']])+' |')
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print('OUTPUT',OUT)
for k,p in [(1,5),(4,7)]:print('SELECTED',json.dumps(next(s for s in summary if s['zero_pct']==k and s['penalty']==p)))
print('RANGE final',min(s['malicious_actual_pct_final'] for s in summary),max(s['malicious_actual_pct_final'] for s in summary))
print('RANGE last10',min(s['malicious_actual_pct_last10'] for s in summary),max(s['malicious_actual_pct_last10'] for s in summary))
