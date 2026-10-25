"""All-cell frozen-benchmark summaries; no refitting or favorable-cell selection."""
import csv,json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete'
def read(name):return list(csv.DictReader((p/name).open()))
def write(name,rows):
 with (p/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stat(v):
 v=np.asarray(v,float);return float(v.mean()),float(v.std(ddof=1)/math.sqrt(len(v)))
ev=read('evidence.csv');pr=read('predictions.csv');raw=read('datasets.csv');groups=defaultdict(list)
for r in ev:groups[(r['cell'],r['method'])].append(r)
es=[]
for (cell,method),rows in groups.items():
 r=rows[0];v=np.array([float(x['error']) for x in rows]);bias,se=stat(v);es.append(dict(cell=cell,n=r['n'],regime=r['regime'],separation=r['separation'],method=method,bias=bias,bias_mc_se=se,mae=float(np.abs(v).mean()),max_abs_error=float(np.abs(v).max()),within_point1_fraction=float(np.mean(np.abs(v)<=.1))))
write('analysis_evidence.csv',es)
groups=defaultdict(list)
for r in pr:groups[(r['cell'],r['method'])].append(r)
ps=[];contrasts=[]
for (cell,method),rows in groups.items():
 r=rows[0];mean,se=stat([x['log_loss'] for x in rows]);ps.append(dict(cell=cell,n=r['n'],regime=r['regime'],separation=r['separation'],method=method,mean_log_loss=mean,mc_se_log_loss=se,mean_weight_m1=float(np.mean([float(x['weight_m1']) for x in rows])),choice_disagreement_exact=float(np.mean([int(x['disagree_exact']) for x in rows])) if rows[0]['disagree_exact'] else None,truth_label_error=float(np.mean([int(x['truth_label_error']) for x in rows])) if rows[0]['truth_label_error'] else None))
 for baseline in ['exact','bma','stacking','shape_ablation']:
  base={x['replicate']:x for x in groups[(cell,baseline)]};diff=[float(x['log_loss'])-float(base[x['replicate']]['log_loss']) for x in rows];d,se=stat(diff);contrasts.append(dict(cell=cell,n=r['n'],regime=r['regime'],separation=r['separation'],method=method,baseline=baseline,mean_difference=d,paired_mc_se=se,below_minus_point001=d<-.001,above_plus_point001=d>.001))
write('analysis_predictions.csv',ps);write('analysis_risk_contrasts.csv',contrasts)
# Paired absolute evidence error comparison isolates the shape ablation.
index={(r['cell'],r['replicate'],r['method']):r for r in ev};abl=[]
for cell in sorted(set(r['cell'] for r in ev),key=int):
 rows=[r for r in ev if r['cell']==cell and r['method']=='local_shape'];v=[abs(float(r['error']))-abs(float(index[(cell,r['replicate'],'shape_ablation')]['error'])) for r in rows];mean,se=stat(v);abl.append(dict(cell=cell,n=rows[0]['n'],regime=rows[0]['regime'],separation=rows[0]['separation'],absolute_error_difference=mean,paired_mc_se=se))
write('analysis_shape_ablation.csv',abl)
result=dict(cases=len(raw),replicates=m['config']['replicates'],evidence_by_regime=[],local_shape_risk_counts=[],timing={})
for regime in ['local','boundary','regular']:
 for method in ['local_shape','shape_ablation','rlct_only','bic']:
  rows=[r for r in ev if r['regime']==regime and r['method']==method];v=np.array([float(x['error']) for x in rows]);result['evidence_by_regime'].append(dict(regime=regime,method=method,mae=float(np.abs(v).mean()),bias=float(v.mean()),max_abs_error=float(np.abs(v).max()),within_point1_fraction=float(np.mean(np.abs(v)<=.1))))
 for baseline in ['exact','bma','stacking','shape_ablation']:
  rows=[r for r in contrasts if r['regime']==regime and r['method']=='local_shape' and r['baseline']==baseline];v=[r['mean_difference'] for r in rows];result['local_shape_risk_counts'].append(dict(regime=regime,baseline=baseline,lower=sum(x<0 for x in v),tied=sum(x==0 for x in v),higher=sum(x>0 for x in v),below_minus_point001=sum(x<-.001 for x in v),above_plus_point001=sum(x>.001 for x in v)))
for field in ['reference_fit_seconds','local_shape_seconds','loo_waic_stacking_seconds','prediction_seconds']:
 vals=[float(r[field]) for r in raw];result['timing'][field]=dict(median=float(np.median(vals)),sum=float(np.sum(vals)))
(p/'analysis_overview.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
