"""Prespecified all-cell/region analysis with fixed simultaneous coverage budget."""
import csv,json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from wall_tail_selection.crossover import SCORES
from wall_tail_selection.crossover_inference import selections,risk_values,RISK_BASELINES,STRUCT_BASELINES,paired_risk,paired_binary,cp
p=Path(sys.argv[1]);fp=Path(sys.argv[2]);m=json.loads((p/'manifest.json').read_text());cfg=m['config'];z=np.load(p/'records.npz');f=json.loads(fp.read_text());native,cal=selections(cfg,z,f['thresholds']);cirows=[];summ=[];approx=[];null={};used=0
native_names=[x for x in SCORES if x not in ['lrt','moment']]+['bma','stacking'];gn= risk_values(z,native,'eta_laplace')
def save(name,rows):
 with (p/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for ci,cell in enumerate(cfg['cells']):
 mask=z['cell']==ci;n=int(mask.sum());common=dict(cell=ci,**cell);isnull=cell['regime']=='local' and cell['value']==0
 for name in native_names:
  r=risk_values(z,native,name)[mask];idx=SCORES.index(name) if name in SCORES else None
  summ.append(dict(common,method=name,mean_risk=float(r.mean()),risk_mc_se=float(r.std(ddof=1)/math.sqrt(n)),native_m1_rate=float(native[mask,idx].mean()) if idx is not None else None,calibrated_m1_rate=float(cal[mask,idx].mean()) if idx is not None else None))
 for name in RISK_BASELINES:
  d=(gn-risk_values(z,native,name))[mask];mean,se,lo,hi=paired_risk(d);used+=1;cirows.append(dict(common,target='risk',baseline=name,difference=mean,mc_se=se,lower=lo,upper=hi,practical_margin=.001))
 for name in STRUCT_BASELINES:
  idx=12 if name=='analytic_score' else SCORES.index(name);d=cal[mask,0].astype(int)-cal[mask,idx].astype(int);mean,lo,hi=paired_binary(d);used+=1
  row=dict(common,target='null_excess' if isnull else 'power',baseline=name,difference=mean,mc_se=float(d.std(ddof=1)/math.sqrt(n)),lower=lo,upper=hi,practical_margin=.01 if isnull else .05);cirows.append(row)
  if isnull:null[(cell['family'],cell['n'],name)]=row
 if isnull:
  for j,name in enumerate(SCORES+['analytic_score']):
   k=int(cal[mask,j].sum());lo,hi=cp(k,n);used+=1;cirows.append(dict(common,target='null_rate',baseline=name,difference=k/n,mc_se=math.sqrt(k/n*(1-k/n)/n),lower=lo,upper=hi,practical_margin=.075))
 for j,name in enumerate(SCORES):
  if name in ['loo','waic','lrt','moment']:continue
  err=z['scores'][mask,j]-z['scores'][mask,2];approx.append(dict(common,method=name,mean_error=float(err.mean()),mae=float(abs(err).mean()),max_error=float(abs(err).max()),fallback_rate=float((z['fallback'][mask,j]>0).mean()) if j<2 else None,choice_disagreement_exact=float(np.mean(native[mask,j]!=native[mask,2]))))
assert used<=1000,used
save('analysis_intervals.csv',cirows);save('analysis_predictive_summary.csv',summ);save('analysis_evidence.csv',approx)
regions=[];claims=[]
for fam in ['mixture','variance']:
 for reg in ['local','boundary','regular']:
  for target,baselines in [('risk',RISK_BASELINES),('power',STRUCT_BASELINES)]:
   for b in baselines:
    rows=[r for r in cirows if r['family']==fam and r['regime']==reg and r['target']==target and r['baseline']==b];mean=float(np.mean([r['difference'] for r in rows]));lo=float(np.mean([r['lower'] for r in rows]));hi=float(np.mean([r['upper'] for r in rows]));guard=True
    if target=='power':
     guard=all(null[(fam,n,b)]['upper']<=.01 for n in [32,128,512]) and all(r['upper']<=.075 for r in cirows if r['family']==fam and r['target']=='null_rate' and r['baseline']=='eta_laplace')
    success=hi<-.001 if target=='risk' else lo>.05 and guard
    row=dict(family=fam,region=reg,target=target,baseline=b,cells=len(rows),difference=mean,lower=lo,upper=hi,null_guard=guard,meets_practical_and_simultaneous_rule=success);regions.append(row)
    if success:claims.append(row)
save('analysis_regions.csv',regions)
nullrates=[r for r in cirows if r['target']=='null_rate'];save('analysis_null_rates.csv',nullrates)
# Known monotone controls must have identical calibrated choices, up to no ties at the upper-tail cutoff.
checks={}
for name in ['bic','rlct_only','sbic']:checks[name+'_versus_lrt_disagreements']=int(np.sum(cal[:,SCORES.index(name)]!=cal[:,7]))
summary=dict(cases=len(z['cell']),replicates=cfg['replicates'],interval_statements=used,bonferroni_budget=1000,calibration_job=f['calibration_job_id'],confirmation_job=m['slurm_job_id'],uncapped_required=f['uncapped_required'],size_capped=f['capped'],fallback_counts=z['fallback'].astype(bool).sum(axis=0).tolist(),positive_region_claims=claims,monotone_controls=checks,timing_medians=np.median(z['timing'],axis=0).tolist(),elapsed_seconds=m['elapsed_seconds'])
(p/'analysis_overview.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
