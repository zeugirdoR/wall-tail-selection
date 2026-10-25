"""One shared resolution constant; prediction-blind finite-pair calibration."""
import math
import numpy as np
from .geometry import resolution_log_step
from .mechanism import task

def log_counts(shapes,ns,domains,cs,rounded=True):
 out=np.empty((len(cs),len(domains),len(ns),len(shapes)))
 for ci,c in enumerate(cs):
  for ni,n in enumerate(ns):
   for j,a in enumerate(shapes):
    step=resolution_log_step(a,n,float(c))
    for bi,(L,U) in enumerate(domains):
     ratio=math.log(U/L)/step
     out[ci,bi,ni,j]=math.log(math.floor(ratio)+1 if rounded else ratio)
 return out

def objective(costs,config,penalties):
 models=['exponential' if a is None else f'endpoint_a{a:g}' for a in config['shapes']]
 groups={}
 for r in costs:groups.setdefault((int(r['domain']),r['truth'],int(r['n']),int(r['replicate'])),[]).append(r)
 bycell={};losses=[];excluded=0
 for (bi,tr,n,rep),rows in groups.items():
  finite=[r for r in rows if r['status']=='finite'];ni=config['sample_sizes'].index(n)
  if len(finite)<2:excluded+=1;continue
  vals=[]
  for j,r in enumerate(finite):
   for s in finite[j+1:]:
    diff=penalties[:,bi,ni,models.index(r['model'])]-penalties[:,bi,ni,models.index(s['model'])]
    vals.append((diff-(float(r['occam'])-float(s['occam'])))**2)
  loss=np.mean(vals,axis=0);bycell.setdefault((bi,tr,n),[]).append(loss)
  losses.append(dict(domain=bi,truth=tr,n=n,replicate=rep,pairs=len(vals),loss=loss))
 curve=np.mean([np.mean(v,axis=0) for v in bycell.values()],axis=0)
 return curve,losses,dict(included_cells=len(bycell),excluded_cells=len(config['domains'])*len(models)*len(config['sample_sizes'])-len(bycell),included_datasets=len(losses),excluded_datasets=excluded,finite_pairs=sum(r['pairs'] for r in losses))

def choose(curve,cs):
 valid=np.flatnonzero(curve<=np.min(curve)+1e-12)
 return min(valid,key=lambda i:(abs(math.log(float(cs[i]))),float(cs[i])))

def paired_task(args):
 cfg,ti,n,rep,bi=args
 tuned=task(args)
 baseline=dict(cfg,resolution_c=1.)
 base=task((baseline,ti,n,rep,bi))
 assert tuned[3]==base[3]
 for a,b in zip(tuned[0],base[0]):
  assert a['model']==b['model'] and a['status']==b['status'] and a['log_evidence']==b['log_evidence']
 extras=[dict(r,method='geometry_c1') for r in base[1] if r['method']=='geometry']
 return tuned[0],tuned[1]+extras,tuned[2],tuned[3]
