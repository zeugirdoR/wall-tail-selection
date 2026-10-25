import json,math
from pathlib import Path
from scipy.integrate import quad
import numpy as np
from wall_tail_selection.singular import task,dataset,reference,mle,logratio
out=[]
for path in ['configs/singular-pilot.json','configs/singular-boundary-pilot.json']:
 cfg=json.loads(Path(path).read_text());stats=dict(config=path,cases=0,max_logbf_refinement=0.,max_predictive_loss_refinement=0.,max_adaptive_logbf_error=0.,choice_changes=0)
 for ci in range(len(cfg['cells'])):
  for rep in range(cfg['replicates']):
   a=task((cfg,ci,rep));b=task((dict(cfg,nodes=512),ci,rep));x,_,_=dataset(cfg,ci,rep);peak=mle(x,2)[1]
   exact=peak+math.log(quad(lambda mu:math.exp(float(logratio(x,np.array([mu])).sum())-peak)/2,0,2,epsabs=1e-12,epsrel=1e-11,limit=200)[0]);stats['cases']+=1;stats['max_adaptive_logbf_error']=max(stats['max_adaptive_logbf_error'],abs(exact-a[0]['log_bf']));stats['max_logbf_refinement']=max(stats['max_logbf_refinement'],abs(a[0]['log_bf']-b[0]['log_bf']))
   for r,s in zip(a[2],b[2]):
    stats['max_predictive_loss_refinement']=max(stats['max_predictive_loss_refinement'],abs(r['log_loss']-s['log_loss']))
    if r['method'] not in ['bma','stacking']:stats['choice_changes']+=r['weight_m1']!=s['weight_m1']
 assert stats['max_logbf_refinement']<1e-7 and stats['max_adaptive_logbf_error']<1e-7 and stats['max_predictive_loss_refinement']<1e-6 and stats['choice_changes']==0
 out.append(stats)
Path('runs/singular-numerical-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
