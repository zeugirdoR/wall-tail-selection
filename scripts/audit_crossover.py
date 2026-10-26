import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from wall_tail_selection import crossover as c
cfg=json.loads(Path('configs/crossover-pilot.json').read_text());cfg['seed']=2026102699;stats=dict(cases=0,max_score_refinement=0.,max_risk_refinement=0.,max_adaptive_error=0.,max_independent_test_z=0.,fallback_changes=0)
for ci in range(len(cfg['cells'])):
 x,mu,family=c.observations(cfg,ci,0);a,ref,fa,_,stack=c.fit(x,cfg,family);b,ref2,fb,_,st2=c.fit(x,dict(cfg,nodes=512),family);stats['max_score_refinement']=max(stats['max_score_refinement'],float(np.max(abs(a-b))));stats['fallback_changes']+=int(np.any(fa!=fb));stats['cases']+=1
 r=c.risks(mu,family,ref,a[2],stack,64);r2=c.risks(mu,family,ref2,b[2],st2,128);r3=c.risks(mu,family,ref2,b[2],st2,256);stats['max_risk_refinement']=max(stats['max_risk_refinement'],float(np.max(abs(r-r2))),float(np.max(abs(r2-r3))))
 if cfg['cells'][ci]['value']==0. or cfg['cells'][ci]['regime']=='regular':
  peak=c.mle(x,2,family)[1];v=quad(lambda mu:math.exp(float(c.logratio(x,np.array([mu]),family).sum())-peak)/2,0,2,epsabs=1e-12,epsrel=1e-11,limit=200)[0];stats['max_adaptive_error']=max(stats['max_adaptive_error'],abs(peak+math.log(v)-a[2]))
  rng=np.random.default_rng(np.random.SeedSequence([2026102699,ci,999]));y=c.sg.sample(rng,mu,65536) if family=='mixture' else rng.normal(size=65536)*math.sqrt(1+mu*mu)
  losses=[]
  for j in range(0,len(y),256):
   yy=y[j:j+256];lr=logsumexp(c.logratio(yy,ref2['mu'],family)+ref2['logpost'],axis=1);base=c.sg.logphi(yy);losses.append(np.array([-base,-base-lr,-base-c.sg.mixture_ratio(lr,float(c.expit(b[2]))),-base-c.sg.mixture_ratio(lr,st2)]).T)
  loss=np.concatenate(losses);z=abs(loss.mean(axis=0)-r3)/(loss.std(axis=0,ddof=1)/math.sqrt(len(y)));stats['max_independent_test_z']=max(stats['max_independent_test_z'],float(z.max()))
assert stats['max_score_refinement']<1e-7 and stats['max_risk_refinement']<1e-6 and stats['max_adaptive_error']<1e-7 and stats['max_independent_test_z']<5 and stats['fallback_changes']==0,stats
Path('runs/crossover-numerical-audit.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
