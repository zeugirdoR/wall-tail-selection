"""Fixed-prior symmetric Gaussian-mixture crossover, stable log-space reference."""
import math,time,hashlib
import numpy as np
from scipy.special import roots_legendre,logsumexp,expit,gamma
from scipy.integrate import quad
from scipy.optimize import brentq
LOG2=math.log(2);LOG2PI=math.log(2*math.pi)
def logcosh(x):return np.logaddexp(x,-x)-LOG2
def logphi(x):return -.5*np.asarray(x)**2-.5*LOG2PI
def logratio(x,mu):return logcosh(np.asarray(x)[:,None]*np.asarray(mu)[None,:])-.5*np.asarray(mu)[None,:]**2
def local_integral(z):
 peak=max(z,0.)**2/2;mode=math.sqrt(math.sqrt(2)*max(z,0.))
 f=lambda t:math.exp(t*t*z/math.sqrt(2)-t**4/4-peak)
 if mode>0:
  a,ea=quad(f,0,mode,epsabs=1e-12,epsrel=1e-11,limit=200);b,eb=quad(f,mode,math.inf,epsabs=1e-12,epsrel=1e-11,limit=200)
 else:a,ea=0.,0.;b,eb=quad(f,0,math.inf,epsabs=1e-12,epsrel=1e-11,limit=200)
 if a+b<=0 or (ea+eb)/(a+b)>1e-8:raise ArithmeticError('local integral precision')
 return peak+math.log(a+b)
def mle(x,U):
 n=len(x)
 def d(mu):return float(np.sum(x*np.tanh(mu*x)/mu)-n) if mu else float(np.sum(x*x)-n)
 mu=0. if d(0)<=0 else U if d(U)>=0 else brentq(d,1e-12,U,xtol=1e-13)
 return mu,float(logratio(x,np.array([mu])).sum())
def reference(x,U=2.,nodes=256):
 q,w=roots_legendre(nodes);mu=(q+1)*U/2;logw=np.log(w/2);li=logratio(x,mu);ll=li.sum(axis=0);logbf=float(logsumexp(logw+ll));post=logw+ll-logbf
 return dict(mu=mu,logpost=post,logbf=logbf,li=li)
def predictive_ratio(y,ref):
 # Bounded blocks avoid oversized test-by-grid arrays.
 return np.concatenate([logsumexp(logratio(y[i:i+256],ref['mu'])+ref['logpost'],axis=1) for i in range(0,len(y),256)])
def predictive_scores(ref):
 li=ref['li'];lp=ref['logpost'];w=np.exp(lp);mean=li@w;var=((li-mean[:,None])**2)@w
 lppd=logsumexp(li+lp,axis=1);loo=-logsumexp(lp-li,axis=1)
 return loo,float(loo.sum()),float(np.sum(lppd-var))
def stacking_weight(logratios):
 # Derivative of sum log((1-w)+w*exp(r)), evaluated stably.
 def derivative(w):
  if w==0:return float(np.expm1(logratios).sum())
  if w==1:return float((-np.expm1(-logratios)).sum())
  return float(((expit(logratios+math.log(w/(1-w)))-w)/(w*(1-w))).sum())
 if derivative(0)<=0:return 0.
 if derivative(1)>=0:return 1.
 return float(brentq(derivative,1e-12,1-1e-12,xtol=1e-12))
def mixture_ratio(lr,w):
 if w==0:return np.zeros_like(lr)
 if w==1:return lr
 return np.logaddexp(math.log1p(-w),math.log(w)+lr)
def sample(rng,mu,n):return rng.normal(size=n)+mu*rng.choice([-1.,1.],size=n)
def dataset(cfg,ci,rep):
 cell=cfg['cells'][ci];n=cell['n'];mu=cell['value']*n**(-.25) if cell['regime']=='local' else cell['value']*(math.log(n)/n)**.25 if cell['regime']=='boundary' else cell['value']
 x=sample(np.random.default_rng(np.random.SeedSequence([cfg['seed'],ci,rep,0])),mu,n)
 y=sample(np.random.default_rng(np.random.SeedSequence([cfg['seed'],ci,rep,1])),mu,cfg['test_size'])
 return x,y,mu

def task(args):
 cfg,ci,rep=args;x,y,truth=dataset(cfg,ci,rep);n=len(x);U=cfg['prior_upper'];start=time.perf_counter();ref=reference(x,U,cfg['nodes']);muhat,peak=mle(x,U);fitsec=time.perf_counter()-start
 z=float(np.sum(x*x-1)/math.sqrt(2*n));start=time.perf_counter();local=-.25*math.log(n)-math.log(U)+local_integral(z);localsec=time.perf_counter()-start
 evidence=dict(exact=ref['logbf'],bic=peak-.5*math.log(n),rlct_only=peak-.25*math.log(n),local_shape=local,shape_ablation=max(z,0.)**2/2-.25*math.log(n)-math.log(U)+math.log(gamma(.25)/(2*math.sqrt(2))))
 start=time.perf_counter();loo,looscore,waic=predictive_scores(ref);stack=stacking_weight(loo);cvsec=time.perf_counter()-start
 weights={k:float(v>0) for k,v in evidence.items()};weights.update(loo=float(looscore>0),waic=float(waic>0),bma=float(expit(ref['logbf'])),stacking=stack,always_m0=0.,always_m1=1.)
 start=time.perf_counter();lr=predictive_ratio(y,ref);base=logphi(y);oracle=base+logratio(y,np.array([truth])).ravel();common=dict(cell=ci,n=n,regime=cfg['cells'][ci]['regime'],separation=cfg['cells'][ci]['value'],true_mu=truth,replicate=rep)
 preds=[dict(common,method=k,weight_m1=w,log_loss=float(-np.mean(base+mixture_ratio(lr,w))),excess_vs_oracle=float(np.mean(oracle-base-mixture_ratio(lr,w))),disagree_exact=(int(w!=weights['exact']) if k not in ['bma','stacking'] else None),truth_label_error=(int(w!=float(truth>0)) if k not in ['bma','stacking'] else None)) for k,w in weights.items()]
 predsec=time.perf_counter()-start
 ev=[dict(common,method=k,log_bf=v,error=v-ref['logbf'],within_point1_nat=abs(v-ref['logbf'])<=.1) for k,v in evidence.items()]
 raw=dict(common,z=z,muhat=muhat,peak=peak,log_bf=ref['logbf'],loo_score_difference=looscore,waic_score_difference=waic,reference_fit_seconds=fitsec,local_shape_seconds=localsec,loo_waic_stacking_seconds=cvsec,prediction_seconds=predsec,train_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),test_sha256=hashlib.sha256(y.astype('<f8').tobytes()).hexdigest())
 return raw,ev,preds
