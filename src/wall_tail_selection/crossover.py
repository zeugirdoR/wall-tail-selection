"""Boundary-Laplace candidates with fixed priors; truth enters evaluation only."""
import math,time,hashlib
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre,roots_hermitenorm,logsumexp,expit
from scipy.optimize import brentq
from . import singular as sg
SCORES=['eta_laplace','mu_laplace','exact','sbic','wbic','loo','waic','lrt','moment','bic','rlct_only','local_shape']
@lru_cache(None)
def grid(nodes,U):
 q,w=roots_legendre(nodes);return (q+1)*U/2,np.log(w/2)
def logratio(x,mu,family):
 if family=='mixture':return sg.logratio(x,mu)
 eta=np.asarray(mu)**2;return -.5*np.log1p(eta)[None,:]+.5*np.asarray(x)[:,None]**2*(eta/(1+eta))[None,:]
def derivatives(x,eta,family):
 n=len(x);S2=float(x@x)
 if family=='variance':return (S2-n*(1+eta))/(2*(1+eta)**2),(n*(1+eta)-2*S2)/(2*(1+eta)**3)
 if eta==0:return (S2-n)/2,-float(np.sum(x**4))/6
 if np.max(np.abs(x))*math.sqrt(eta)<.02:
  S4=float(np.sum(x**4));S6=float(np.sum(x**6));S8=float(np.sum(x**8))
  return (S2-n)/2-eta*S4/6+eta**2*S6/15-17*eta**3*S8/630,-S4/6+2*eta*S6/15-17*eta**2*S8/210
 z=math.sqrt(eta)*x;t=np.tanh(z)
 return float(np.sum(z*t)/(2*eta)-n/2),float(np.sum(z*z*(1-t*t)-z*t)/(4*eta**2))
def mle(x,U,family):
 if family=='mixture':return sg.mle(x,U)
 mu=math.sqrt(min(U*U,max(0.,float(x@x)/len(x)-1)));return mu,float(logratio(x,np.array([mu]),family).sum())
def sbic(peak,n):
 # r=L1'/L0 solves r²+(1-b)r-a=0 with a=e^peak*n^-1/4,b=e^peak*n^-1/2.
 la=peak-.25*math.log(n);lb=peak-.5*math.log(n)
 if abs(lb)<1e-12:return .5*la
 logabs=math.log(-math.expm1(lb)) if lb<0 else lb+math.log1p(-math.exp(-lb))
 ld=.5*np.logaddexp(2*logabs,math.log(4)+la)
 return float(math.log(2)+la-np.logaddexp(logabs,ld) if lb<0 else np.logaddexp(logabs,ld)-math.log(2))
def approximation(x,U,nodes,family,ref,coordinate):
 muhat,peak=mle(x,U,family);eta=muhat*muhat;s,h=derivatives(x,eta,family);mu,lw=grid(nodes,U)
 if coordinate=='eta':delta=mu*mu-eta
 else:delta=mu-muhat;s,h=2*muhat*s,2*s+4*eta*h
 if not math.isfinite(h) or h>=0 or not math.isfinite(s):return ref,1
 value=float(logsumexp(lw+peak+s*delta+.5*h*delta*delta))
 if not math.isfinite(value) or value>peak+1e-7:return ref,2
 return value,0
def stacking_weight(r):
 # Algebraically stable (exp(r)-1)/((1-w)+w exp(r)); no sigmoid-minus-w cancellation.
 r=np.asarray(r);positive=r>=0
 def derivative(w):
  v=np.empty_like(r);a=r[positive];b=r[~positive]
  v[positive]=-np.expm1(-a)/(w+(1-w)*np.exp(-a))
  v[~positive]=np.expm1(b)/((1-w)+w*np.exp(b))
  return float(v.sum())
 if derivative(0)<=0:return 0.
 if derivative(1)>=0:return 1.
 return float(brentq(derivative,0.,1.,xtol=1e-12))
def fit(x,cfg,family):
 start=time.perf_counter();U=cfg['prior_upper'];mu,lw=grid(cfg['nodes'],U);li=logratio(x,mu,family);ll=li.sum(axis=0);bf=float(logsumexp(lw+ll));lp=lw+ll-bf;refsec=time.perf_counter()-start
 start=time.perf_counter();eta,fe=approximation(x,U,cfg['nodes'],family,bf,'eta');ab,fm=approximation(x,U,cfg['nodes'],family,bf,'mu');appsec=time.perf_counter()-start
 n=len(x);_,peak=mle(x,U,family);r=dict(mu=mu,logpost=lp,li=li);loo,lc,wc=sg.predictive_scores(r);temp=lw+ll/math.log(n);temp-=logsumexp(temp);wbic=float(np.exp(temp)@ll)
 Z=float(np.sum(x*x-1)/math.sqrt(2*n));local=-.25*math.log(n)-math.log(U)+sg.local_integral(Z)
 scores=np.array([eta,ab,bf,sbic(peak,n),wbic,lc,wc,peak,Z,peak-.5*math.log(n),peak-.25*math.log(n),local]);assert np.isfinite(scores).all()
 return scores,r,np.array([fe,fm],int),np.array([refsec,appsec]),stacking_weight(loo)
@lru_cache(None)
def hermite(nodes):
 x,w=roots_hermitenorm(nodes);return x,w/math.sqrt(2*math.pi)
def evaluation_points(mu,family,nodes):
 x,w=hermite(nodes)
 return (x*math.sqrt(1+mu*mu),w) if family=='variance' else (np.r_[x-mu,x+mu],np.r_[w/2,w/2])
def risks(mu,family,ref,bf,stack,nodes):
 y,w=evaluation_points(mu,family,nodes);lr=logsumexp(logratio(y,ref['mu'],family)+ref['logpost'],axis=1);base=sg.logphi(y)
 return np.array([-w@base,-w@(base+lr),-w@(base+sg.mixture_ratio(lr,float(expit(bf)))),-w@(base+sg.mixture_ratio(lr,stack))])
def observations(cfg,ci,rep):
 cell=cfg['cells'][ci];n=cell['n'];v=cell['value'];mu=v if cell['regime']=='regular' else v*n**(-.25) if cell['regime']=='local' else v*(math.log(n)/n)**.25;family=cell['family'];rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],ci,rep]))
 x=sg.sample(rng,mu,n) if family=='mixture' else rng.normal(size=n)*math.sqrt(1+mu*mu)
 return x,mu,family
def task(args):
 cfg,ci,rep=args;x,mu,family=observations(cfg,ci,rep);scores,ref,fallback,timing,stack=fit(x,cfg,family);risk=risks(mu,family,ref,scores[2],stack,cfg['evaluation_nodes']) if cfg.get('evaluate',True) else np.full(4,np.nan)
 return ci,rep,scores,risk,fallback,timing,hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),stack
