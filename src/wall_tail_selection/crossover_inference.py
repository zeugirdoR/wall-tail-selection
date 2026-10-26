"""Frozen comparisons and confidence intervals; no method fitting."""
import math
import numpy as np
from scipy.stats import beta,t,chi2
from .crossover import SCORES
RISK_BASELINES=['mu_laplace','exact','loo','waic','sbic','wbic','bma','stacking']
STRUCT_BASELINES=['mu_laplace','exact','loo','waic','sbic','wbic','lrt','moment','analytic_score']
ALPHA=.05;MULTIPLICITY=1000

def selections(cfg,z,thresholds):
 native=z['scores']>0;cal=np.zeros((len(native),13),bool)
 for ci,cell in enumerate(cfg['cells']):
  mask=z['cell']==ci;key=cell['family']+':'+str(cell['n']);cal[mask,:12]=z['scores'][mask]>np.array(thresholds[key]);cal[mask,12]=z['scores'][mask,8]>(chi2.ppf(.95,cell['n'])-cell['n'])/math.sqrt(2*cell['n'])
 return native,cal

def risk_values(z,native,name):
 if name=='bma':return z['risks'][:,2]
 if name=='stacking':return z['risks'][:,3]
 return z['risks'][:,0]+native[:,SCORES.index(name)]*(z['risks'][:,1]-z['risks'][:,0])
def cp(k,n,tail=ALPHA/(2*MULTIPLICITY)):
 return (0. if k==0 else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.ppf(1-tail,k+1,n-k)))
def paired_binary(d):
 n=len(d);plus=int(np.sum(d==1));minus=int(np.sum(d==-1));a,b=cp(plus,n,ALPHA/(4*MULTIPLICITY));c,e=cp(minus,n,ALPHA/(4*MULTIPLICITY));return float(np.mean(d)),a-e,b-c

def paired_risk(d):
 n=len(d);mean=float(np.mean(d));se=float(np.std(d,ddof=1)/math.sqrt(n));radius=float(t.ppf(1-ALPHA/(2*MULTIPLICITY),n-1))*se;return mean,se,mean-radius,mean+radius
