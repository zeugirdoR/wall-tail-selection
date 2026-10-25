"""Continuous constrained likelihood/evidence; no likelihood floors."""
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import logsumexp, roots_legendre
from .geometry import separated_grid_count


def model_name(a): return 'exponential' if a is None else f'endpoint_a{a:g}'
def cdf(y,t,a):
    y=np.asarray(y)
    return -np.expm1(-np.maximum(y,0)/t) if a is None else -np.expm1(a*np.log(np.maximum(1-np.clip(y/t,0,1),np.finfo(float).tiny)))
def outside(t,a):
    if a is None: return math.exp(-t)
    return max(1-t,0.)**a


def fit(x,a,bounds,ys=None,rtol=1e-8):
    x=np.asarray(x,dtype=float);L,U=bounds;n=len(x);mx=float(x.max());S=float(x.sum());D=math.log(U/L)
    if not np.isfinite(x).all() or (x<0).any(): raise ValueError('invalid observations')
    if a is not None and mx>=U:
        return dict(status='impossible',scale=None,max_loglik=None,log_evidence=None,occam=None,boundary='no_support',cdf=None,relative_error=None)
    def ll(t):
        if a is None: return -n*math.log(t)-S/t
        if a==1: return -n*math.log(t) if mx<=t else -math.inf
        if mx>=t: return -math.inf
        return n*math.log(a)-n*math.log(t)+(a-1)*float(np.log1p(-x/t).sum())
    if a is None: t=min(U,max(L,S/n));boundary='lower' if t==L else 'upper' if t==U else 'interior'
    elif a==1: t=max(L,mx);boundary='lower' if t==L else 'sample_max'
    else:
        def derivative(t): return -n+(a-1)*float((x/(t-x)).sum())
        if L>mx and derivative(L)<=0: t=L;boundary='lower'
        elif derivative(U)>=0: t=U;boundary='upper'
        else:
            t=brentq(derivative,max(L,np.nextafter(mx,math.inf)),U,xtol=1e-13,rtol=1e-13);boundary='interior'
    lm=ll(t);lo=math.log(L if a is None else max(L,mx));hi=math.log(U);peak=math.log(t)
    # Geometric panels around the peak resolve even boundary-dominated likelihoods.
    points=[peak];step=1/n
    while step<hi-lo:
        points.extend([peak-step,peak+step]);step*=4
    points=sorted(set(v for v in points if lo<v<hi))
    def integrand(z):
        scale=math.exp(z);v=ll(scale)-lm
        if v>1e-7:raise ArithmeticError('likelihood exceeds constrained maximum')
        w=math.exp(v)
        return np.array([w]) if ys is None else np.r_[w,w*cdf(ys,scale,a)]
    value,error,info=quad_vec(integrand,lo,hi,epsabs=1e-13,epsrel=rtol,norm='max',points=points,full_output=True,limit=1000)
    if not info.success or not value[0]>0 or not np.isfinite(value).all():raise ArithmeticError('evidence quadrature failed: '+str(info.message))
    logz=lm+math.log(value[0])-math.log(D)
    return dict(status='finite',scale=t,max_loglik=lm,log_evidence=logz,occam=lm-logz,boundary=boundary,cdf=None if ys is None else value[1:]/value[0],relative_error=float(error/value[0]))


def task(args):
    config,ti,n,rep,bi=args;shapes=config['shapes'];truth=shapes[ti];bounds=config['domains'][bi]
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],ti,rep]));u=rng.random(max(config['sample_sizes']))[:n]
    x=-np.log1p(-u) if truth is None else -np.expm1(np.log1p(-u)/truth)
    nodes,weights=roots_legendre(config['evaluation_nodes']);ys=(nodes+1)*config['evaluation_upper']/2;weights=weights/2
    truthcdf=cdf(ys,1.,truth);fits=[fit(x,a,bounds,ys,config['integration_rtol']) for a in shapes]
    lm=np.array([-np.inf if f['status']=='impossible' else f['max_loglik'] for f in fits]);lz=np.array([-np.inf if f['status']=='impossible' else f['log_evidence'] for f in fits])
    penalties=np.array([math.log(separated_grid_count(a,n,*bounds,config['resolution_c'])) for a in shapes])
    common=dict(domain=bi,lower=bounds[0],upper=bounds[1],truth=model_name(truth),n=n,replicate=rep)
    costs=[]
    for j,(a,f) in enumerate(zip(shapes,fits)):
        costs.append(dict(common,model=model_name(a),status=f['status'],scale=f['scale'],boundary=f['boundary'],max_loglik=f['max_loglik'],log_evidence=f['log_evidence'],occam=f['occam'],logN=penalties[j],residual=None if f['occam'] is None else penalties[j]-f['occam'],relative_error=f['relative_error']))
    predictive=np.array([np.zeros_like(ys) if f['status']=='impossible' else f['cdf'] for f in fits])
    rows=[];choices={'aic':int(np.argmax(lm-1)),'bic':int(np.argmax(lm-.5*math.log(n))),'geometry':int(np.argmax(lm-penalties)),'bayes_continuous':int(np.argmax(lz))}
    for method,j in choices.items():
        f=fits[j];a=shapes[j];assert f['status']=='finite'
        for pred in ['plugin','within_model_bayes']:
            pc=cdf(ys,f['scale'],a) if pred=='plugin' else f['cdf']
            mass=0. if a is None else outside(f['scale'] if pred=='plugin' else bounds[1],truth)
            rows.append(dict(common,method=method,predictive=pred,selected_model=model_name(a),cdf_risk=float(np.dot(weights,(pc-truthcdf)**2)),false_support_mass=mass,infinite_population_log_loss=mass>0))
    w=np.exp(lz-logsumexp(lz));assert (w[~np.isfinite(lz)]==0).all()
    rows.append(dict(common,method='bayes_model_average',predictive='within_model_bayes',selected_model='mixture',cdf_risk=float(np.dot(weights,(w@predictive-truthcdf)**2)),false_support_mass=0.,infinite_population_log_loss=False))
    pairs=[]
    for j in range(len(shapes)):
        for k in range(j+1,len(shapes)):
            finite=fits[j]['status']==fits[k]['status']=='finite'
            pairs.append(dict(common,model1=model_name(shapes[j]),model2=model_name(shapes[k]),status='finite' if finite else 'undefined_impossible',delta_logN=float(penalties[j]-penalties[k]),delta_occam=fits[j]['occam']-fits[k]['occam'] if finite else None,residual=(penalties[j]-penalties[k])-(fits[j]['occam']-fits[k]['occam']) if finite else None))
    import hashlib
    return costs,rows,pairs,hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()
