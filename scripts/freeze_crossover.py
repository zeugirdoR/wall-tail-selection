import json,math,hashlib,csv,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import norm
from wall_tail_selection.crossover import SCORES
from wall_tail_selection.crossover_inference import selections,risk_values,RISK_BASELINES,STRUCT_BASELINES
calp=Path(sys.argv[1]);pilot=Path('runs/local-crossover-pilot-v2');cm=json.loads((calp/'manifest.json').read_text());pm=json.loads((pilot/'manifest.json').read_text());assert cm['status']==pm['status']=='complete'
for p,m in [(calp,cm),(pilot,pm)]:
 for f,h in m['output_sha256'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
z=np.load(calp/'records.npz');pz=np.load(pilot/'records.npz');assert not set(z['train_hash'])&set(pz['train_hash']);thresholds={};rank=math.ceil((cm['config']['replicates']+1)*.95)
for ci,cell in enumerate(cm['config']['cells']):thresholds[cell['family']+':'+str(cell['n'])]=np.sort(z['scores'][z['cell']==ci],axis=0)[rank-1].tolist()
cfg=pm['config'];native,cs=selections(cfg,pz,thresholds);v=[];q=float(norm.ppf(1-.05/2000));g=risk_values(pz,native,'eta_laplace')
for ci,cell in enumerate(cfg['cells']):
 mask=pz['cell']==ci
 for b in RISK_BASELINES:
  d=(g-risk_values(pz,native,b))[mask];sd=float(d.std(ddof=1));v.append(dict(cell=ci,target='risk',baseline=b,sd=sd,required=(q*sd/.0005)**2))
 for b in STRUCT_BASELINES:
  ix=12 if b=='analytic_score' else SCORES.index(b);d=cs[mask,0].astype(int)-cs[mask,ix].astype(int);sd=float(d.std(ddof=1));v.append(dict(cell=ci,target='structural',baseline=b,sd=sd,required=(q*sd/.025)**2))
required=max(r['required'] for r in v);N=min(4096,max(1024,2**math.ceil(math.log2(max(1,required)))))
out=Path('runs/crossover-freeze-v1');out.mkdir(exist_ok=False)
f=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),confirmation_replicates=N,uncapped_required=required,capped=required>4096,calibration_order_rank=rank,thresholds=thresholds,pilot_manifest_sha256=hashlib.sha256((pilot/'manifest.json').read_bytes()).hexdigest(),calibration_manifest_sha256=hashlib.sha256((calp/'manifest.json').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path('CROSSOVER_PROTOCOL.md').read_bytes()).hexdigest(),calibration_job_id=cm['slurm_job_id'],confirmation_seed=2026102603)
(out/'frozen.json').write_text(json.dumps(f,indent=2)+'\n')
with (out/'size_requirements.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
cfg=dict(cfg,seed=2026102603,replicates=N,frozen_sha256=hashlib.sha256((out/'frozen.json').read_bytes()).hexdigest());(out/'confirmation.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps({k:v for k,v in f.items() if k!='thresholds'},indent=2))
