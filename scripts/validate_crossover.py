import json,hashlib,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete';cfg=m['config'];N=len(cfg['cells'])*cfg['replicates']
for name,h in m['output_sha256'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
z=np.load(p/'records.npz',allow_pickle=False);assert z['scores'].shape==(N,12) and z['risks'].shape==(N,4);assert np.isfinite(z['scores']).all();assert len(set(z['train_hash']))==N;assert np.array_equal(z['cell'],np.repeat(np.arange(len(cfg['cells'])),cfg['replicates']));assert np.array_equal(z['replicate'],np.tile(np.arange(cfg['replicates']),len(cfg['cells'])))
s=z['scores'];assert np.all(s[:,7]>=-1e-8);assert np.all(s[:,2]<=s[:,7]+1e-7);assert np.all(s[:,3]>=s[:,9]-1e-7) and np.all(s[:,3]<=s[:,10]+1e-7);assert np.all(s[:,4]<=s[:,7]+1e-7)
assert np.isin(z['fallback'],[0,1,2]).all()
for j in [0,1]:assert np.all(abs(s[z['fallback'][:,j]>0,j]-s[z['fallback'][:,j]>0,2])<1e-12)
if cfg.get('evaluate',True):assert np.isfinite(z['risks']).all()
else:assert np.isnan(z['risks']).all()
print(json.dumps(dict(status='validated',cases=N,seconds=m['elapsed_seconds'],fallback_counts=z['fallback'].astype(bool).sum(axis=0).tolist())))
