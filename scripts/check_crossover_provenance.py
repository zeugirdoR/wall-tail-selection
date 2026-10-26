import json,hashlib,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);q=Path('runs/local-crossover-platform-v1');f=Path('runs/crossover-freeze-v1/frozen.json');m=json.loads((p/'manifest.json').read_text());qm=json.loads((q/'manifest.json').read_text());frozen=json.loads(f.read_text());z=np.load(p/'records.npz');local=np.load(q/'records.npz');assert m['config']['frozen_sha256']==hashlib.sha256(f.read_bytes()).hexdigest();assert frozen['frozen_utc']<m['started_utc'];assert m['source_sha256']==qm['source_sha256'];ix=local['cell']*m['config']['replicates']+local['replicate'];assert np.array_equal(z['train_hash'][ix],local['train_hash']);assert np.array_equal(z['fallback'][ix],local['fallback']);maxdiff={}
for k in ['scores','risks','stack_weight']:
 maxdiff[k]=float(np.max(abs(z[k][ix]-local[k])));assert maxdiff[k]<1e-7
assert np.array_equal(z['scores'][ix]>0,local['scores']>0)
for path in ['runs/deltaai-crosscal-3462202','runs/local-crossover-pilot-v2']:
 other=np.load(Path(path)/'records.npz');assert not set(z['train_hash'])&set(other['train_hash'])
result=dict(frozen_before_confirmation=True,frozen_hash_verified=True,source_hashes_equal=True,calibration_pilot_confirmation_hashes_disjoint=True,local_platform_cases=len(ix),native_choices_equal=True,max_absolute_differences=maxdiff)
(p/'provenance_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
