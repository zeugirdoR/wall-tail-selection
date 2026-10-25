import csv,json,hashlib,math,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());assert m['status']=='complete';c=m['config'];N=len(c['cells'])*c['replicates']
for n,h in m['output_sha256'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
rows={name:list(csv.DictReader((p/name).open())) for name in ['datasets.csv','evidence.csv','predictions.csv']};assert [len(rows[k]) for k in rows]==[N,N*5,N*11]
keys=set();ev={};train=set();test=set()
for r in rows['datasets.csv']:
 key=(r['cell'],r['replicate']);assert key not in keys;keys.add(key);train.add(r['train_sha256']);test.add(r['test_sha256']);assert math.isfinite(float(r['log_bf']));assert 0<=float(r['muhat'])<=c['prior_upper'];assert float(r['peak'])>=-1e-8;assert float(r['log_bf'])<=float(r['peak'])+1e-8
assert len(train)==len(test)==N and not train&test
for r in rows['evidence.csv']:
 assert all(math.isfinite(float(r[k])) for k in ['log_bf','error']);ev[(r['cell'],r['replicate'],r['method'])]=float(r['log_bf']);assert r['method']!='exact' or float(r['error'])==0
for r in rows['predictions.csv']:
 assert math.isfinite(float(r['log_loss'])) and math.isfinite(float(r['excess_vs_oracle']));w=float(r['weight_m1']);assert 0<=w<=1
 key=(r['cell'],r['replicate'],r['method'])
 if key in ev:assert w==float(ev[key]>0)
 if r['method']=='always_m0':assert w==0
 if r['method']=='always_m1':assert w==1
print(json.dumps(dict(status='validated',cases=N,source=m['source_sha256']['src/wall_tail_selection/singular.py'],elapsed_seconds=m['elapsed_seconds'])))
