import argparse,time,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from wall_tail_selection.provenance import start_run,finish_run,save_manifest
from wall_tail_selection.crossover import task,SCORES
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();cfg,out,m=start_run(a.config,a.output);start=time.monotonic();m.update(workers=a.workers,execution='CPU quadrature',classification='Prespecified bounded suite; held-out confirmation only for frozen seed 2026102603',score_columns=SCORES,risk_columns=['m0','m1','bma','stacking'])
args=[(cfg,ci,rep) for ci in range(len(cfg['cells'])) for rep in range(cfg['replicates'])];records=[]
try:
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,r in enumerate(pool.map(task,args,chunksize=16)):
   records.append(r)
   if (i+1)%1024==0:print(f'{i+1}/{len(args)}',flush=True)
 names=['cell','replicate','scores','risks','fallback','timing','train_hash','stack_weight'];np.savez_compressed(out/'records.npz',**{name:np.asarray([r[j] for r in records]) for j,name in enumerate(names)})
 m.update(dataset_cases=len(records),elapsed_seconds=time.monotonic()-start);finish_run(out,m);print('COMPLETE',m['elapsed_seconds'],flush=True)
except Exception as e:m.update(status='failed',error=str(e));save_manifest(out,m);raise
