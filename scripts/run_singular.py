import argparse,csv,time
from concurrent.futures import ProcessPoolExecutor
from wall_tail_selection.provenance import start_run,finish_run,save_manifest
from wall_tail_selection.singular import task
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();cfg,out,m=start_run(a.config,a.output);start=time.monotonic();m.update(workers=a.workers,execution='CPU scipy quadrature, no GPU acceleration')
args=[(cfg,ci,rep) for ci in range(len(cfg['cells'])) for rep in range(cfg['replicates'])];files=[];writers=[None]*3
try:
 for name in ['datasets.csv','evidence.csv','predictions.csv']:files.append((out/name).open('w',newline=''))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,(raw,ev,pred) in enumerate(pool.map(task,args)):
   for j,rows in enumerate([[raw],ev,pred]):
    if writers[j] is None:writers[j]=csv.DictWriter(files[j],fieldnames=list(rows[0]));writers[j].writeheader()
    writers[j].writerows(rows)
   if (i+1)%16==0:print(f'{i+1}/{len(args)}',flush=True)
 for f in files:f.close()
 m.update(dataset_cases=len(args),elapsed_seconds=time.monotonic()-start);finish_run(out,m);print('COMPLETE',m['elapsed_seconds'],flush=True)
except Exception as e:
 for f in files:f.close()
 m.update(status='failed',error=str(e),error_type=type(e).__name__);save_manifest(out,m);raise
