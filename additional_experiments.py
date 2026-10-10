"""Adaptive-policy, backlog-identity, and computational validation."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(v,'1')
from pathlib import Path
import numpy as np,pandas as pd,concurrent.futures,time,resource,json,argparse
from monitor import replay,delay_mechanism,boundaries
from run_experiments import summarize
ROOT=Path(__file__).resolve().parent

def adaptive(job):
 kind,seed=job;n=4096;rng=np.random.default_rng(seed)
 if kind=='binary':loss=(rng.random(n)<np.where(np.arange(n)<n//2,.1,.35)).astype(float)
 else:loss=np.where(np.arange(n)<n//2,rng.beta(1,7,n),rng.beta(3,5,n))
 delay=delay_mechanism(loss,'error_slow',rng);u=np.random.default_rng(seed+500000).random(n)
 due=[[] for _ in range(n+1)];s=0.;cnt=0;q=np.empty(n)
 for i in range(n):
  for j in due[i]:s+=loss[j];cnt+=1
  q[i]=.4 if cnt and s/cnt>.12 else .1
  if u[i]<q[i]:
   at=max(i+1,i+int(delay[i]))
   if at<n:due[at].append(i)
 rows=replay(loss,q,delay,seed=seed+500000)
 return summarize(rows,.2,dict(loss_type=kind,seed=seed,mean_propensity=float(q.mean())))

def runtime(n):
 rng=np.random.default_rng(8);l=(rng.random(n)<.2).astype(float);q=np.where(rng.random(n)<.5,.05,.35);d=delay_mechanism(l,'error_slow',rng)
 start=time.perf_counter();rows=replay(l,q,d,step=max(64,n//128));elapsed=time.perf_counter()-start
 return dict(n=n,checkpoints=len(rows),elapsed_seconds=elapsed,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args();out=ROOT/'results';t=time.perf_counter()
 jobs=[(kind,900000+s) for kind in ['binary','continuous'] for s in range(1000)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:r=list(ex.map(adaptive,jobs,chunksize=4))
 pd.DataFrame([v for rr in r for v in rr]).to_csv(out/'adaptive_runs.csv',index=False)
 n=8192;rng=np.random.default_rng(109);loss=(rng.random(n)<.25).astype(float);d=delay_mechanism(loss,'censored',rng);h=np.arange(1,n+1)+d>n;rows=[]
 for q0 in [.05,.1,.2,.4,.8]:
  qr=np.full(n,q0);b=float(boundaries(qr,[n])[0])/n
  for s in range(2000):
   queried=np.random.default_rng(2000000+s).random(n)<q0;w=float(np.sum(queried*h)/q0/n)
   rows.append(dict(propensity=q0,seed=s,pending_width=w,population_pending=float(h.mean()),statistical_width=2*b,unclipped_width=w+2*b))
 pd.DataFrame(rows).to_csv(out/'backlog_runs.csv',index=False)
 rr=[]
 for n in [2000,20000,100000]:
  with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:rr.append(ex.submit(runtime,n).result())
 pd.DataFrame(rr).to_csv(out/'runtime.csv',index=False)
 (out/'additional_timing.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-t,'workers':a.workers,'adaptive_paths':len(jobs),'backlog_paths':len(rows)},indent=2))
 print(pd.DataFrame([v for rr in r for v in rr]).groupby(['loss_type','method'])[['any_miss','mean_width']].mean().to_string());print(rr)
if __name__=='__main__':main()
