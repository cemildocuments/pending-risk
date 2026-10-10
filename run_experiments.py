"""Run paired public-data replays and controlled validation experiments."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(v,'1')
from pathlib import Path
import argparse, concurrent.futures, json, time, hashlib
import numpy as np
import pandas as pd
from monitor import replay,delay_mechanism,boundaries
ROOT=Path(__file__).resolve().parent

def summarize(rows,tau,metadata):
 rows=[r for r in rows if r['n']>=512];out=[]
 for method in rows[0]['intervals']:
  lo=np.array([r['intervals'][method][0] for r in rows]);hi=np.array([r['intervals'][method][1] for r in rows]);truth=np.array([r['risk'] for r in rows]);ns=np.array([r['n'] for r in rows]);valid=np.isfinite(lo)&np.isfinite(hi)&(hi>=lo)
  covered=valid&(lo<=truth+1e-10)&(hi>=truth-1e-10);safe=valid&(hi<=tau);harm=valid&(lo>tau)
  first_true=np.flatnonzero(truth>tau);first_alarm=np.flatnonzero(harm)
  dd=(float(ns[first_alarm[0]]-ns[first_true[0]]) if len(first_true) and len(first_alarm) and first_alarm[0]>=first_true[0] else np.nan)
  out.append(dict(**metadata,method=method,any_miss=int(not covered.all()),pointwise_coverage=float(covered.mean()),mean_width=float(np.mean(np.where(valid,hi-lo,1.))),final_width=float(hi[-1]-lo[-1]) if valid[-1] else 1.,false_safe=int(np.any(safe&(truth>tau))),false_alarm=int(np.any(harm&(truth<=tau))),decided_fraction=float(np.mean((safe|harm)&valid)),correct_decision_fraction=float(np.mean(((safe&(truth<=tau))|(harm&(truth>tau)))&valid)),final_decided=int((safe[-1] or harm[-1]) and valid[-1]),final_risk=float(truth[-1]),threshold=float(tau),queries=rows[-1]['queried'],expedited=rows[-1]['expedited'],final_pending=rows[-1]['pending'],final_pending_weight=rows[-1]['pending_weight'],detection_delay=dd,n=int(ns[-1])))
 return out

def worker(job):
 f,policy,mode,seed,action=job;t0=time.perf_counter();d=np.load(f);loss=d['loss'];unc=d['uncertainty'];median=float(np.median(d['calibration_uncertainty']))
 q=np.full(len(loss),.2) if policy=='uniform' else np.where(unc>=median,.35,.05)
 delay=delay_mechanism(loss,mode,np.random.default_rng(100000+seed));rows=replay(loss,q,delay,seed=seed,accelerate=action)
 stem=Path(f).stem;parts=stem.split('_');meta=dict(dataset=parts[0],model=parts[1],training_seed=int(parts[2]),audit_seed=seed,policy=policy,delay=mode,action=action)
 if seed==101 and parts[2]=='11' and policy=='selective' and action=='none':
  path=ROOT/'results'/'paths';path.mkdir(parents=True,exist_ok=True);(path/f'{stem}_{mode}.json').write_text(json.dumps(rows))
 out=summarize(rows,min(.9,float(d['calibration_loss'].mean())+.05),meta)
 return out

def synthetic_worker(job):
 regime,mode,policy,seed,n=job;rng=np.random.default_rng(seed);x=rng.random(n);base=np.where(x>.5,.25,.05)
 if regime=='shift':base[np.arange(n)>=n//3]+=.25
 loss=(rng.random(n)<base).astype(float)
 q=np.full(n,.2) if policy=='uniform' else np.where(x>.5,.35,.05)
 delay=delay_mechanism(loss,mode,rng);rows=replay(loss,q,delay,seed=seed+200000)
 return summarize(rows,.2,dict(regime=regime,delay=mode,policy=policy,seed=seed))

def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=['pilot','public','synthetic','expedite'],default='pilot');p.add_argument('--workers',type=int,default=6);p.add_argument('--seeds',type=int,default=None);a=p.parse_args()
 outdir=ROOT/'results';outdir.mkdir(exist_ok=True);t0=time.perf_counter()
 if a.suite=='synthetic':
  count=a.seeds or 1000;jobs=[(r,m,p,800000+s,4096) for r in ['stable','shift'] for m in ['immediate','independent','error_slow','correct_slow','censored'] for p in ['uniform','selective'] for s in range(count)];func=synthetic_worker
 else:
  files=sorted((ROOT/'data'/'traces').glob('*.npz'));files=[f for f in files if f.stem.split('_')[1] not in ['logistic','boosting'] or f.stem.endswith('_11')];count=a.seeds or (3 if a.suite=='pilot' else 20)
  if a.suite=='pilot':files=[f for f in files if f.name.startswith('adult_boosting_11')]
  actions=['none'] if a.suite!='expedite' else ['random','oldest','weight'];modes=['immediate','independent','error_slow','correct_slow','censored'] if a.suite!='expedite' else ['error_slow','censored']
  jobs=[(str(f),p,m,101+s,act) for f in files for p in ['uniform','selective'] for m in modes for s in range(count) for act in actions];func=worker
 print('jobs',len(jobs),'workers',a.workers,flush=True);allrows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for i,rows in enumerate(ex.map(func,jobs,chunksize=4)):
   allrows.extend(rows)
   if (i+1)%100==0:print('completed',i+1,'seconds',round(time.perf_counter()-t0,1),flush=True)
 frame=pd.DataFrame(allrows);frame.to_csv(outdir/f'{a.suite}_runs.csv',index=False)
 (outdir/f'{a.suite}_timing.json').write_text(json.dumps(dict(jobs=len(jobs),workers=a.workers,elapsed_seconds=time.perf_counter()-t0,seed_count=count),indent=2))
 print(frame.groupby(['delay','method'])[['any_miss','mean_width','false_safe','false_alarm','correct_decision_fraction']].mean().round(4).to_string(),flush=True)
if __name__=='__main__':main()
