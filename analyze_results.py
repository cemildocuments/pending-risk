"""Aggregate measured results with paired seed-level Monte Carlo intervals."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent;O=ROOT/'results';A=O/'analysis';A.mkdir(exist_ok=True)

def paired_ci(d,value,method1,method0,relative=False):
 p=d.pivot_table(index='audit_seed',columns='method',values=value,aggfunc='mean')
 x=p[method1].to_numpy();y=p[method0].to_numpy();rng=np.random.default_rng(309);ix=rng.integers(0,len(x),(10000,len(x)))
 z=(1-x[ix].mean(1)/y[ix].mean(1)) if relative else (x[ix]-y[ix]).mean(1)
 estimate=1-x.mean()/y.mean() if relative else (x-y).mean()
 return {'estimate':float(estimate),'low':float(np.quantile(z,.025)),'high':float(np.quantile(z,.975))}

def main():
 p=pd.read_csv(O/'public_runs.csv');s=pd.read_csv(O/'synthetic_runs.csv');e=pd.read_csv(O/'expedite_runs.csv')
 metrics=['any_miss','mean_width','final_width','false_safe','false_alarm','correct_decision_fraction','final_decided','queries','expedited','final_pending_weight']
 p.groupby(['delay','method'])[metrics].mean().reset_index().to_csv(A/'public_summary.csv',index=False)
 p.groupby(['dataset','policy','delay','method'])[metrics].mean().reset_index().to_csv(A/'public_strata.csv',index=False)
 e.groupby(['policy','delay','action','method'])[metrics].mean().reset_index().to_csv(A/'expedite_summary.csv',index=False)
 sy=s.groupby(['regime','policy','delay','method']).agg(misses=('any_miss','sum'),repetitions=('any_miss','size'),width=('mean_width','mean'),false_safe=('false_safe','mean'),false_alarm=('false_alarm','mean')).reset_index()
 sy['noncoverage']=sy.misses/sy.repetitions
 sy['ci_low']=[0 if x==0 else beta.ppf(.025,x,n-x+1) for x,n in zip(sy.misses,sy.repetitions)]
 sy['ci_high']=[1 if x==n else beta.ppf(.975,x+1,n-x) for x,n in zip(sy.misses,sy.repetitions)]
 sy.to_csv(A/'synthetic_cells.csv',index=False)
 claims={}
 for delay in ['independent','error_slow','correct_slow','censored']:
  sub=p[p.delay==delay]
  for m in ['Coarse absolute','Coarse centered','Weighted envelope']:
   claims[f'width_reduction_{delay}_{m}']=paired_ci(sub,'mean_width','PACE',m,True)
  sub=p[(p.delay==delay)&(p.policy=='selective')]
  claims[f'selective_width_reduction_{delay}']=paired_ci(sub,'mean_width','PACE','Coarse centered',True)
 for delay in ['error_slow','censored']:
  z=e[(e.delay==delay)&(e.policy=='selective')&(e.method=='PACE')].copy();z['method']=z.action
  for method in ['random','oldest']:
   claims[f'priority_width_reduction_{delay}_{method}']=paired_ci(z,'mean_width','weight',method,True)
   claims[f'priority_decision_gain_{delay}_{method}']=paired_ci(z,'correct_decision_fraction','weight',method)
  z0=p[(p.delay==delay)&(p.policy=='selective')&(p.method=='PACE')].copy();z0['method']='none';z=pd.concat([z,z0])
  claims[f'priority_width_reduction_{delay}_none']=paired_ci(z,'mean_width','weight','none',True)
  claims[f'priority_decision_gain_{delay}_none']=paired_ci(z,'correct_decision_fraction','weight','none')
 (A/'contrasts.json').write_text(json.dumps(claims,indent=2))
 print(json.dumps(claims,indent=2));print('Synthetic PACE maximum cell:',sy[sy.method=='PACE'].sort_values('noncoverage',ascending=False).head().to_string(index=False))
 print(p.groupby(['delay','method'])[metrics[:6]+['correct_decision_fraction']].mean().round(5).to_string())
if __name__=='__main__':main()
