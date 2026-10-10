import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linprog
from monitor import cgf,joint_interval,replay

def run_tests():
 rng=np.random.default_rng(9831)
 # Independent LP oracle verifies the analytic interval on arbitrary observed states.
 for _ in range(1000):
  n=12;q=rng.choice([.05,.2,.5,1.],n);a=rng.random(n)<q;l=rng.random(n);obs=a&(rng.random(n)<.5)
  c=a/q-1;u=np.flatnonzero(~obs);levels=np.unique(q);pending=a&~obs
  pc=np.array([np.sum(pending&(q==v)) for v in levels]);bound=rng.uniform(0,8)
  actual=joint_interval(n,l[obs].sum(),np.dot(c[obs],l[obs]),c.sum(),pc,levels,np.sum(~a),bound)
  s0=np.dot(c[obs],l[obs])-.5*c.sum();mat=np.stack([c[u],-c[u]]);rhs=np.array([bound-s0,bound+s0])
  lo=linprog(np.ones(len(u)),A_ub=mat,b_ub=rhs,bounds=(0,1),method='highs') if len(u) else None
  hi=linprog(-np.ones(len(u)),A_ub=mat,b_ub=rhs,bounds=(0,1),method='highs') if len(u) else None
  if lo is not None and lo.success:
   expected=((l[obs].sum()+lo.fun)/n,(l[obs].sum()-hi.fun)/n)
   assert np.allclose(actual,expected,atol=1e-8),(actual,expected)
  elif lo is not None:assert actual==(np.inf,-np.inf)
 # Verify the conditional MGF majorant on a dense loss/propensity/lambda grid.
 for q in [.02,.05,.2,.8,1.]:
  for lam in [-3,-.5,-.01,.01,.5,3]:
   r=np.linspace(-.5,.5,501)
   mg=(1-q)*np.exp(-lam*r)+q*np.exp(lam*(1/q-1)*r)
   assert np.max(np.log(mg))<=cgf(q,abs(lam))+1e-10
 # Full auditing and immediate feedback identifies exact losses.
 l=rng.integers(0,2,1024).astype(float)
 for row in replay(l,np.ones(1024),np.zeros(1024),step=64):
  assert np.allclose(row['intervals']['PACE'],[row['risk'],row['risk']])
 # Joint inversion is contained in the propensity-specific completion envelope.
 for row in replay(l,np.tile([.05,.35],512),np.where(l,500,20),step=64):
  a,b=row['intervals']['PACE'];x,y=row['intervals']['Weighted envelope']
  assert a>=x-1e-9 and b<=y+1e-9,(a,b,x,y)
 print('PASS: 1,000 independent LP comparisons, MGF grid, complete-audit identity, and envelope containment.')
if __name__=='__main__':run_tests()
