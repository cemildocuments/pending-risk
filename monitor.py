"""Finite-prefix confidence sets under randomized audits and arbitrary label delay.

All inference uses float64. A request must be randomized independently of its
unobserved loss, conditional on the pre-request history and recorded propensity.
Delay and post-request acceleration may depend arbitrarily on outcomes.
"""
import numpy as np

LAMBDAS=np.geomspace(2**-12,4.,40)

def cgf(q,lam):
 """Worst-case centered Bernoulli audit log-MGF over losses in [0,1]."""
 q=np.asarray(q,dtype=np.float64)
 a=np.log1p(-np.minimum(q,1-1e-16));b=np.log(q)
 plus=np.logaddexp(a-lam/2,b+lam*(1/q-1)/2)
 minus=np.logaddexp(a+lam/2,b-lam*(1/q-1)/2)
 return np.where(q==1,0.,np.maximum(plus,minus))

def boundaries(q,checkpoints,alpha=.05):
 q=np.asarray(q,dtype=np.float64);checks=np.asarray(checkpoints,dtype=int)
 levels,group=np.unique(q,return_inverse=True)
 counts=np.stack([np.cumsum(group==g)[checks-1] for g in range(len(levels))],axis=1)
 psi=np.stack([cgf(levels,lam) for lam in LAMBDAS],axis=1)
 return np.min((np.log(2*len(LAMBDAS)/alpha)+counts@psi)/LAMBDAS,axis=1)

def _mass_for_gain(need,weights,counts):
 """Fractional knapsack: smallest outcome mass yielding a required score gain."""
 if need<=1e-10:return 0.
 mass=0.
 for w,c in zip(weights,counts):
  if w<=0:continue
  take=min(float(c),need/w);mass+=take;need-=take*w
  if need<=1e-8:return mass
 return np.inf

def joint_interval(n,known_sum,known_c_loss,sum_c,pending_counts,q_levels,n_unaudited,bound):
 """Exact range of mean bounded loss compatible with a linear audit certificate.

Unknown losses are allowed in [0,1]; hence this also covers binary losses.
Returns an empty set (inf,-inf) if the certificate constraints are infeasible.
"""
 cs=1/np.asarray(q_levels)-1
 order=np.argsort(-cs);weights=cs[order];counts=np.asarray(pending_counts)[order]
 n_pending=float(np.sum(counts));s0=known_c_loss-.5*sum_c
 min_s=s0-n_unaudited;max_s=s0+float(np.dot(weights,counts))
 if min_s>bound+1e-8 or max_s < -bound-1e-8:return np.inf,-np.inf
 if s0 < -bound:
  lo_extra=_mass_for_gain(-bound-s0,weights,counts)
 elif s0>bound:lo_extra=s0-bound
 else:lo_extra=0.
 s1=s0-n_unaudited+float(np.dot(weights,counts))
 if s1>bound:removed=_mass_for_gain(s1-bound,weights,counts)
 elif s1 < -bound:removed=-bound-s1
 else:removed=0.
 return max(0.,(known_sum+lo_extra)/n),min(1.,(known_sum+n_unaudited+n_pending-removed)/n)

def replay(loss,q,delay,seed=0,alpha=.05,step=64,accelerate='none',budget_fraction=.02,save_path=False,boundary_curve=None):
 loss=np.asarray(loss,dtype=np.float64);q=np.asarray(q,dtype=np.float64);delay=np.asarray(delay);N=len(loss)
 rng=np.random.default_rng(seed);requested=rng.random(N)<q
 issued=np.arange(1,N+1);arrival=np.where(requested,issued+delay,np.inf)
 levels=np.unique(q);groups=np.searchsorted(levels,q)
 checkpoints=np.unique(np.r_[np.arange(step,N+1,step),N]);allbs=boundaries(q,np.arange(1,N+1),alpha) if boundary_curve is None else boundary_curve;bs=allbs[checkpoints-1]
 cumloss=np.cumsum(loss);c=np.where(requested,1/q,0)-1;cumc=np.cumsum(c)
 full=np.cumsum(.5+np.where(requested,(loss-.5)/q,0)); all_audited_loss=np.cumsum(requested*loss)
 # Each completion is consumed once. Actual labels are only read for completed indices.
 completed=np.zeros(N,dtype=bool);used=0;max_expedite=int(budget_fraction*N)
 records=[];last=0
 for j,n in enumerate(checkpoints):
  completed[:n] |= arrival[:n]<=n
  pending=np.flatnonzero(requested[:n]&~completed[:n])
  if accelerate!='none' and n%256==0 and used<max_expedite and len(pending):
   k=min(4,max_expedite-used,len(pending))
   if accelerate=='weight': chosen=pending[np.argsort(q[pending],kind='stable')[:k]]
   elif accelerate=='oldest':chosen=pending[:k]
   elif accelerate=='random':chosen=rng.choice(pending,size=k,replace=False)
   else:raise ValueError(accelerate)
   completed[chosen]=True;used+=k;pending=np.flatnonzero(requested[:n]&~completed[:n])
  observed=np.flatnonzero(completed[:n]);r=loss[observed];qo=q[observed]
  known=float(r.sum());center=n*.5+float(np.sum((r-.5)/qo));w=float(np.sum(1/q[pending]));b=float(bs[j]);p=len(pending)
  levels_count=np.bincount(groups[pending],minlength=len(levels));nu=int(n-requested[:n].sum())
  joint=joint_interval(n,known,float(np.dot(c[observed],r)),float(cumc[n-1]),levels_count,levels,nu,b)
  env=(max(0.,(center-w/2-b)/n),min(1.,(center+w/2+b)/n))
  coarse=(max(0.,(center-p/(2*q[:n].min())-b)/n),min(1.,(center+p/(2*q[:n].min())+b)/n))
  # Symmetric absolute-residual correction used as a deliberately conservative comparator.
  coarse_abs=(max(0.,(center-p/q[:n].min()-b)/n),min(1.,(center+p/q[:n].min()+b)/n))
  ignore=(max(0.,(center-b)/n),min(1.,(center+b)/n))
  # Arrived-only estimation is diagnostic: its target does not generally equal prefix risk.
  if len(observed):
   mean=float(r.mean());rad=np.sqrt(np.log(2*len(checkpoints)/alpha)/(2*len(observed)));arrived=(max(0.,mean-rad),min(1.,mean+rad))
  else:arrived=(0.,1.)
  oracle=(max(all_audited_loss[n-1]/n,(full[n-1]-b)/n,0.),min((all_audited_loss[n-1]+nu)/n,(full[n-1]+b)/n,1.))
  # Delay-independent honest baseline: last issuance prefix with no unresolved audit.
  mature=int(pending[0]) if p else n
  if mature:
   mb=allbs[mature-1]
   il=max(0.,(full[mature-1]-mb)/n);iu=min(1.,(full[mature-1]+mb+n-mature)/n)
  else:il,iu=0.,1.
  intervals={'PACE':joint,'Weighted envelope':env,'Coarse centered':coarse,'Coarse absolute':coarse_abs,'Ignore pending':ignore,'Arrived only':arrived,'Mature prefix':(il,iu),'Full-feedback oracle':oracle}
  true=float(cumloss[n-1]/n)
  records.append(dict(n=int(n),risk=true,queried=int(requested[:n].sum()),returned=len(observed),pending=p,pending_weight=w/n,expedited=used,intervals=intervals))
 return records

def delay_mechanism(loss,mode,rng,scale=1.):
 loss=np.asarray(loss);n=len(loss)
 if mode=='immediate':return np.zeros(n)
 if mode=='independent':mean=np.full(n,256.*scale)
 elif mode in ['error_slow','censored']:mean=np.where(loss>.5,2048.,64.)*scale
 elif mode=='correct_slow':mean=np.where(loss>.5,64.,2048.)*scale
 else:raise ValueError(mode)
 d=rng.geometric(1/(mean+1)).astype(float)-1
 if mode=='censored':d[rng.random(n)<np.where(loss>.5,.25,.02)]=np.inf
 return d
