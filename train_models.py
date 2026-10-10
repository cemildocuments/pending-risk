"""Fit independent CPU/GPU predictors and archive fixed prediction traces."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(v,'1')
from pathlib import Path
import argparse, concurrent.futures, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
ROOT=Path(__file__).resolve().parent

def train(job):
 name,kind,seed,device=job;t0=time.perf_counter();d=np.load(ROOT/'data'/'processed'/f'{name}.npz')
 x,y=d['x_train'],d['y_train'];xc,yc=d['x_cal'],d['y_cal'];xt,yt=d['x_test'],d['y_test']; nc=int(max(y.max(),yc.max(),yt.max())+1)
 if kind=='mlp':
  import torch
  torch.set_num_threads(2);torch.manual_seed(seed)
  use='mps' if device=='auto' and torch.backends.mps.is_available() else device if device!='auto' else 'cpu'
  model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,nc)).to(use)
  opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  tx=torch.tensor(x,device=use);ty=torch.tensor(y,dtype=torch.long,device=use)
  rng=np.random.default_rng(seed)
  for ep in range(35):
   idx=rng.permutation(len(x));model.train()
   for start in range(0,len(x),1024):
    ids=torch.tensor(idx[start:start+1024],device=use);opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(tx[ids]),ty[ids]);loss.backward();opt.step()
  model.eval()
  def predict(z):
   with torch.no_grad():return torch.softmax(model(torch.tensor(z,device=use)),dim=1).cpu().numpy().astype(np.float64)
  pc=predict(xc);pt=predict(xt);backend=use
 else:
  if kind=='logistic': model=LogisticRegression(max_iter=2000,C=1.,random_state=seed)
  elif kind=='forest':model=RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_features='sqrt',n_jobs=1,random_state=seed)
  else:model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=seed)
  model.fit(x,y);pc=model.predict_proba(xc);pt=model.predict_proba(xt);backend='cpu'
 preds=pt.argmax(1);losses=(preds!=yt).astype(np.float64);cal_loss=(pc.argmax(1)!=yc).astype(float)
 out=ROOT/'data'/'traces';out.mkdir(exist_ok=True)
 np.savez_compressed(out/f'{name}_{kind}_{seed}.npz',loss=losses,uncertainty=1-pt.max(1),probabilities=pt,labels=yt,predictions=preds,source_index=d['source_index'],calibration_loss=cal_loss,calibration_uncertainty=1-pc.max(1))
 r=dict(dataset=name,model=kind,seed=seed,backend=backend,n_test=len(yt),calibration_error=float(cal_loss.mean()),test_error=float(losses.mean()),test_logloss=float(log_loss(yt,pt,labels=list(range(nc)))),elapsed_seconds=time.perf_counter()-t0)
 (out/f'{name}_{kind}_{seed}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cpu','gpu'],default='cpu');p.add_argument('--workers',type=int,default=6);p.add_argument('--device',default='auto');a=p.parse_args()
 jobs=[(n,m,s,a.device) for n in ['adult','bank','electricity','gas'] for m in (['mlp'] if a.mode=='gpu' else ['logistic','forest','boosting']) for s in ([11] if m in ['logistic','boosting'] else [11,29,47])]
 if a.mode=='gpu':
  for j in jobs:train(j)
 else:
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:list(ex.map(train,jobs))
