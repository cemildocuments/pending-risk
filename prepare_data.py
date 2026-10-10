"""Download, verify and transform public benchmarks into fixed evaluation splits."""
from pathlib import Path
import hashlib, io, json, zipfile, requests
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
ROOT=Path(__file__).resolve().parent

def obtain():
 raw=ROOT/'data'/'raw'; raw.mkdir(parents=True,exist_ok=True)
 specs={'adult':'https://archive.ics.uci.edu/static/public/2/adult.zip','bank':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip','gas':'https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip','electricity':'https://maxhalford.github.io/files/datasets/electricity.zip'}
 reference_file=ROOT/'data'/'sources.json'
 expected={r['name']:r['sha256'] for r in json.loads(reference_file.read_text())} if reference_file.exists() else {}
 records=[]
 for name,url in specs.items():
  f=raw/f'{name}.zip'
  if not f.exists():
   r=requests.get(url,timeout=90);r.raise_for_status();f.write_bytes(r.content)
  digest=hashlib.sha256(f.read_bytes()).hexdigest()
  if name in expected and digest!=expected[name]:raise ValueError('Source archive checksum changed: '+name)
  if not (raw/name).exists():zipfile.ZipFile(f).extractall(raw/name)
  records.append(dict(name=name,url=url,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
 (ROOT/'data'/'sources.json').write_text(json.dumps(records,indent=2))
 for fn in ['bank.zip','bank-additional.zip']:
  z=raw/'bank'/fn
  if z.exists():zipfile.ZipFile(z).extractall(raw/'bank'/fn[:-4])

def load(name):
 raw=ROOT/'data'/'raw'
 if name=='adult':
  cols=['age','workclass','fnlwgt','education','education_num','marital','occupation','relationship','race','sex','capital_gain','capital_loss','hours','country','target']
  a=pd.read_csv(raw/'adult'/'adult.data',names=cols,skipinitialspace=True,na_values='?')
  b=pd.read_csv(raw/'adult'/'adult.test',names=cols,skipinitialspace=True,na_values='?',comment='|')
  df=pd.concat([a,b],ignore_index=True); y=df.pop('target').str.rstrip('.').eq('>50K').astype(int).to_numpy()
  # Original training set split deterministically into fit/calibration; original test is held out.
  fit=np.arange(int(.8*len(a))); cal=np.arange(len(fit),len(a)); test=np.arange(len(a),len(df))
 elif name=='bank':
  f=next((raw/'bank').rglob('bank-additional-full.csv'));df=pd.read_csv(f,sep=';');y=df.pop('y').eq('yes').astype(int).to_numpy()
  df=df.drop(columns=['duration']) # Not known before the marketing call.
  n=len(df);fit=np.arange(int(.5*n));cal=np.arange(int(.5*n),int(.65*n));test=np.arange(int(.65*n),n)
 elif name=='electricity':
  f=next((raw/'electricity').rglob('electricity.csv'));df=pd.read_csv(f);y=df.pop('class').eq('UP').astype(int).to_numpy()
  n=len(df);fit=np.arange(int(.5*n));cal=np.arange(int(.5*n),int(.65*n));test=np.arange(int(.65*n),n)
 else:
  xs=[];ys=[];bs=[]
  for b in range(1,11):
   f=raw/'gas'/'Dataset'/f'batch{b}.dat';x,yb=load_svmlight_file(str(f),n_features=128);xs.append(x.toarray());ys.append(yb.astype(int)-1);bs.extend([b]*len(yb))
  df=pd.DataFrame(np.concatenate(xs));y=np.concatenate(ys);bs=np.asarray(bs);fit=np.where(bs<=3)[0];cal=np.where(bs==4)[0];test=np.where(bs>=5)[0]
 num=list(df.select_dtypes(include=np.number).columns);cat=[c for c in df.columns if c not in num]
 # Fit transformations using training records only.
 trans=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cat)],sparse_threshold=0)
 xf=trans.fit_transform(df.iloc[fit]).astype(np.float32);xc=trans.transform(df.iloc[cal]).astype(np.float32);xt=trans.transform(df.iloc[test]).astype(np.float32)
 out=ROOT/'data'/'processed';out.mkdir(exist_ok=True)
 np.savez_compressed(out/f'{name}.npz',x_train=xf,y_train=y[fit],x_cal=xc,y_cal=y[cal],x_test=xt,y_test=y[test],source_index=test)
 return dict(dataset=name,total=len(df),train=len(fit),calibration=len(cal),test=len(test),encoded_features=xf.shape[1],classes=len(np.unique(y)),ordering='original predefined split' if name=='adult' else 'source chronological order')
if __name__=='__main__':
 obtain();r=[load(n) for n in ['adult','bank','electricity','gas']];(ROOT/'data'/'dataset_summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
