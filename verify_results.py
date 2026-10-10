"""Verify immutable trace hashes and independently regenerated result tables."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--traces-only',action='store_true');a=p.parse_args()
 manifest=json.loads((ROOT/'data/trace_manifest.json').read_text())
 for rel,digest in manifest['files'].items():
  actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
  if actual!=digest:raise AssertionError('Trace checksum differs: '+rel)
 print('Verified',len(manifest['files']),'frozen trace files.')
 if a.traces_only:return
 count=0
 for f in sorted((ROOT/'expected').rglob('*.csv')):
  rel=f.relative_to(ROOT/'expected');g=ROOT/'results'/rel
  x=pd.read_csv(f);y=pd.read_csv(g)
  if list(x.columns)!=list(y.columns) or x.shape!=y.shape:raise AssertionError('Table structure differs: '+str(rel))
  for c in x:
   if pd.api.types.is_numeric_dtype(x[c]):
    if not np.allclose(x[c].to_numpy(),y[c].to_numpy(),atol=1e-10,rtol=1e-10,equal_nan=True):raise AssertionError(f'Numerical mismatch: {rel}/{c}')
   elif not x[c].fillna('').equals(y[c].fillna('')):raise AssertionError(f'Categorical mismatch: {rel}/{c}')
  count+=1
 exp=json.loads((ROOT/'expected/analysis/contrasts.json').read_text());act=json.loads((ROOT/'results/analysis/contrasts.json').read_text())
 for k in exp:
  for v in exp[k]:
   if not np.isclose(exp[k][v],act[k][v],rtol=1e-10,atol=1e-10):raise AssertionError('Contrast mismatch: '+k+'/'+v)
 print('Verified',count,'result tables and',len(exp),'paired contrasts.')
if __name__=='__main__':main()
