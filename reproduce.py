"""Run all experiment suites and verify numerical reproduction."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 if a.workers<1:raise ValueError('workers must be positive')
 cmds=[['verify_results.py','--traces-only'],['tests/test_monitor.py']]
 cmds += [['run_experiments.py','--suite',s,'--workers',str(a.workers)] for s in ['public','synthetic','expedite']]
 cmds += [['additional_experiments.py','--workers',str(a.workers)],['analyze_results.py'],['verify_results.py']]
 for cmd in cmds:
  print('Running:', ' '.join(cmd),flush=True);subprocess.run([sys.executable]+cmd,cwd=ROOT,check=True)
 print('All experiment results reproduced and verified.',flush=True)
if __name__=='__main__':main()
