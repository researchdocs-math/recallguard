"""Secondary exact-audit comparison and implementation consistency checks."""
import argparse,gzip,json,itertools
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--runs",default="results/runs");ap.add_argument("--out",default="results/analysis");args=ap.parse_args()
 c=json.loads((ROOT/'configs/frozen_exact.json').read_text());rows=[];comparisons=0
 for t,s,n in itertools.product(c['topics'],c['seeds'],c['scenarios']):
  stem=f'{t}__{n}__frozen_exact__{s}__a0.05__t0.95.json.gz'
  d=json.loads(gzip.decompress((ROOT/args.runs/stem).read_bytes()));old=json.loads(gzip.decompress((ROOT/args.runs/stem.replace('frozen_exact','frozen')).read_bytes()))
  for k in ['found','screen','reference_calls','audit','recall','cycle']:
   assert d['stops']['dynamic'][k]==old['stops']['dynamic'][k],(stem,k)
  comparisons+=1
  for method in ['dynamic','exact']:
   x={k:d[k] for k in ['topic','N','R','seed','scenario']};x.update(d['stops'][method]);x['certificate']=method;x['violation']=int(x['recall']<.95-1e-12)
   for rho in [1,2,4,8]:x[f'cost{rho}']=(x['screen']+rho*x['reference_calls'])/(rho*x['N'])
   rows.append(x)
 out=ROOT/args.out;df=pd.DataFrame(rows);df.to_csv(out/'frozen_exact_stops.csv',index=False)
 a=df.groupby(['scenario','certificate'])[['violation','recall','cost4']].mean();a.to_csv(out/'frozen_exact_summary.csv');print(a.to_string())
 print('Matched primary frozen trajectories',comparisons)
 print(df.groupby('certificate')[['violation','recall','cost1','cost2','cost4','cost8']].mean().to_string())
 (out/'secondary_validation.json').write_text(json.dumps({'matched_primary_trajectories':comparisons,'exact_false_certifications':int(df[df.certificate=='exact'].violation.sum())},indent=2))
if __name__=='__main__':main()
