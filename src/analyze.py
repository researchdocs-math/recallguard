"""Aggregate fixed-protocol experiments using review topics as paired units."""
import argparse,gzip,json,itertools,csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1]
def load(config,runs="results/runs"):
 c=json.loads((ROOT/config).read_text());rows=[];missing=[]
 for t,s,n,p,a,r in itertools.product(c['topics'],c['seeds'],c['scenarios'],c['policies'],c.get('alphas',[.05]),c.get('targets',[.95])):
  path=ROOT/runs/f'{t}__{n}__{p}__{s}__a{a}__t{r}.json.gz'
  if not path.exists():missing.append(str(path));continue
  d=json.loads(gzip.decompress(path.read_bytes()))
  for method,stop in d['stops'].items():
   if p=='frozen' and method=='prn':continue
   row={k:d[k] for k in ['topic','N','R','seed','scenario','policy','alpha','target','seconds']};row.update(stop);row['certificate']=method
   for cost in [1,2,4,8]:row[f'cost{cost}']=(row['screen']+cost*row['reference_calls'])/(cost*row['N'])
   row['violation']=int(row['recall']<row['target']-1e-12)
   row['reference_fraction']=row['reference_calls']/row['N'];row['screen_fraction']=row['screen']/row['N'];rows.append(row)
 return pd.DataFrame(rows),missing

def paired(a,b,rng):
 d=(a-b).to_numpy();ix=rng.integers(0,len(d),(20000,len(d)));bd=d[ix].mean(axis=1)
 signs=rng.choice([-1,1],size=(100000,len(d)));pv=(1+(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-15).sum())/(len(signs)+1)
 # Reduction is averaged within each topic before inference.
 reduction=(1-a/b).to_numpy();br=reduction[ix].mean(axis=1)
 return {'difference':float(d.mean()),'difference_ci':np.quantile(bd,[.025,.975]).tolist(),'p':float(pv),'relative_reduction':float(reduction.mean()),'relative_reduction_ci':np.quantile(br,[.025,.975]).tolist(),'topics_better':int((d<0).sum()),'topics':len(d)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/main.json');ap.add_argument('--allow-partial',action='store_true');ap.add_argument("--runs",default="results/runs");ap.add_argument("--out",default="results/analysis");args=ap.parse_args()
 df,missing=load(args.config,args.runs)
 print('stop records',len(df),'missing runs',len(missing),flush=True)
 if missing and not args.allow_partial:raise RuntimeError('Experiment runs are incomplete')
 out=ROOT/args.out;out.mkdir(exist_ok=True)
 df.to_csv(out/'stops.csv',index=False)
 cols=['recall','violation','cost1','cost2','cost4','cost8','reference_fraction','screen_fraction','audit','repair','repair_hits','falsepositives']
 summary=df.groupby(['scenario','policy','certificate'])[cols].mean().reset_index();summary.to_csv(out/'summary.csv',index=False)
 topic=df.groupby(['topic','scenario','policy','certificate'])[cols].mean().reset_index();topic.to_csv(out/'topic_summary.csv',index=False)
 rng=np.random.default_rng(20261007);comparisons={}
 for scenario in ['all']+sorted(df.scenario.unique().tolist()):
  sub=df if scenario=='all' else df[df.scenario==scenario]
  wide=sub.groupby(['topic','policy','certificate']).cost4.mean().unstack(['policy','certificate']).dropna()
  for name,aa,bb in [('dynamic_vs_prn',('adaptive','dynamic'),('adaptive','prn')),('adaptive_vs_fixed',('adaptive','dynamic'),('fixed','dynamic')),('adaptive_vs_forward',('adaptive','dynamic'),('forward','dynamic')),('adaptive_vs_frozen',('adaptive','dynamic'),('frozen','dynamic'))]:
   if aa in wide.columns and bb in wide.columns:comparisons[scenario+'__'+name]=paired(wide[aa],wide[bb],rng)
 # Holm family correction for the four prespecified aggregate comparisons.
 keys=sorted([k for k in comparisons if k.startswith('all__')],key=lambda k:comparisons[k]['p']);last=0.
 for i,k in enumerate(keys):last=max(last,min(1.,(len(keys)-i)*comparisons[k]['p']));comparisons[k]['holm_p']=last
 (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
 counts=df.groupby(['policy','certificate']).agg(n=('violation','size'),failures=('violation','sum'),mean_recall=('recall','mean'),cost4=('cost4','mean'))
 counts.to_csv(out/'aggregate.csv');print(counts.to_string());print(json.dumps({k:v for k,v in comparisons.items() if k.startswith('all')},indent=2))
 if not missing:
  assert len(df[(df.policy=='adaptive')&(df.certificate=='dynamic')])==1320
  valid=df[(df.policy!='blind')&(df.certificate=='dynamic')]
  report={'n_workflows':6600,'valid_dynamic_workflows':len(valid),'valid_dynamic_failures':int(valid.violation.sum()),'minimum_valid_recall':float(valid.recall.min()),'mean_valid_recall':float(valid.recall.mean()),'topic_count':df.topic.nunique(),'logical_violations':int((df.found>df.R).sum()),'wall_seconds_sum':float(df.drop_duplicates(['topic','seed','scenario','policy']).seconds.sum())}
  (out/'validation.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
