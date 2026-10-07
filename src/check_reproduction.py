"""Validate data integrity and compare complete deterministic experiment traces."""
import argparse,gzip,hashlib,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',default='results/reproduced');ap.add_argument('--config',nargs='+',default=['configs/main.json','configs/frozen_exact.json']);args=ap.parse_args()
 for d in json.loads((ROOT/'data/datasets.json').read_text()):
  p=ROOT/'data/processed'/(d['topic']+'.jsonl.gz');assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256'],p
 count=0
 for cfg in args.config:
  c=json.loads((ROOT/cfg).read_text())
  for t,s,n,p,a,r in itertools.product(c['topics'],c['seeds'],c['scenarios'],c['policies'],c.get('alphas',[.05]),c.get('targets',[.95])):
   fn=f'{t}__{n}__{p}__{s}__a{a}__t{r}.json.gz';expected=json.loads(gzip.decompress((ROOT/'results/runs'/fn).read_bytes()));actual=json.loads(gzip.decompress((ROOT/args.runs/fn).read_bytes()))
   expected.pop('seconds',None);actual.pop('seconds',None)
   assert actual==expected,f'Numerical or trajectory mismatch: {fn}'
   count+=1
 print(f'PASS: all 26 processed-data checksums and {count} complete experiment traces match.')
if __name__=='__main__':main()
