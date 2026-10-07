"""Unsupervised text features, fitted separately within each review corpus."""
import argparse,concurrent.futures,gzip,json,os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parents[1]
def build(path):
    topic=path.name.removesuffix(".jsonl.gz")
    rows=[json.loads(x) for x in gzip.decompress(path.read_bytes()).decode().split('\n') if x]
    v=TfidfVectorizer(max_features=10000,min_df=2,ngram_range=(1,2),sublinear_tf=True,stop_words="english",dtype=np.float64)
    X=v.fit_transform([r["title"]+" "+r["abstract"] for r in rows])
    out=ROOT/"data/features";out.mkdir(exist_ok=True)
    sparse.save_npz(out/(topic+".npz"),X)
    np.save(out/(topic+"_y.npy"),np.array([r["label"] for r in rows],dtype=np.int8))
    return topic,X.shape,X.nnz
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=6);ap.add_argument("--topics",nargs="*");args=ap.parse_args()
    paths=sorted((ROOT/"data/processed").glob("*.gz"))
    if args.topics:paths=[p for p in paths if p.name.removesuffix(".jsonl.gz") in args.topics]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(build,paths):print(r,flush=True)
