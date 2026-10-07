"""Sequential screening with fallible first-pass labels and reference audits."""
import argparse,concurrent.futures,gzip,json,os,time,hashlib
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from certificates import ShrinkingAudit,prn_upper
ROOT=Path(__file__).resolve().parents[1]

def make_noise(y,X,seed,scenario,positive_seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,314159]))
    weak=y.copy();p=np.zeros(len(y));fp=0.
    if scenario.startswith("uniform"):
        p[:]=int(scenario.removeprefix("uniform"))/100;fp=.01
    elif scenario=="difficulty20":
        sim=np.asarray(X@X[positive_seed].T.toarray()).ravel()
        pos=np.flatnonzero(y);order=pos[np.argsort(sim[pos],kind="stable")]
        p[order]=np.linspace(.4,0,len(pos));fp=.01
    elif scenario=="cluster20":
        # A persistent lexical blind spot: 20% of relevant documents least
        # similar to the supplied relevant seed receive negative first-pass labels.
        sim=np.asarray(X@X[positive_seed].T.toarray()).ravel()
        pos=np.flatnonzero(y);order=pos[np.argsort(sim[pos],kind="stable")]
        p[order[:int(np.ceil(.2*len(pos)))]]=1.;fp=.01
    elif scenario!="clean":raise ValueError(scenario)
    weak[(y==1)&(rng.random(len(y))<p)]=0
    weak[(y==0)&(rng.random(len(y))<fp)]=1
    return weak

def run_one(task):
    topic,seed,scenario,policy,alpha,target=task[:6]
    options=task[6] if len(task)>6 else {}
    frozen_policy=policy in ("frozen","frozen_exact")
    key=f"{topic}__{scenario}__{policy}__{seed}__a{alpha}__t{target}"
    out=ROOT/options.get("output","results/runs")/(key+".json.gz")
    if out.exists() and not options.get("force",False):return key,"cached"
    tic=time.monotonic();X=sparse.load_npz(ROOT/"data/features"/(topic+".npz"));y=np.load(ROOT/"data/features"/(topic+"_y.npy"));N=len(y)
    random_seed=seed+int(hashlib.sha256(topic.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(np.random.SeedSequence([random_seed,2718]))
    init=np.random.default_rng(np.random.SeedSequence([random_seed,42]))
    ps=int(init.choice(np.flatnonzero(y)));ns=int(init.choice(np.flatnonzero(1-y)))
    weak=make_noise(y,X,random_seed,scenario,ps)
    permutation=rng.permutation(N); pointer=0;prefix=0;prefix_positive=0
    # -1 unknown, 0/1 reference adjudicated. Weak negative decisions stay unknown.
    gold=np.full(N,-1,dtype=np.int8);gold[ps]=1;gold[ns]=0
    screened=np.zeros(N,dtype=bool);screened[[ps,ns]]=True
    F=1;verified=2;weak_removed=0;negative_streak=0;frozen=False;screen_calls=0;random_calls=0;repair_calls=0;repair_hits=0;screen_hits=0;falsepos=0
    cert=ShrinkingAudit(alpha);model=LogisticRegression(C=1.,solver="liblinear",max_iter=200,tol=1e-4,random_state=0)
    checkpoints=[];stops={};cycle=0;freeze_F=None;freeze_n=None;freeze_looks=0
    def adjudicate(i,route):
        nonlocal F,verified,random_calls,repair_calls,repair_hits,falsepos
        assert gold[i]<0
        label=int(y[i]);gold[i]=label;F+=label;verified+=1
        if route=="audit":random_calls+=1
        if route=="repair":repair_calls+=1;repair_hits+=label
        if route=="positive" and label==0:falsepos+=1
        return label
    while verified<N:
        cycle+=1
        # Every decision depends only on text, weak labels already requested,
        # and adjudications already returned. Hidden reference labels are not predictors.
        pos=np.flatnonzero(gold==1);neg=np.flatnonzero((gold==0)|((gold<0)&screened))
        if len(neg)>2000:
            # A fixed, reproducible subsample limits fitting cost.
            neg=np.random.default_rng(np.random.SeedSequence([seed,cycle,123])).choice(neg,2000,replace=False)
        train=np.r_[pos,neg];labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
        w=np.where(gold[train]<0,.25,1.);w[:len(pos)]*=max(1.,w[len(pos):].sum()/max(1,len(pos)))
        model.fit(X[train],labels,sample_weight=w)
        scores=model.decision_function(X)
        unseen=np.flatnonzero((gold<0)&~screened);rejected=np.flatnonzero((gold<0)&screened)
        if policy in ("forward","blind","frozen","frozen_exact"):nr=0
        elif policy=="fixed":nr=10
        elif policy=="adaptive":
            rrate=(repair_hits+1)/(repair_calls+20)
            srate=(screen_hits+1)/(screen_calls+20)
            share=np.clip(rrate/(rrate+srate/(1+srate)),.05,.5)
            nr=int(round(60*share))
        elif policy=="audit_only":nr=0
        else:raise ValueError(policy)
        if not len(unseen):nr=60
        nr=min(nr,len(rejected));nscreen=min(60-nr,len(unseen))
        if policy=="audit_only" or (frozen_policy and frozen):nr=0;nscreen=0
        if nr:
            chosen=rejected[np.argsort(-scores[rejected],kind="stable")[:nr]]
            for i in chosen:adjudicate(i,"repair")
        if nscreen:
            chosen=unseen[np.argsort(-scores[unseen],kind="stable")[:nscreen]]
            for i in chosen:
                screened[i]=True;screen_calls+=1
                if weak[i]:
                    negative_streak=0
                    label=adjudicate(i,"positive");screen_hits+=label
                else:
                    negative_streak+=1
                if not weak[i] and policy=="blind":
                    # Deliberately invalid diagnostic: treat a weak rejection as final.
                    gold[i]=0;verified+=1;weak_removed+=1
        if frozen_policy and (negative_streak>=200 or not ((gold<0)&~screened).any()):frozen=True
        if frozen_policy and frozen and freeze_F is None:freeze_F=F;freeze_n=N-verified
        if frozen_policy and frozen:freeze_looks+=1
        audit_batch=25 if policy!="audit_only" else 85
        if frozen_policy:audit_batch=85 if frozen else 0
        for j in range(audit_batch):
            while pointer<N and gold[permutation[pointer]]>=0:pointer+=1
            if pointer==N:break
            i=int(permutation[pointer]);pointer+=1
            # Unseen suffix of the independent permutation is hidden from the ranker.
            cert.observe(F,N-verified,int(y[i]))
            adjudicate(i,"audit")
        while prefix<N and gold[permutation[prefix]]>=0:
            prefix_positive+=int(gold[permutation[prefix]]);prefix+=1
        rem=N-verified
        dyn=bool(cert.certifies(F,rem,target))
        upper=prn_upper(N,prefix,prefix_positive,F,rem,alpha)
        prn=bool((F/upper>=target-1e-12) if upper else rem==0)
        row=dict(cycle=cycle,found=F,verified=verified,reference_calls=verified-weak_removed,screen=screen_calls,audit=random_calls,repair=repair_calls,repair_hits=repair_hits,screen_hits=screen_hits,falsepositives=falsepos,remaining=rem,unseen=int(((gold<0)&~screened).sum()),rejected=int(((gold<0)&screened).sum()),recall=F/int(y.sum()),dynamic=dyn,prn=prn,prefix=prefix,prn_upper=upper,log_e=cert.log_e(int(np.floor(F/target+1e-10))+1) if rem else None)
        if row["log_e"] is not None and not np.isfinite(row["log_e"]):row["log_e"]=None
        if policy=="frozen_exact" and frozen:
            k=int(np.floor(F/target+1e-10))+1-freeze_F
            logp=float(hypergeom.logcdf(F-freeze_F,freeze_n,k,random_calls)) if 0<=k<=freeze_n else -np.inf
            exact=bool(logp<=np.log(alpha/(freeze_looks*(freeze_looks+1))))
            row.update(exact=exact,exact_looks=freeze_looks,freeze_found=freeze_F,freeze_remaining=freeze_n)
            if exact and "exact" not in stops:stops["exact"]=row.copy()
        checkpoints.append(row)
        for method,ok in [("dynamic",dyn),("prn",prn)]:
            if ok and method not in stops:stops[method]=row.copy()
        if policy=="frozen_exact":
            if "dynamic" in stops and "exact" in stops:break
        elif len(stops)==2 or (policy=="frozen" and "dynamic" in stops):break
    for method in ["dynamic","prn"]:
        if method not in stops and verified==N:stops[method]=checkpoints[-1].copy()
    result=dict(topic=topic,N=N,R=int(y.sum()),seed=seed,scenario=scenario,policy=policy,alpha=alpha,target=target,weak_false_negatives=int(((weak==0)&(y==1)).sum()),weak_false_positives=int(((weak==1)&(y==0)).sum()),seconds=time.monotonic()-tic,stops=stops,trajectory=checkpoints)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(gzip.compress((json.dumps(result,sort_keys=True)+"\n").encode(),mtime=0))
    return key,round(result["seconds"],2),{m:round(r["recall"],3) for m,r in stops.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/pilot.json");ap.add_argument("--workers",type=int,default=8);ap.add_argument("--output",default="results/runs");ap.add_argument("--force",action="store_true");args=ap.parse_args()
    c=json.loads((ROOT/args.config).read_text());topics=c.get("topics") or [d["topic"] for d in json.loads((ROOT/"data/datasets.json").read_text())]
    tasks=[(t,s,n,p,a,r,{"output":args.output,"force":args.force}) for t in topics for s in c["seeds"] for n in c["scenarios"] for p in c["policies"] for a in c.get("alphas",[.05]) for r in c.get("targets",[.95])]
    print('tasks',len(tasks),'workers',args.workers,flush=True)
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(run_one,tasks,chunksize=1):print(res,flush=True)
if __name__=="__main__":main()
