"""Vectorized finite-population stress tests on CPU or Metal GPU."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def simulate(tape,N,R,bias,device,dtype):
    u=torch.as_tensor(tape,device=device,dtype=dtype);B=tape.shape[1]
    n=torch.full((B,),N,device=device,dtype=dtype)
    m=torch.full((B,),R,device=device,dtype=dtype)
    g=torch.tensor([.1,.25,.5,.75,.9,.99],device=device,dtype=dtype)
    w=torch.zeros((B,6),device=device,dtype=dtype);mx=torch.zeros(B,device=device,dtype=dtype)
    # Each targeted removal is intentionally informative. This synthetic
    # selector is an adversarial stress design, not a deployable ranker.
    for k in range(N//2):
        prob=bias*m/(bias*m+n-m).clamp(min=1e-12)
        yt=(u[k,:,0]<prob).to(dtype);m=m-yt;n=n-1
        pr=m/n;ya=(u[k,:,1]<pr).to(dtype)
        neg=torch.log1p(g[None,:]*pr[:,None]/(1-pr[:,None]).clamp(min=1e-12))
        w=w+torch.where(ya[:,None]>0,torch.log1p(-g)[None,:],neg)
        e=torch.logsumexp(w,dim=1)-np.log(6)
        mx=torch.maximum(mx,e);m=m-ya;n=n-1
    return mx.detach().cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--paths",type=int,default=30000);ap.add_argument("--device",default="auto");args=ap.parse_args()
    device="mps" if args.device=="auto" and torch.backends.mps.is_available() else ("cpu" if args.device=="auto" else args.device)
    torch.set_num_threads(2);out=[];N=500
    for R in [5,25,100]:
        for bias in [.25,1.,4.]:
            seed=100000+R*100+int(bias*10);rng=np.random.default_rng(seed)
            tape=rng.random((N//2,args.paths,2),dtype=np.float32)
            tic=time.monotonic();mx=simulate(tape,N,R,bias,device,torch.float32)
            cpu=simulate(tape[:,:128,:],N,R,bias,"cpu",torch.float64)
            err=float(np.max(np.abs(cpu-mx[:128])))
            assert err<.002,(R,bias,err)
            assert np.array_equal(cpu>=np.log(20),mx[:128]>=np.log(20))
            row=dict(N=N,R=R,bias=bias,paths=args.paths,seed=seed,device=device,crossings=int((mx>=np.log(20)).sum()),rate=float((mx>=np.log(20)).mean()),maximum_cpu_gpu_log_difference=err,seconds=time.monotonic()-tic)
            out.append(row);print(row,flush=True)
    (ROOT/"results/martingale_validation.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
