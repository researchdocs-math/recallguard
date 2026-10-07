"""Anytime recall certificates; all inference uses reference adjudications only."""
import numpy as np
from scipy.special import betaln, gammaln, logsumexp
GAMMAS = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.99],dtype=float)

class ShrinkingAudit:
    def __init__(self, alpha=0.05):
        self.alpha=alpha; self.f=[]; self.n=[]; self.y=[]
    def observe(self, found, remaining, label):
        assert remaining>0 and label in (0,1)
        self.f.append(found); self.n.append(remaining); self.y.append(label)
    def log_e(self, total):
        if not self.y:return 0.0
        f=np.asarray(self.f); n=np.asarray(self.n); y=np.asarray(self.y)
        if np.any(total<f) or np.any(total>f+n):return np.inf
        p=(total-f)/n
        if np.any((p==0)&(y==1)) or np.any((p==1)&(y==0)):return np.inf
        zero=p[y==0]
        # A positive draw multiplies wealth by 1-gamma, including p=0
        # only after the logically impossible outcome has been excluded above.
        logw=np.log1p(-GAMMAS)*int(y.sum())
        if len(zero): logw += np.log1p(GAMMAS[:,None]*zero[None,:]/(1-zero[None,:])).sum(axis=1)
        return float(logsumexp(logw)-np.log(len(GAMMAS)))
    def certifies(self, found, remaining, target=0.95):
        if found==0:return remaining==0
        first_bad=int(np.floor(found/target+1e-10))+1
        if first_bad>found+remaining:return True
        return self.log_e(first_bad)>=np.log(1/self.alpha)
    def upper(self,found,remaining):
        lo=found; hi=found+remaining
        while lo<hi:
            mid=(lo+hi+1)//2
            if self.log_e(mid)<np.log(1/self.alpha):lo=mid
            else:hi=mid-1
        return lo

def prn_log_e(total,N,n,x,a=1/99,b=1):
    if total<x or N-total<n-x:return np.inf
    loglik=(gammaln(total+1)-gammaln(total-x+1)+gammaln(N-total+1)-gammaln(N-total-n+x+1)-gammaln(N+1)+gammaln(N-n+1))
    return float(betaln(a+x,b+n-x)-betaln(a,b)-loglik)

def prn_upper(N,n,x,found,remaining,alpha=0.05):
    if n==0:return found+remaining
    # The prior/posterior ratio is convex in the hypothesized total.
    # Start on its nondecreasing side, then truncate by logical bounds.
    lo=max(found,int(np.floor(N*x/n)));hi=found+remaining
    if lo>hi:return hi
    if prn_log_e(lo,N,n,x)>=np.log(1/alpha):return hi  # empty/inconsistent CS: never certify
    while lo<hi:
        mid=(lo+hi+1)//2
        if prn_log_e(mid,N,n,x)<np.log(1/alpha):lo=mid
        else:hi=mid-1
    return lo

def robust_recall(found,upper,error_budget):
    if upper==0:return 1.0 if error_budget==0 else 0.0
    if error_budget>=found:return 0.0
    b=int(error_budget);f=min(b,found)
    return max(0.,min(found/(upper+b),(found-f)/(upper-f) if upper>f else 1.))
