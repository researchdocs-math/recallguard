"""Exact enumeration tests do not rely on the experiment simulator."""
import itertools,sys,unittest
from pathlib import Path
import numpy as np
from scipy.stats import betabinom,hypergeom
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from certificates import ShrinkingAudit,prn_log_e,robust_recall

class Certificates(unittest.TestCase):
    def test_prior_posterior_identity(self):
        for N in [9,23]:
            for n in range(1,N):
                for x in range(n+1):
                    for R in range(x,N-n+x+1):
                        direct=betabinom.logpmf(R,N,1/99,1)-betabinom.logpmf(R-x,N-n,1/99+x,1+n-x)
                        self.assertAlmostEqual(prn_log_e(R,N,n,x),direct,places=9)
    def test_martingale_one_step(self):
        for p in [0.001,0.2,0.7,0.999]:
            for g in [.1,.5,.99]:
                self.assertAlmostEqual((1-p)*(1+g*p/(1-p))+p*(1-g),1.)
    def test_exhaustive_adaptive_removal(self):
        # Enumerate all audit branches. Target removals depend on previous
        # revealed labels and remove either endpoint, sometimes a positive.
        worst=0.
        for N in range(3,8):
            for bits in itertools.product([0,1],repeat=N):
                if not sum(bits):continue
                def walk(rem,found,cert,prob,bad):
                    nonlocal worst
                    if cert.certifies(found,len(rem),.7) and found/sum(bits)<.7:bad=True
                    if not rem:return prob if bad else 0.
                    # Alternate predictable targeted adjudication and random audit.
                    if len(rem)%3==0:
                        j=0 if found%2 else len(rem)-1
                        return walk(rem[:j]+rem[j+1:],found+rem[j],cert,prob,bad)
                    acc=0.
                    for j,y in enumerate(rem):
                        c=ShrinkingAudit(.2);c.f=cert.f.copy();c.n=cert.n.copy();c.y=cert.y.copy()
                        c.observe(found,len(rem),y)
                        acc+=walk(rem[:j]+rem[j+1:],found+y,c,prob/len(rem),bad)
                    return acc
                rate=walk(tuple(bits),0,ShrinkingAudit(.2),1.,False)
                self.assertLessEqual(rate,.2+1e-12);worst=max(worst,rate)
        print('exhaustive worst crossing probability',worst)
    def test_robust_bound(self):
        for N in range(1,9):
            for R in range(1,N+1):
                for F in range(R+1):
                    for b in range(N+1):
                        bound=robust_recall(F,R,b)
                        for falseaccepted in range(min(F,b)+1):
                            for missed in range(min(N-R,b-falseaccepted)+1):
                                den=R-falseaccepted+missed
                                if den:self.assertLessEqual(bound,(F-falseaccepted)/den+1e-12)
if __name__=='__main__':unittest.main()
