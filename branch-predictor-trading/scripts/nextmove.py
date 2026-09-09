import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lobmodels import feats
for path in sys.argv[1:]:
    z=np.load(path); bp,bq,ap,aq=z['bp'],z['bq'],z['ap'],z['aq']; X,mid,imb=feats(bp,bq,ap,aq)
    # next mid move: for each sample t, direction of the first future mid change, and how many samples until it
    N=len(mid); nxt=np.zeros(N); wait=np.zeros(N); last_dir=0; last_t=N
    for t in range(N-2,-1,-1):
        if mid[t+1]!=mid[t]: last_dir=np.sign(mid[t+1]-mid[t]); last_t=t+1
        nxt[t]=last_dir; wait[t]=last_t-t
    ok=nxt!=0; y=(nxt>0).astype(int)
    spread=(ap[:,0]-bp[:,0])/mid*1e4; tick=np.min(np.diff(np.unique(ap[:,0])))/np.median(mid)*1e4
    print(f"\n=== {path}: {N:,} samples | tick {tick:.2f} bps | median spread {np.median(spread):.2f} bps | median wait to next mid move {np.median(wait[ok])/10:.1f}s ===")
    # 1. imbalance sign
    sel=ok&(imb!=0); print(f"queue imbalance sign -> next move: acc {np.mean((imb[sel]>0)==(y[sel]==1))*100:.1f}%  (n={sel.sum():,})")
    for lo,hi in [(-1,-0.6),(-0.6,-0.2),(-0.2,0.2),(0.2,0.6),(0.6,1.01)]:
        m=ok&(imb>=lo)&(imb<hi); print(f"   imbalance in [{lo:+.1f},{hi:+.1f}): P(next move up) = {y[m].mean()*100:5.1f}%  n={m.sum():,}")
    # 2/3. learned models, train first 60%, test last 40%
    cut=int(N*0.6); tr=ok.copy(); tr[cut:]=False; te=ok.copy(); te[:cut]=False
    lr=LogisticRegression(max_iter=500).fit(X[tr],y[tr]); p=lr.predict_proba(X[te])[:,1]
    gb=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,random_state=0).fit(X[tr],y[tr]); pg=gb.predict_proba(X[te])[:,1]
    for name,pp in [('logistic',p),('gradient boosting',pg)]:
        acc=np.mean((pp>0.5)==(y[te]==1)); conf=np.abs(pp-0.5)
        top=conf>=np.quantile(conf,0.8); acc_top=np.mean((pp[top]>0.5)==(y[te][top]==1))
        print(f"{name:18} -> next move: acc {acc*100:.1f}% overall, {acc_top*100:.1f}% on the 20% most confident (n={top.sum():,})")
    # economics for a maker who joins the queue on the predicted side and earns half spread if right, loses half spread + a tick if wrong
    hs=np.median(spread)/2; 
    for name,pp in [('logistic',p),('gradient boosting',pg)]:
        conf=np.abs(pp-0.5); top=conf>=np.quantile(conf,0.8); a=np.mean((pp[top]>0.5)==(y[te][top]==1))
        print(f"   {name}: if right earn +{hs:.2f} bps (half spread), if wrong lose {hs+tick:.2f} bps -> expected {a*hs-(1-a)*(hs+tick):+.2f} bps per fill before fees")
