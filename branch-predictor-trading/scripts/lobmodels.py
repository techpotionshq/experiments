"""Compare LOB predictors on Bybit 10-level books sampled at 100 ms.
Label: direction of smoothed mid over horizon H samples (DeepLOB convention): up / flat / down with threshold alpha.
Models: (1) queue imbalance sign, (2) logistic on handcrafted features, (3) gradient boosting on same, (4) DeepLOB CNN-LSTM on raw 100x40 windows.
Train on day A, early-stop on day B, report on day C. Then convert to economics: mean realised mid change conditional on a confident up/down call."""
import sys, time, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
import torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
H=int(sys.argv[4]) if len(sys.argv)>4 else 10        # horizon in samples (10 = 1 s)
K=int(sys.argv[5]) if len(sys.argv)>5 else 5         # smoothing window
ALPHA=float(sys.argv[6]) if len(sys.argv)>6 else None  # label threshold in relative mid change; default = half a tick
def load(p):
    z=np.load(p); return z['bp'],z['bq'],z['ap'],z['aq']
def feats(bp,bq,ap,aq):
    mid=(bp[:,0]+ap[:,0])/2
    imb1=(bq[:,0]-aq[:,0])/(bq[:,0]+aq[:,0])
    imb3=(bq[:,:3].sum(1)-aq[:,:3].sum(1))/(bq[:,:3].sum(1)+aq[:,:3].sum(1))
    imb10=(bq.sum(1)-aq.sum(1))/(bq.sum(1)+aq.sum(1))
    spread=(ap[:,0]-bp[:,0])/mid
    # order flow imbalance (Cont et al.) at best level, summed over last 10 and 50 samples
    e=np.zeros(len(mid))
    for t in range(1,len(mid)):
        eb = bq[t,0] if bp[t,0]>bp[t-1,0] else (-bq[t-1,0] if bp[t,0]<bp[t-1,0] else bq[t,0]-bq[t-1,0])
        ea = aq[t,0] if ap[t,0]<ap[t-1,0] else (-aq[t-1,0] if ap[t,0]>ap[t-1,0] else aq[t,0]-aq[t-1,0])
        e[t]=eb-ea
    c=np.cumsum(e); ofi10=np.r_[np.zeros(10), c[10:]-c[:-10]]; ofi50=np.r_[np.zeros(50), c[50:]-c[:-50]]
    r10=np.r_[np.zeros(10), np.log(mid[10:]/mid[:-10])]; r50=np.r_[np.zeros(50), np.log(mid[50:]/mid[:-50])]
    depthratio=np.log((bq[:,:5].sum(1)+1e-9)/(aq[:,:5].sum(1)+1e-9))
    X=np.c_[imb1,imb3,imb10,spread,ofi10/(bq[:,0]+aq[:,0]+1e-9),ofi50/(bq[:,:3].sum(1)+aq[:,:3].sum(1)+1e-9),r10*1e4,r50*1e4,depthratio]
    return np.nan_to_num(X), mid, imb1
def labels(mid, H, K, alpha):
    # DeepLOB-style: compare mean of next H mids with mean of previous K mids
    cs=np.cumsum(np.r_[0.0,mid]); fut=(cs[K+H:]-cs[K:-H])/H if False else None
    m_prev=np.array([mid[max(0,t-K+1):t+1].mean() for t in range(len(mid))])
    m_next=np.full(len(mid),np.nan); s=np.cumsum(np.r_[0.0,mid]); m_next[:-H]=(s[H+1:]-s[1:-H])/H
    l=(m_next-m_prev)/m_prev; y=np.where(l>alpha,2,np.where(l<-alpha,0,1)); y[np.isnan(l)]=-1
    real=np.full(len(mid),np.nan); real[:-H]=(mid[H:]-mid[:-H])/mid[:-H]*1e4   # realised mid change in bps over H
    return y, real
def raw_windows(bp,bq,ap,aq,W=100):
    # DeepLOB input: (N, 1, W, 40) with levels [ap1,aq1,bp1,bq1,...], z-scored per day
    X=np.zeros((len(bp),40),dtype=np.float32)
    for i in range(10): X[:,4*i]=ap[:,i]; X[:,4*i+1]=aq[:,i]; X[:,4*i+2]=bp[:,i]; X[:,4*i+3]=bq[:,i]
    X=(X-X.mean(0))/(X.std(0)+1e-9); return X
class DeepLOB(nn.Module):
    def __init__(s,y_len=3):
        super().__init__()
        s.c1=nn.Sequential(nn.Conv2d(1,32,(1,2),(1,2)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01))
        s.c2=nn.Sequential(nn.Conv2d(32,32,(1,2),(1,2)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01))
        s.c3=nn.Sequential(nn.Conv2d(32,32,(1,10)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01),nn.Conv2d(32,32,(4,1),padding=(2,0)),nn.LeakyReLU(0.01))
        s.i1=nn.Sequential(nn.Conv2d(32,64,(1,1)),nn.LeakyReLU(0.01),nn.Conv2d(64,64,(3,1),padding=(1,0)),nn.LeakyReLU(0.01))
        s.i2=nn.Sequential(nn.Conv2d(32,64,(1,1)),nn.LeakyReLU(0.01),nn.Conv2d(64,64,(5,1),padding=(2,0)),nn.LeakyReLU(0.01))
        s.i3=nn.Sequential(nn.MaxPool2d((3,1),stride=(1,1),padding=(1,0)),nn.Conv2d(32,64,(1,1)),nn.LeakyReLU(0.01))
        s.lstm=nn.LSTM(192,64,batch_first=True); s.fc=nn.Linear(64,y_len)
    def forward(s,x):
        x=s.c1(x); x=s.c2(x); x=s.c3(x)
        x=torch.cat([s.i1(x),s.i2(x),s.i3(x)],1)          # (B,192,T',1)
        x=x.squeeze(3).permute(0,2,1); o,_=s.lstm(x); return s.fc(o[:,-1])
def batches(X,y,idx,W,bs):
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]; xb=np.stack([X[t-W+1:t+1] for t in j])[:,None]; yield torch.from_numpy(xb), torch.from_numpy(y[j]).long()
def evaluate(name,p,y,real,mask):
    pred=p.argmax(1); acc=accuracy_score(y[mask],pred[mask]); f1=f1_score(y[mask],pred[mask],average='macro')
    conf=p.max(1); out=[]
    for q in [0.0,0.5,0.8]:
        thr=np.quantile(conf[mask],q); sel=mask&(conf>=thr)&(pred!=1)
        up=sel&(pred==2); dn=sel&(pred==0)
        edge=(np.nanmean(real[up]) if up.any() else 0) - (np.nanmean(real[dn]) if dn.any() else 0)
        hit=(np.mean(real[up]>0) if up.any() else np.nan, np.mean(real[dn]<0) if dn.any() else np.nan)
        out.append((q,sel.sum(),edge/2,hit))
    print(f"{name:22} acc {acc*100:5.1f}%  macroF1 {f1*100:5.1f}  | " + " | ".join(f"top{int((1-q)*100)}%: n={n:6d} edge {e:5.2f}bps hit {h[0]*100:4.0f}/{h[1]*100:4.0f}" for q,n,e,h in out))
def main():
    A,B,C=sys.argv[1:4]; W=100
    data={k:load(p) for k,p in zip('ABC',[A,B,C])}
    F={}; 
    for k,(bp,bq,ap,aq) in data.items():
        X,mid,imb=feats(bp,bq,ap,aq); tick=np.min(np.diff(np.unique(ap[:,0])))
        alpha=ALPHA if ALPHA is not None else 0.5*tick/np.median(mid)
        y,real=labels(mid,H,K,alpha); F[k]=(X,y,real,imb,raw_windows(bp,bq,ap,aq),mid)
    XA,yA,_,_,RA,_=F['A']; XB,yB,_,_,RB,_=F['B']; XC,yC,realC,imbC,RC,midC=F['C']
    mC=(yC>=0); mC[:W]=False
    print(f"H={H} samples ({H/10:.1f}s), K={K}, alpha={alpha*1e4:.2f} bps | test-day label mix down/flat/up: {[(yC[mC]==i).mean().round(3) for i in range(3)]}")
    # 1. imbalance sign
    p=np.zeros((len(yC),3)); p[:,2]=np.clip(imbC,0,1); p[:,0]=np.clip(-imbC,0,1); p[:,1]=1-np.abs(imbC); evaluate('queue imbalance sign',p,yC,realC,mC)
    # 2. logistic
    tr=yA>=0; lr=LogisticRegression(max_iter=500,C=1.0).fit(XA[tr],yA[tr]); evaluate('logistic, 9 features',lr.predict_proba(XC),yC,realC,mC)
    # 3. gradient boosting
    gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=31,early_stopping=True,validation_fraction=0.1,random_state=0).fit(XA[tr],yA[tr])
    evaluate('gradient boosting, 9 f',gb.predict_proba(XC),yC,realC,mC)
    # 4. DeepLOB
    torch.set_num_threads(4); net=DeepLOB(); opt=torch.optim.Adam(net.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss()
    idxA=np.where(yA>=0)[0]; idxA=idxA[idxA>=W]; idxB=np.where(yB>=0)[0]; idxB=idxB[idxB>=W]
    rng=np.random.default_rng(0); nA=int(sys.argv[7]) if len(sys.argv)>7 else 120000; nB=20000
    subA=np.sort(rng.choice(idxA,min(nA,len(idxA)),replace=False)); subB=np.sort(rng.choice(idxB,min(nB,len(idxB)),replace=False))
    best=1e9; bad=0; t0=time.time()
    for ep in range(6):
        net.train(); rng.shuffle(subA)
        for xb,yb in batches(RA,yA,subA,W,256):
            opt.zero_grad(); l=lossf(net(xb),yb); l.backward(); opt.step()
        net.eval(); vl=0; n=0
        with torch.no_grad():
            for xb,yb in batches(RB,yB,subB,W,1024): vl+=lossf(net(xb),yb).item()*len(yb); n+=len(yb)
        vl/=n; print(f"  DeepLOB epoch {ep+1}: val loss {vl:.4f} ({time.time()-t0:.0f}s)")
        if vl<best-1e-4: best=vl; bad=0; torch.save(net.state_dict(),'deeplob_best.pt')
        else:
            bad+=1
            if bad>=2: break
    net.load_state_dict(torch.load('deeplob_best.pt')); net.eval()
    idxC=np.where(mC)[0]; p=np.zeros((len(yC),3)); p[:,1]=1
    with torch.no_grad():
        for i in range(0,len(idxC),2048):
            j=idxC[i:i+2048]; xb=torch.from_numpy(np.stack([RC[t-W+1:t+1] for t in j])[:,None]); p[j]=torch.softmax(net(xb),1).numpy()
    evaluate('DeepLOB (CNN-LSTM)',p,yC,realC,mC)
    print(f"\nreference: spread on test day median {np.median((data['C'][2][:,0]-data['C'][0][:,0])/midC*1e4):.2f} bps; Bybit spot fee base 10 bps maker / 10 bps taker")
if __name__=='__main__': main()
