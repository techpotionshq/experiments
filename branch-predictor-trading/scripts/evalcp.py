import numpy as np, torch, sys
sys.argv=['x','','','','10','5']
from lobmodels import load, feats, labels, raw_windows, DeepLOB, evaluate
torch.set_num_threads(4); W=100; H=10; K=5
bp,bq,ap,aq=load('ob/wif03.npz'); X,mid,imb=feats(bp,bq,ap,aq)
tick=np.min(np.diff(np.unique(ap[:,0]))); alpha=0.5*tick/np.median(mid)
y,real=labels(mid,H,K,alpha); R=raw_windows(bp,bq,ap,aq); m=(y>=0); m[:W]=False
net=DeepLOB(); net.load_state_dict(torch.load('deeplob_best.pt')); net.eval()
idx=np.where(m)[0][::5]; m2=np.zeros(len(y),bool); m2[idx]=True; m=m2; p=np.zeros((len(y),3)); p[:,1]=1
with torch.no_grad():
    for i in range(0,len(idx),4096):
        j=idx[i:i+4096]; xb=torch.from_numpy(np.stack([R[t-W+1:t+1] for t in j])[:,None]); p[j]=torch.softmax(net(xb),1).numpy()
print("WIF/USDT, test day 2026-09-03, horizon 1 s, DeepLOB checkpoint after 4 epochs:")
evaluate('DeepLOB (CNN-LSTM)',p,y,real,m)
N=len(mid); nxt=np.zeros(N); last=0
for t in range(N-2,-1,-1):
    if mid[t+1]!=mid[t]: last=np.sign(mid[t+1]-mid[t])
    nxt[t]=last
sel=m&(nxt!=0)&(p[:,2]!=p[:,0]); pu=p[sel,2]>p[sel,0]; d=np.abs(p[sel,2]-p[sel,0]); top=d>=np.quantile(d,0.8)
print(f"DeepLOB up-vs-down score -> next mid move direction: acc {np.mean(pu==(nxt[sel]>0))*100:.1f}% (n={sel.sum():,}); on the 20% most decisive: {np.mean((pu==(nxt[sel]>0))[top])*100:.1f}%")
