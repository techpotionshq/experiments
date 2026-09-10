import json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

def load(t):
    d=json.load(open(f'stocks/{t}.json'))['chart']['result'][0]
    ts=pd.to_datetime(d['timestamp'], unit='s', utc=True).normalize()
    return pd.Series(d['indicators']['quote'][0]['close'], index=ts).dropna()

tickers=[os.path.basename(f)[:-5] for f in glob.glob('stocks/*.json')]
panel={t:load(t) for t in tickers}; spy=panel.pop('SPY')
close=pd.DataFrame(panel).sort_index().ffill()
ret=close.pct_change(); BPY=252; fee=2/1e4

def perf(net, label):
    net=net.dropna(); eq=(1+net).cumprod()
    cagr=eq.iloc[-1]**(BPY/len(net))-1; sh=net.mean()/net.std()*np.sqrt(BPY) if net.std()>0 else 0
    print(f"  {label:<34} CAGR {cagr*100:>6.1f}%  Sharpe {sh:>5.2f}")

print(f"US stocks daily, {close.index.min().date()} to {close.index.max().date()}, {close.shape[1]} names, {fee*1e4:.0f}bps\n")
print("=== 1-BIT predictor (go the way the last day went), equal-weight basket ===")
sig=np.sign(ret.shift(1))                      # yesterday's direction
for name, s in [("1-bit MOMENTUM long/flat", (sig>0).astype(float)),
                ("1-bit MOMENTUM long/short", sig),
                ("1-bit REVERSAL long/flat (buy yesterday's losers)", (sig<0).astype(float))]:
    w=s.div(s.abs().sum(1).replace(0,np.nan),axis=0).fillna(0.0)
    turn=w.diff().abs().sum(1).fillna(0.0)
    net=(w*ret).sum(1)-fee*turn
    perf(net, name)

# directional-accuracy of yesterday->today (pooled)
y=(ret>0).astype(int); prevup=(ret.shift(1)>0)
acc=(y.values[1:]==prevup.values[1:])
print(f"\n  yesterday's direction predicts today: {np.nanmean(acc)*100:.1f}% (50% = coin flip)")

print("\n=== LOGISTIC REGRESSION (5 lagged daily returns -> next-day up), pooled across stocks ===")
X=[]; Y=[]; idx=[]
for c in close.columns:
    r=ret[c].dropna()
    for k in range(5, len(r)-1):
        X.append(r.values[k-5:k]); Y.append(1 if r.values[k+1]>0 else 0); idx.append(r.index[k])
X=np.array(X); Y=np.array(Y); idx=pd.to_datetime(idx)
order=np.argsort(idx.values); X,Y=X[order],Y[order]
cut=int(len(X)*0.6)
lr=LogisticRegression(max_iter=500).fit(X[:cut],Y[:cut])
p=lr.predict_proba(X[cut:])[:,1]; yte=Y[cut:]
conf=np.abs(p-0.5); top=conf>=np.quantile(conf,0.8)
print(f"  overall next-day accuracy: {np.mean((p>0.5)==(yte==1))*100:.1f}%")
print(f"  on the 20% most confident: {np.mean((p[top]>0.5)==(yte[top]==1))*100:.1f}%")
print(f"  base rate (share up days) : {np.mean(yte==1)*100:.1f}%")
