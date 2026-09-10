import json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

def load(t):
    d=json.load(open(f'stocks/{t}.json'))['chart']['result'][0]
    ts=pd.to_datetime(d['timestamp'], unit='s', utc=True).normalize()
    cl=d['indicators']['quote'][0]['close']
    return pd.Series(cl, index=ts).dropna()

tickers=[os.path.basename(f)[:-5] for f in glob.glob('stocks/*.json')]
panel={t:load(t) for t in tickers}
spy=panel.pop('SPY')
close=pd.DataFrame(panel).sort_index().ffill()
ret=close.pct_change().fillna(0.0); BPY=252; fee=2/1e4   # ~commission-free, 2bps slippage

eww=close.notna().astype(float).div(close.notna().sum(1).replace(0,np.nan),axis=0)
mkt=(1+(ret*eww).sum(1)).cumprod(); regime=mkt>mkt.rolling(30).mean()

def perf(net,label):
    net=net.dropna(); eq=(1+net).cumprod()
    cagr=eq.iloc[-1]**(BPY/len(net))-1; sh=net.mean()/net.std()*np.sqrt(BPY) if net.std()>0 else 0
    dd=(eq/eq.cummax()-1).min(); wk=((1+net).resample('W').prod()-1)
    print(f"  {label:<26} CAGR {cagr*100:>6.1f}%  Sharpe {sh:>4.2f}  maxDD {dd*100:>6.1f}%  wkPos {np.mean(wk>0)*100:>3.0f}%")

def topk(K, use_regime):
    mom=close.pct_change(30); reb=(close.index.weekday==0)
    W=pd.DataFrame(0.0,index=close.index,columns=close.columns); val=close.notna()&mom.notna()
    for d in close.index[reb]:
        row=mom.loc[d][val.loc[d]].dropna()
        if len(row)<3: continue
        W.loc[d, row.sort_values(ascending=False).index[:K]]=1.0/K
    Wv=W.values.copy(); Wv[~reb,:]=np.nan
    W=pd.DataFrame(Wv,index=W.index,columns=W.columns).ffill().fillna(0.0)
    if use_regime: W=W.mul(regime.astype(float),axis=0)
    Wh=W.shift(1).fillna(0.0)
    return (Wh*ret).sum(1) - fee*Wh.diff().abs().sum(1).fillna(0.0)

print(f"US stocks, {close.index.min().date()} to {close.index.max().date()}, {close.shape[1]} names, ~2bps fee\n")
perf(topk(5,True),  'top-5 mom + regime')
perf(topk(5,False), 'top-5 mom (no regime)')
perf(topk(3,True),  'top-3 mom + regime')
perf((ret*eww).sum(1), 'equal-weight all 30')
perf(spy.pct_change().fillna(0), 'SPY buy & hold')
