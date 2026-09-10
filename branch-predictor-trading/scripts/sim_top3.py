import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import bt_search as X, bt_topk as T

CAP=100.0; K=3
close=T.build_panel(); ret=close.pct_change().fillna(0.0)
eww=close.notna().astype(float).div(close.notna().sum(axis=1).replace(0,np.nan),axis=0)
mkt=(1+(ret*eww).sum(axis=1)).cumprod(); regime=mkt>mkt.rolling(30).mean()
W=T.topk_weights(close,30,K)
net=X.settle(W, ret, 10/1e4, regime, 0.50).dropna()

eq=CAP*(1+net).cumprod()
peak=eq.cummax(); dd=eq/peak-1
wk=(1+net).resample('W').prod()-1; wk_eq=CAP*(1+net).resample('W').prod().cumprod()  # not used
print(f"=== TOP-3 SIMULATION, start $100, net of 10 bps, {net.index.min().date()} to {net.index.max().date()} ===")
print(f"final balance : ${eq.iloc[-1]:,.0f}   ({eq.iloc[-1]/CAP:.1f}x)   CAGR {(eq.iloc[-1]/CAP)**(365/len(net))-1:.0%}")
print(f"max drawdown  : {dd.min():.0%}   (at the worst, $100 of peak was worth ${100*(1+dd.min()):.0f})")
print(f"in-market     : {np.mean(net!=0)*100:.0f}% of days (regime filter sits in cash the rest)")
print("\n-- balance at each year end (start-of-year $ -> end) --")
yr_eq=eq.resample('YE').last()
start=CAP
for ts,v in yr_eq.items():
    y=ts.year; g=eq[eq.index.year==y]
    yr_ret=(1+net[net.index.year==y]).prod()-1
    print(f"  {y}:  ${start:>10,.0f}  ->  ${v:>12,.0f}   ({yr_ret:+.0%})")
    start=v
w=wk*CAP  # weekly $ pnl scaled to a *flat* $100 (comparable week size)
print("\n-- weekly returns (as % moves, comparable across the run) --")
print(f"  average week {wk.mean()*100:+.2f}%  median {wk.median()*100:+.2f}%  positive {np.mean(wk>0)*100:.0f}%  best {wk.max()*100:+.0f}%  worst {wk.min()*100:+.0f}%")
print("\n-- last 10 weeks (week ending / weekly % / running balance) --")
we=(1+net).resample('W').prod(); bal=CAP*we.cumprod()
for d in wk.index[-10:]:
    print(f"  {d.date()}   {wk[d]*100:+7.2f}%    ${bal[d]:,.0f}")
