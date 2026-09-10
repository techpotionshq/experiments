import json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
def load(t):
    d=json.load(open(f'stocks/{t}.json'))['chart']['result'][0]
    ts=pd.to_datetime(d['timestamp'],unit='s',utc=True).normalize()
    return pd.Series(d['indicators']['quote'][0]['close'],index=ts)
tk=[os.path.basename(f)[:-5] for f in glob.glob('stocks/*.json') if 'SPY' not in f]
close=pd.DataFrame({t:load(t) for t in tk}).sort_index().ffill()
wk=close.resample('W-FRI').last()              # weekly closes
wret=wk.pct_change()                            # weekly returns
fee=2/1e4

def strat(side,K):
    prev=wret.shift(1)                           # last week's return = the signal
    W=pd.DataFrame(0.0,index=wret.index,columns=wret.columns)
    for d in wret.index:
        row=prev.loc[d].dropna()
        if len(row)<K: continue
        picks=row.nsmallest(K).index if side=='loser' else row.nlargest(K).index
        W.loc[d,picks]=1.0/K
    turn=W.diff().abs().sum(1).fillna(1.0)
    net=(W*wret).sum(1)-fee*turn                 # hold one week
    net=net.dropna()
    eq=(1+net).cumprod(); cagr=eq.iloc[-1]**(52/len(net))-1
    sh=net.mean()/net.std()*np.sqrt(52) if net.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    return net,dict(cagr=cagr*100,sh=sh,dd=dd*100,pos=np.mean(net>0)*100,
                    mean=net.mean()*100,worst=net.min()*100,best=net.max()*100)

print(f"WEEKLY flip on 30 stocks, {wk.index.min().date()} to {wk.index.max().date()}, {len(wret)} weeks, 2bps\n")
print(f"{'strategy':<28}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}{'wk+%':>7}{'avg wk':>8}{'worst wk':>9}")
for side,lab in [('loser','buy last-week LOSERS (reversal)'),('winner','buy last-week WINNERS (momentum)')]:
    for K in [3,5]:
        net,s=strat(side,K)
        print(f"{lab[:22]+' K'+str(K):<28}{s['cagr']:>6.0f}%{s['sh']:>8.2f}{s['dd']:>7.0f}%{s['pos']:>6.0f}%{s['mean']:>7.2f}%{s['worst']:>8.1f}%")
# ---- live picks as of the latest bar (informational, no orders placed) ----
asof=close.index.max().date()
dret=close.pct_change().fillna(0.0)
eww=close.notna().astype(float).div(close.notna().sum(1).replace(0,np.nan),axis=0)
mkt=(1+(dret*eww).sum(1)).cumprod()                       # equal-weight index
risk_on=mkt.iloc[-1]>mkt.rolling(30).mean().iloc[-1]      # regime = index above its 30d avg
print(f"\nas of {asof} | market regime: {'RISK-ON' if risk_on else 'RISK-OFF'}")
mom3=close.pct_change(63).iloc[-1].dropna().sort_values(ascending=False)   # ~3-month momentum
print("\nBEST ALGO pick now -> top 5 by 3-month momentum (hold, re-rank weekly):")
for t,v in mom3.head(5).items():
    print(f"   {t:<6} 3mo {v*100:+.0f}%")
lastwk=wret.iloc[-1].dropna().sort_values(ascending=False)                 # last completed week
print("\nWEEKLY FLIP pick now -> top 5 by last-week return (rotate every Friday):")
for t,v in lastwk.head(5).items():
    print(f"   {t:<6} last wk {v*100:+.1f}%")
