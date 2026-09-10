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
# benchmark: SPY weekly
spy=load('SPY') if False else None
