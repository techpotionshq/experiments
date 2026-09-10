import json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
def load(t):
    d=json.load(open(f'stocks/{t}.json'))['chart']['result'][0]
    ts=pd.to_datetime(d['timestamp'],unit='s',utc=True).normalize(); q=d['indicators']['quote'][0]
    return pd.Series(q['close'],index=ts), pd.Series(q['volume'],index=ts)
tickers=[os.path.basename(f)[:-5] for f in glob.glob('stocks/*.json') if 'SPY' not in f]
HOLD=10; fee=2/1e4
def trades(cond_sign):
    out=[]
    for t in tickers:
        c,v=load(t); r=c.pct_change(); vol=r.rolling(20).std(); avgv=v.rolling(20).mean()
        z=r/vol; vs=v/avgv
        sig = ((z< -2)&(vs>2)) if cond_sign<0 else ((z>2)&(vs>2))
        days=np.where(sig.values)[0]
        for i in days:
            if i+HOLD<len(c):
                out.append(c.values[i+HOLD]/c.values[i]-1 - 2*fee)
    return np.array(out)
for label,s in [("BUY the down-spike (fade bad-news drop)",-1),("BUY the up-spike (chase good news)",1)]:
    tr=trades(s)
    print(f"{label}: {len(tr)} trades, hold {HOLD}d")
    print(f"   avg {tr.mean()*100:+.2f}% / trade   median {np.median(tr)*100:+.2f}%   win {np.mean(tr>0)*100:.0f}%   "
          f"best {tr.max()*100:+.0f}%  worst {tr.min()*100:+.0f}%\n")
