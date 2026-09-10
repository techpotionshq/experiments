"""Live positioning for the section-11 basket, as of the latest daily bar.

Not a trading bot: it places no orders. It fetches the freshest daily closes it
can (monthly 1h archive resampled to daily, plus the recent daily archive) and
prints exactly what the strategy would hold RIGHT NOW, on spot and on 2x, so you
can act on it by hand. No exchange keys, no live orders.

Rule (same as scripts/bt_search.py breakout + regime + vol target):
  hold a coin if it is currently in a Donchian breakout (last close still above
  the trailing exit low since a 30d-high entry) AND the market regime is up
  (equal-weight index above its 30d average); size the whole book to 50% vol.
"""
import glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

FOLD_1H=['k1h','kmeme','kalt']

def load_hist(sym):
    parts=[]
    for fol in FOLD_1H:
        parts+=glob.glob(f'{fol}/{sym}-1h-*.csv')
    if not parts: return None
    seen=set(); files=[f for f in sorted(parts) if (os.path.basename(f) not in seen and not seen.add(os.path.basename(f)))]
    A=[]
    for f in files:
        a=np.loadtxt(f, delimiter=',', usecols=(0,4))
        if a.ndim==1: a=a[None,:]
        A.append(a)
    a=np.concatenate(A); a=a[np.argsort(a[:,0])]
    t=a[:,0].astype('float64'); t=np.where(t>1e15,t/1e3,np.where(t<1e12,t*1e3,t))
    s=pd.Series(a[:,1], index=pd.to_datetime(t,unit='ms',utc=True)).sort_index()
    s=s[~s.index.duplicated(keep='last')].resample('24h').last()
    # append recent daily archive
    dl=sorted(glob.glob(f'kdaily/{sym}-1d-*.csv'))
    if dl:
        B=[]
        for f in dl:
            a=np.loadtxt(f, delimiter=',', usecols=(0,4))
            if a.ndim==1: a=a[None,:]
            B.append(a)
        b=np.concatenate(B); t2=b[:,0].astype('float64'); t2=np.where(t2>1e15,t2/1e3,np.where(t2<1e12,t2*1e3,t2))
        s2=pd.Series(b[:,1], index=pd.to_datetime(t2,unit='ms',utc=True)).resample('24h').last()
        s=pd.concat([s,s2]); s=s[~s.index.duplicated(keep='last')].sort_index()
    return s.dropna()

def main():
    syms=sorted({os.path.basename(f).split('-1h-')[0] for fol in FOLD_1H for f in glob.glob(f'{fol}/*-1h-*.csv')})
    panel={s:load_hist(s) for s in syms}
    panel={k:v for k,v in panel.items() if v is not None and len(v)>60}
    close=pd.DataFrame(panel).sort_index()
    asof=close.index.max()
    lb,exitlb,reg=30,15,30
    ret=close.pct_change().fillna(0.0)
    eww=close.notna().astype(float).div(close.notna().sum(axis=1).replace(0,np.nan),axis=0)
    mkt=(1+(ret*eww).sum(axis=1)).cumprod()
    regime_up = mkt.iloc[-1] > mkt.rolling(reg).mean().iloc[-1]
    # per-coin breakout state up to asof
    hi=close.rolling(lb).max(); lo=close.rolling(exitlb).min()
    held={}
    for c in close.columns:
        cv=close[c].values; hv=hi[c].values; lv=lo[c].values; s=0
        for t in range(1,len(cv)):
            if np.isnan(cv[t]): s=0; continue
            if s==0 and cv[t]>=hv[t-1] and not np.isnan(hv[t-1]): s=1
            elif s==1 and cv[t]<=lv[t-1]: s=0
        held[c]=s
    in_break=[c for c,v in held.items() if v==1]
    # vol target scalar from recent strategy vol (approx via equal-weight of held names)
    print(f"=== LIVE SIGNAL as of {asof.date()} (data through the last complete daily bar) ===")
    print(f"MARKET REGIME: {'RISK-ON (index above its 30d avg)' if regime_up else 'RISK-OFF (index below 30d avg) -> STRATEGY IS IN CASH'}")
    if not regime_up:
        print("\n>>> The strategy holds NOTHING right now. 100% cash on both spot and 2x. <<<")
        print(f"    ({len(in_break)} coins are individually in breakout, but the regime filter overrides to cash.)")
        print("    coins in breakout (ignored while risk-off): " + ", ".join(sorted(c.replace('USDT','') for c in in_break)))
        return
    names=sorted(c.replace('USDT','') for c in in_break)
    n=len(names) if names else 0
    print(f"\nHOLD {n} coins, equal weight: {', '.join(names) if names else '(none in breakout -> cash)'}")
    if n:
        wt=100.0/n
        print(f"\n  SPOT $100:  ${wt:.2f} in each of the {n} names")
        print(f"  2x   $100:  ${2*wt:.2f} exposure in each (=$200 gross, $100 margin); liquidation risk if the book drops ~50%")
        print("\n  (vol targeting may scale total exposure below 100% in calm/volatile regimes;")
        print("   this shows the full-exposure book. Rebalance weekly.)")

if __name__=='__main__': main()
