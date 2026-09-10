"""Strategy search across an alt + meme universe, all net of fees.

Loads every coin in k1h/, kmeme/, kalt/, resamples to daily, and runs four
candidate families that could work on volatile alts, reporting each net of a
per-side fee plus weekly-return stats so we can see which one actually survives:

  XSMOM  cross-sectional momentum: each week, long the top-frac by trailing return
  XSREV  cross-sectional reversal: long the bottom-frac (biggest losers)
  TSMOM  time-series momentum: long every coin above its own SMA, equal weight
  BREAK  Donchian breakout: long on a new `lookback`-day high, exit on an N-day low

All long-only (no shorting alts/memes). Cross-sectional books rebalance weekly.
Fee charged on turnover. Compares to equal-weight buy-and-hold of the universe.

Usage: python3 scripts/bt_search.py --fee 10 --top 0.25 --lookback 30
"""
import sys, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

def load_all(folders, interval='24h'):
    series={}
    for folder in folders:
        for f in glob.glob(f'{folder}/*-1h-*.csv'):
            sym=os.path.basename(f).split('-1h-')[0]
            series.setdefault(sym, []).append(f)
    panel={}
    for sym, files in series.items():
        parts=[]
        for f in sorted(files):
            a=np.loadtxt(f, delimiter=',', usecols=(0,4))
            if a.ndim==1: a=a[None,:]
            parts.append(a)
        a=np.concatenate(parts); a=a[np.argsort(a[:,0])]
        t=a[:,0].astype('float64'); t=np.where(t>1e15,t/1e3,np.where(t<1e12,t*1e3,t))
        s=pd.Series(a[:,1], index=pd.to_datetime(t,unit='ms',utc=True)).sort_index()
        s=s[~s.index.duplicated(keep='last')].resample(interval).last()
        panel[sym]=s
    df=pd.DataFrame(panel).sort_index()
    return df

def perf(net, bpy=365):
    net=net.dropna()
    if len(net)<30: return None
    eq=(1+net).cumprod(); cagr=eq.iloc[-1]**(bpy/len(net))-1
    sharpe=net.mean()/net.std()*np.sqrt(bpy) if net.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    wk=((1+net).resample('W').prod()-1)
    return dict(cagr=cagr*100, sharpe=sharpe, maxdd=dd*100,
                wk_mean=wk.mean()*100, wk_med=wk.median()*100, wk_pos=np.mean(wk>0)*100,
                wk_worst=wk.min()*100, wk_best=wk.max()*100, nwk=len(wk))

def settle(W, ret, fee, regime=None, target_vol=0.0):
    """W = target weights per day. Optionally gate to cash in a market downtrend
    and scale the whole book to a target annualized vol (no leverage)."""
    if regime is not None:
        W=W.mul(regime.astype(float), axis=0)             # zero the whole book in a market downtrend
    if target_vol>0:
        gross=(W.shift(1).fillna(0.0)*ret).sum(axis=1)
        rv=gross.rolling(30).std()*np.sqrt(365)
        scale=(target_vol/rv).clip(0,1.0).shift(1).fillna(0.0)
        W=W.mul(scale, axis=0)
    Wheld=W.shift(1).fillna(0.0)
    port=(Wheld*ret).sum(axis=1)
    turn=Wheld.diff().abs().sum(axis=1).fillna(0.0)
    return port - fee*turn

def xsection(close, ret, lookback, frac, side, fee, regime=None):
    mom=close.pct_change(lookback)
    reb=pd.Series(close.index.weekday==0, index=close.index)   # weekly, Mondays
    W=pd.DataFrame(0.0, index=close.index, columns=close.columns)
    valid_all=close.notna() & mom.notna()
    for d in close.index[reb.values]:
        row=mom.loc[d][valid_all.loc[d]].dropna()
        if len(row)<8: continue
        n=max(1,int(round(len(row)*frac)))
        picks=row.sort_values(ascending=False).index[:n] if side=='mom' else row.sort_values().index[:n]
        W.loc[d, picks]=1.0/n
    Wv=W.values.copy(); Wv[~reb.values,:]=np.nan
    W=pd.DataFrame(Wv, index=W.index, columns=W.columns).ffill().fillna(0.0)
    return settle(W, ret, fee, regime)

def tsmom(close, ret, lookback, fee, regime=None):
    sig=(close>close.rolling(lookback).mean()).astype(float)
    live=close.notna().astype(float)
    w=(sig*live); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.0)
    return settle(w, ret, fee, regime)

def breakout(close, ret, lookback, exitlb, fee, regime=None, target_vol=0.0):
    hi=close.rolling(lookback).max(); lo=close.rolling(exitlb).min()
    N=len(close); cols=close.columns
    cv=close.values; hv=hi.values; lv=lo.values; st=np.zeros((N,len(cols)))
    for j in range(len(cols)):
        s=0
        for t in range(1,N):
            if np.isnan(cv[t,j]): st[t,j]=0; s=0; continue
            if s==0 and cv[t,j]>=hv[t-1,j] and not np.isnan(hv[t-1,j]): s=1
            elif s==1 and cv[t,j]<=lv[t-1,j]: s=0
            st[t,j]=s
    sig=pd.DataFrame(st, index=close.index, columns=cols)
    w=sig.div(sig.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.0)
    return settle(w, ret, fee, regime, target_vol)

def main():
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    fee=float(opt('--fee','10'))/1e4
    frac=float(opt('--top','0.25')); lb=int(opt('--lookback','30'))
    close=load_all(['k1h','kmeme','kalt'])
    ret=close.pct_change().fillna(0.0)
    print(f"universe {close.shape[1]} coins | {len(close)} daily bars "
          f"({close.index.min().date()} to {close.index.max().date()}) | fee {fee*1e4:.0f} bps/side")
    print(f"coins: {', '.join(sorted(c.replace('USDT','') for c in close.columns))}\n")
    eww=close.notna().astype(float).div(close.notna().sum(axis=1).replace(0,np.nan),axis=0)
    ewhold=(ret*eww).sum(axis=1)
    # market regime: equal-weight index above its own trailing MA
    mkt=(1+ewhold).cumprod(); reg=int(opt('--regime','30'))
    regime_up=(mkt>mkt.rolling(reg).mean()) if reg>0 else None
    strats={
        f'XSMOM top{frac:.0%} lb{lb}':        xsection(close,ret,lb,frac,'mom',fee),
        f'XSMOM + regime{reg}':               xsection(close,ret,lb,frac,'mom',fee,regime_up),
        f'BREAK {lb}d/{lb//2}d':              breakout(close,ret,lb,max(5,lb//2),fee),
        f'BREAK + regime{reg}':               breakout(close,ret,lb,max(5,lb//2),fee,regime_up),
        f'BREAK + regime + voltgt50':         breakout(close,ret,lb,max(5,lb//2),fee,regime_up,0.50),
        f'XSREV bot{frac:.0%} lb{lb}':        xsection(close,ret,lb,frac,'rev',fee),
        f'TSMOM sma{lb} + regime{reg}':       tsmom(close,ret,lb,fee,regime_up),
        'equal-weight HOLD':                  ewhold,
    }
    print(f"{'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'wk_mean':>8}{'wk_med':>7}{'wk_pos%':>8}{'wk_worst':>9}")
    for name,net in strats.items():
        p=perf(net)
        if p is None: print(f"{name:<22} (too short)"); continue
        print(f"{name:<26}{p['cagr']:>7.1f}%{p['sharpe']:>8.2f}{p['maxdd']:>7.1f}%"
              f"{p['wk_mean']:>7.2f}%{p['wk_med']:>6.2f}%{p['wk_pos']:>7.0f}%{p['wk_worst']:>8.1f}%")
    # per-year for the risk-managed breakout winner and for hold
    win=breakout(close,ret,lb,max(5,lb//2),fee,regime_up,0.50)
    print("\n  per-year net % (BREAK+regime+voltgt50 / HOLD):")
    for yr,g in win.groupby(win.index.year):
        h=ewhold[ewhold.index.year==yr]
        print(f"    {yr}:  {(( 1+g).prod()-1)*100:>7.1f}%  /  {((1+h).prod()-1)*100:>7.1f}%")

if __name__=='__main__': main()
