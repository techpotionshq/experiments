"""Single-coin time-series version of the section-11 trend algo.

The winning algo is CROSS-SECTIONAL (ranks coins against each other), so it cannot
run on one coin. This is its time-series half applied per coin:
  breakout entry (new `lookback`-day high, exit on N-day low)
  own-trend regime (only hold while close > SMA(regime), else cash)
  vol targeting (scale to target annualized vol, no leverage)
Long-flat, net of fee. Reports weekly stats and compares to buy-and-hold.

Usage: python3 scripts/bt_single.py XRPUSDT XMRUSDT --lookback 30 --regime 30 --fee 10
"""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

def load(sym, folder='ksingle', interval='24h'):
    parts=[]
    for f in sorted(glob.glob(f'{folder}/{sym}-1h-*.csv')):
        a=np.loadtxt(f, delimiter=',', usecols=(0,4))
        if a.ndim==1: a=a[None,:]
        parts.append(a)
    if not parts: return None
    a=np.concatenate(parts); a=a[np.argsort(a[:,0])]
    t=a[:,0].astype('float64'); t=np.where(t>1e15,t/1e3,np.where(t<1e12,t*1e3,t))
    s=pd.Series(a[:,1], index=pd.to_datetime(t,unit='ms',utc=True)).sort_index()
    return s[~s.index.duplicated(keep='last')].resample(interval).last().dropna()

def strat(close, lookback, exitlb, regime, target_vol, fee_bps):
    fee=fee_bps/1e4; ret=close.pct_change().fillna(0.0)
    hi=close.rolling(lookback).max(); lo=close.rolling(exitlb).min()
    up=(close>close.rolling(regime).mean()).values
    c=close.values; hv=hi.values; lv=lo.values; N=len(c); pos=np.zeros(N); s=0
    for t in range(1,N):
        if s==0 and c[t]>=hv[t-1] and not np.isnan(hv[t-1]) and up[t]: s=1
        elif s==1 and (c[t]<=lv[t-1] or not up[t]): s=0
        pos[t]=s
    pos=pd.Series(pos, index=close.index)
    if target_vol>0:
        rv=(pos.shift(1)*ret).rolling(30).std()*np.sqrt(365)
        pos=pos*(target_vol/rv).clip(0,1.0).fillna(0.0)
    posh=pos.shift(1).fillna(0.0)
    net=posh*ret - fee*posh.diff().abs().fillna(posh.abs())
    return net, ret

def perf(net, label):
    net=net.dropna()
    eq=(1+net).cumprod(); cagr=eq.iloc[-1]**(365/len(net))-1
    sharpe=net.mean()/net.std()*np.sqrt(365) if net.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    wk=((1+net).resample('W').prod()-1)*100
    print(f"  {label:<16} CAGR {cagr*100:>7.1f}%  Sharpe {sharpe:>5.2f}  maxDD {dd*100:>6.1f}%  "
          f"| wk mean {wk.mean():+.2f}% med {wk.median():+.2f}% pos {np.mean(wk>0)*100:.0f}% worst {wk.min():.1f}%")

def main():
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    lb=int(opt('--lookback','30')); reg=int(opt('--regime','30')); fee=float(opt('--fee','10'))
    syms=[a for a in sys.argv[1:] if not a.startswith('--') and a not in
          [opt('--lookback','30'),opt('--regime','30'),opt('--fee','10')]]
    for sym in syms:
        cl=load(sym)
        if cl is None: print(f"{sym}: no data"); continue
        net,ret=strat(cl, lb, max(5,lb//2), reg, 0.50, fee)
        print(f"\n{sym}  ({cl.index.min().date()} to {cl.index.max().date()}, {len(cl)} days, fee {fee} bps)")
        perf(net, 'TREND net')
        perf(ret, 'buy & hold')

if __name__=='__main__': main()
