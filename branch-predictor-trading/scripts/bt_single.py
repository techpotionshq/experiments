"""Single-coin time-series version of the section-11 trend algo.

The winning algo is CROSS-SECTIONAL (ranks coins against each other), so it cannot
run on one coin. This is its time-series half applied per coin:
  breakout entry (new `lookback`-day high, exit on N-day low)
  own-trend regime (only hold while close > SMA(regime), else cash)
  vol targeting (scale to target annualized vol, no leverage)
Long-flat, net of fee. Reports weekly stats and compares to buy-and-hold.

Usage: python3 scripts/bt_single.py XRPUSDT XMRUSDT --lookback 30 --regime 30 --fee 10
"""
import sys, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

FOLDERS=['k1h','kmeme','kalt','ksingle']

def load(sym, folder=None, interval='24h'):
    parts=[]; files=[]
    for fol in (FOLDERS if folder is None else [folder]):
        files+=glob.glob(f'{fol}/{sym}-1h-*.csv')
    seen=set(); files=[f for f in sorted(files) if (os.path.basename(f) not in seen and not seen.add(os.path.basename(f)))]
    for f in files:
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

def metrics(net):
    net=net.dropna()
    if len(net)<60: return None
    eq=(1+net).cumprod(); cagr=eq.iloc[-1]**(365/len(net))-1
    sharpe=net.mean()/net.std()*np.sqrt(365) if net.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    wk=((1+net).resample('W').prod()-1)*100
    return dict(cagr=cagr*100, sharpe=sharpe, maxdd=dd*100, wk_mean=wk.mean(),
                wk_pos=np.mean(wk>0)*100, wk_worst=wk.min(), n=len(net))

def all_symbols():
    syms=set()
    for fol in FOLDERS:
        for f in glob.glob(f'{fol}/*-1h-*.csv'):
            syms.add(os.path.basename(f).split('-1h-')[0])
    return sorted(syms)

def main():
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    lb=int(opt('--lookback','30')); reg=int(opt('--regime','30')); fee=float(opt('--fee','10'))
    given=[a for a in sys.argv[1:] if not a.startswith('--') and a not in
           [opt('--lookback','30'),opt('--regime','30'),opt('--fee','10')]]
    syms=given if given else all_symbols()
    rows=[]
    for sym in syms:
        cl=load(sym)
        if cl is None: continue
        net,ret=strat(cl, lb, max(5,lb//2), reg, 0.50, fee)
        m=metrics(net); h=metrics(ret)
        if m is None or h is None: continue
        rows.append((sym, m, h))
    rows.sort(key=lambda r: r[1]['sharpe'], reverse=True)
    print(f"single-coin TREND (breakout+regime+voltgt) net {fee} bps, ranked by Sharpe | lb={lb} reg={reg}")
    print(f"{'coin':<10}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'wkPos%':>8}{'wkWorst':>9}   {'vs HOLD Sharpe':>16}{'HOLD CAGR':>11}")
    for sym,m,h in rows:
        edge='  ***' if m['sharpe']>h['sharpe'] else ''
        print(f"{sym.replace('USDT',''):<10}{m['cagr']:>7.0f}%{m['sharpe']:>8.2f}{m['maxdd']:>7.0f}%"
              f"{m['wk_pos']:>7.0f}%{m['wk_worst']:>8.0f}%   {h['sharpe']:>10.2f}{'':>6}{h['cagr']:>9.0f}%{edge}")

if __name__=='__main__': main()
