"""The fee-robust strategy: vol-targeted trend-following on a crypto basket.

Everything we learned pointed here. Order-book signals are worth ~4-8 bps and a
round trip costs 20 bps at 10 bps/side, so no fast strategy clears the fee. The
one edge that survives is slow time-series momentum: trade a few times a month,
capture moves of hundreds of bps, and the fee becomes noise.

Design (all long-flat, no shorting, which the repo proved is a disaster):
  1. Resample 1h klines to a slow bar (default 12h).
  2. Trend filter: only allow a long when close > SMA(slow). Sidesteps bear markets.
  3. Entry momentum: close > SMA(fast). Confirmation: the on/off state only flips
     after the new signal has held `confirm` bars (the branch-predictor 2-bit idea),
     which cuts whipsaw turnover and therefore fees.
  4. Vol targeting: size = clip(target_vol / realized_vol, 0, 1). No leverage.
  5. Basket: equal weight across coins that are live, rebalanced each bar.
  6. Fee `fee_bps` per side charged on |position change| every bar.

Usage: python3 scripts/bt_trend.py --interval 12H --slow 30 --fast 8 --confirm 3 \
          --targetvol 0.60 --fee 10 --split 2024-01-01
"""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

COINS=['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT','BNBUSDT','XRPUSDT']

def load_coin(sym, interval):
    files=sorted(glob.glob(f'k1h/{sym}-1h-*.csv'))
    if not files: return None
    parts=[]
    for f in files:
        a=np.loadtxt(f, delimiter=',', usecols=(0,4))   # open_time(ms), close
        if a.ndim==1: a=a[None,:]
        parts.append(a)
    a=np.concatenate(parts); a=a[np.argsort(a[:,0])]
    t=a[:,0].astype('float64')                          # Binance switched ms->microseconds in 2025
    t=np.where(t>1e15, t/1e3, np.where(t<1e12, t*1e3, t))   # normalize everything to ms
    ts=pd.to_datetime(t, unit='ms', utc=True)
    s=pd.Series(a[:,1], index=ts).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    close=s.resample(interval.lower()).last().dropna()
    return close

def bars_per_year(interval):
    hours={'1H':1,'4H':4,'6H':6,'8H':8,'12H':12,'24H':24,'1D':24}.get(interval.upper(),12)
    return 365*24/hours

def confirmed(raw, k):
    """Flip the held state only after the raw signal has agreed for k bars."""
    out=np.zeros(len(raw)); state=0; run=0; prev=np.nan
    r=raw.values
    for i in range(len(r)):
        v=r[i]
        if np.isnan(v): out[i]=state; continue
        if v==prev: run+=1
        else: run=1; prev=v
        if run>=k: state=v
        out[i]=state
    return pd.Series(out, index=raw.index)

def coin_pnl(close, interval, slow, fast, confirm, target_vol, fee_bps):
    bpy=bars_per_year(interval); fee=fee_bps/1e4
    ret=close.pct_change().fillna(0.0)
    sma_s=close.rolling(slow).mean(); sma_f=close.rolling(fast).mean()
    raw=((close>sma_s)&(close>sma_f)).astype(float)     # 1 = want long, 0 = flat
    onoff=confirmed(raw, confirm)
    rv=ret.rolling(max(fast,10)).std()*np.sqrt(bpy)     # annualized realized vol
    size=(target_vol/rv).clip(0,1.0).fillna(0.0)
    pos=(onoff*size)
    pos=pos.shift(1).fillna(0.0)                         # trade on next bar, no lookahead
    turn=pos.diff().abs().fillna(pos.abs())
    strat=pos*ret - fee*turn
    hold=ret
    return pd.DataFrame({'strat':strat,'hold':hold,'pos':pos,'turn':turn})

def stats(r, bpy):
    r=r.dropna()
    if len(r)<10: return None
    eq=(1+r).cumprod(); cagr=eq.iloc[-1]**(bpy/len(r))-1
    vol=r.std()*np.sqrt(bpy); sharpe=r.mean()/r.std()*np.sqrt(bpy) if r.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    return dict(cagr=cagr*100, vol=vol*100, sharpe=sharpe, maxdd=dd*100, n=len(r))

def run(interval, slow, fast, confirm, target_vol, fee_bps, split):
    bpy=bars_per_year(interval)
    S=[]; H=[]; T=[]
    live={}
    for c in COINS:
        cl=load_coin(c, interval)
        if cl is None: continue
        df=coin_pnl(cl, interval, slow, fast, confirm, target_vol, fee_bps)
        S.append(df['strat'].rename(c)); H.append(df['hold'].rename(c)); T.append(df['turn'].rename(c))
        live[c]=(cl.index.min(), cl.index.max())
    strat=pd.concat(S, axis=1); hold=pd.concat(H, axis=1); turn=pd.concat(T, axis=1)
    # equal weight across coins live at each bar
    w=(~strat.isna()).astype(float); w=w.div(w.sum(1).replace(0,np.nan), axis=0)
    port=(strat*w).sum(1); porthold=(hold*w).sum(1)
    portturn=(turn*w).sum(1)
    idx=port.index
    print(f"\n=== vol-targeted trend basket | {interval} bars | slow={slow} fast={fast} "
          f"confirm={confirm} targetVol={target_vol:.0%} | fee {fee_bps} bps/side ===")
    print(f"coins: {', '.join([c.replace('USDT','') for c in live])} | {len(idx)} bars "
          f"({idx.min().date()} to {idx.max().date()})")
    fee_drag=(fee_bps/1e4*portturn).sum()/len(idx)*bpy*100
    print(f"avg turnover/bar {portturn.mean():.3f} -> ~{portturn.mean()*bpy:.0f} unit-trades/yr "
          f"| fee drag ~{fee_drag:.1f}%/yr")
    def block(mask,label):
        st=stats(port[mask], bpy); hd=stats(porthold[mask], bpy)
        if st is None or hd is None: return
        print(f"  {label:<18} STRAT net: CAGR {st['cagr']:>7.1f}%  Sharpe {st['sharpe']:>4.2f}  maxDD {st['maxdd']:>6.1f}%   "
              f"| HOLD: CAGR {hd['cagr']:>7.1f}%  Sharpe {hd['sharpe']:>4.2f}  maxDD {hd['maxdd']:>6.1f}%")
    block(pd.Series(True,index=idx), 'full sample')
    if split:
        sp=pd.Timestamp(split, tz='UTC')
        block(idx<sp, f'in-samp <{split}')
        block(idx>=sp, f'OUT-samp >={split}')
    # per year
    print("  per-year net (STRAT / HOLD):")
    for yr,g in port.groupby(port.index.year):
        h=porthold[porthold.index.year==yr]
        se=(1+g).prod()-1; he=(1+h).prod()-1
        print(f"    {yr}:  {se*100:>7.1f}%  /  {he*100:>7.1f}%")

if __name__=='__main__':
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    run(opt('--interval','24H'), int(opt('--slow','30')), int(opt('--fast','8')),
        int(opt('--confirm','2')), float(opt('--targetvol','0.60')),
        float(opt('--fee','10')), opt('--split','2024-01-01'))
