"""Deviation harvesting: short-term mean reversion on high-vol meme coins.

Meme coins swing hard, so a dip can bounce several percent, maybe enough to clear
the fee even trading often (the point of section 10 was the opposite regime).

Rule, per coin, long-flat (never short a meme, squeezes are lethal):
  z = (close - SMA(lookback)) / std(lookback)
  flat and z <= -entry_z  -> BUY next bar (price is entry_z sigmas below its mean)
  in  and (z >= exit_z, or held >= maxhold bars, or price <= stop) -> SELL next bar
Basket of memes, each an independent 1/N sleeve, averaged over coins live that bar.
Fee `fee_bps` per side on every position change.

HONEST WARNING baked in: this uses only memes still listed on Binance. Coins that
dipped and never came back (delisted, went to zero) are missing, and "buy the dip"
is exactly the rule they would have killed. So these numbers are survivorship-
biased UPWARD. The stop-loss and the reported worst-coin help but do not remove it.

Usage: python3 scripts/bt_meanrev.py --interval 1H --lookback 48 --entry 2.0 \
          --exit 0.0 --maxhold 48 --stop 0.15 --fee 10
"""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

MEMES=['DOGEUSDT','SHIBUSDT','PEPEUSDT','WIFUSDT','BONKUSDT','FLOKIUSDT',
       '1000SATSUSDT','BOMEUSDT','MEMEUSDT','NEIROUSDT','PNUTUSDT']

def load_coin(sym, interval, folder):
    files=sorted(glob.glob(f'{folder}/{sym}-1h-*.csv'))
    if not files: return None
    parts=[]
    for f in files:
        a=np.loadtxt(f, delimiter=',', usecols=(0,4))
        if a.ndim==1: a=a[None,:]
        parts.append(a)
    a=np.concatenate(parts); a=a[np.argsort(a[:,0])]
    t=a[:,0].astype('float64')
    t=np.where(t>1e15, t/1e3, np.where(t<1e12, t*1e3, t))
    ts=pd.to_datetime(t, unit='ms', utc=True)
    s=pd.Series(a[:,1], index=ts).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    return s.resample(interval.lower()).last().dropna()

def bars_per_year(interval):
    hours={'1H':1,'2H':2,'4H':4,'6H':6,'8H':8,'12H':12,'24H':24}.get(interval.upper(),1)
    return 365*24/hours

def coin_pnl(close, lookback, entry_z, exit_z, maxhold, stop, fee_bps, trend=0):
    fee=fee_bps/1e4
    ret=close.pct_change().fillna(0.0)
    ma=close.rolling(lookback).mean(); sd=close.rolling(lookback).std()
    z=((close-ma)/sd).values
    up=(close>close.rolling(trend).mean()).values if trend>0 else np.ones(len(close),bool)
    c=close.values; N=len(c)
    pos=np.zeros(N); held=0; entry_px=0.0
    # state machine
    state=0
    for t in range(1,N):
        if state==1:
            held+=1
            hit_stop = stop>0 and c[t]<=entry_px*(1-stop)
            if z[t]>=exit_z or held>=maxhold or hit_stop:
                state=0
        else:
            if not np.isnan(z[t]) and z[t]<=-entry_z and up[t]:
                state=1; held=0; entry_px=c[t]
        pos[t]=state
    pos=pd.Series(pos, index=close.index).shift(1).fillna(0.0)   # trade next bar
    turn=pos.diff().abs().fillna(pos.abs())
    strat=pos*ret - fee*turn
    return pd.DataFrame({'strat':strat,'hold':ret,'pos':pos,'turn':turn})

def wstats(r, bpy):
    r=r.dropna()
    if len(r)<10: return None
    eq=(1+r).cumprod(); cagr=eq.iloc[-1]**(bpy/len(r))-1
    sharpe=r.mean()/r.std()*np.sqrt(bpy) if r.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    return dict(cagr=cagr*100, sharpe=sharpe, maxdd=dd*100)

def run(interval, lookback, entry_z, exit_z, maxhold, stop, fee_bps, folder, trend=0):
    bpy=bars_per_year(interval)
    S=[]; H=[]; T=[]; per_coin={}
    for c in MEMES:
        cl=load_coin(c, interval, folder)
        if cl is None or len(cl)<lookback+50: continue
        df=coin_pnl(cl, lookback, entry_z, exit_z, maxhold, stop, fee_bps, trend)
        S.append(df['strat'].rename(c)); H.append(df['hold'].rename(c)); T.append(df['turn'].rename(c))
        st=wstats(df['strat'], bpy)
        per_coin[c]=(st['cagr'] if st else float('nan'), df['turn'].sum()/2)
    if not S: print("no data"); return
    strat=pd.concat(S,axis=1); hold=pd.concat(H,axis=1); turn=pd.concat(T,axis=1)
    w=(~strat.isna()).astype(float); w=w.div(w.sum(axis=1).replace(0,np.nan),axis=0)
    port=(strat*w).sum(axis=1); porthold=(hold*w).sum(axis=1); portturn=(turn*w).sum(axis=1)
    idx=port.index
    print(f"\n=== meme mean-reversion basket | {interval} | lookback={lookback} entry={entry_z}s "
          f"exit={exit_z}s maxhold={maxhold} stop={stop:.0%} trend={trend} | fee {fee_bps} bps/side ===")
    print(f"coins {len(S)} | {len(idx)} bars ({idx.min().date()} to {idx.max().date()}) | "
          f"round trips/coin avg ~{np.nanmean([v[1] for v in per_coin.values()]):.0f}")
    st=wstats(port,bpy); hd=wstats(porthold,bpy)
    fee_drag=(fee_bps/1e4*portturn).sum()/len(idx)*bpy*100
    print(f"STRAT net: CAGR {st['cagr']:+.1f}%  Sharpe {st['sharpe']:.2f}  maxDD {st['maxdd']:.1f}%  "
          f"| basket HOLD: CAGR {hd['cagr']:+.1f}%  maxDD {hd['maxdd']:.1f}%  | fee drag ~{fee_drag:.0f}%/yr")
    # weekly
    wk=((1+port).resample('W').prod()-1)*100; wk=wk[wk.index<=idx.max()]
    print(f"  --- WEEKLY (net), {len(wk)} weeks ---")
    print(f"    mean {wk.mean():+.2f}%/wk | median {wk.median():+.2f}% | std {wk.std():.2f}% | "
          f"positive {np.mean(wk>0)*100:.0f}% | best {wk.max():+.1f}% | worst {wk.min():+.1f}%")
    print(f"    compounded {(((1+wk/100).prod())**(1/len(wk))-1)*100:+.2f}%/wk avg")
    # per year
    print("  per-year net %:", {int(y): round(((1+g).prod()-1)*100,1) for y,g in port.groupby(port.index.year)})
    print("  per-coin standalone net CAGR %:", {k.replace('USDT',''): round(v[0],0) for k,v in per_coin.items()})

if __name__=='__main__':
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    run(opt('--interval','1H'), int(opt('--lookback','48')), float(opt('--entry','2.0')),
        float(opt('--exit','0.0')), int(opt('--maxhold','48')), float(opt('--stop','0.15')),
        float(opt('--fee','10')), opt('--folder','kmeme'), int(opt('--trend','0')))
