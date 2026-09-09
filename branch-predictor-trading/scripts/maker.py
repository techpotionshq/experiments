import glob, numpy as np, itertools
files=sorted(glob.glob('k/BTCUSDT-1s-*.csv'))
a=np.concatenate([np.loadtxt(f,delimiter=',',usecols=(1,2,3,4,5,9)) for f in files])
o,h,l,c,vol,buy=a.T; signed=buy-(vol-buy); T=len(c)
def roll(x,k): cs=np.cumsum(np.r_[0.0,x]); return np.r_[np.full(k-1,np.nan), cs[k:]-cs[:-k]]

def run(k, q, hold, exit_wait, maker_bps, taker_bps, side_short=True, strict=True):
    sig=roll(signed,k); hi=np.nanquantile(sig,1-q); lo=np.nanquantile(sig,q)
    pnl=[]; n_signal=n_fill=n_taker_exit=0; t=k
    while t<T-hold-exit_wait-2:
        s=sig[t]
        if np.isnan(s) or not (s>=hi or (side_short and s<=lo)): t+=1; continue
        d=1 if s>=hi else -1; n_signal+=1
        P=c[t]                                   # rest at the last price (touch)
        filled = (l[t+1]<P) if d==1 else (h[t+1]>P)
        if not strict: filled = (l[t+1]<=P) if d==1 else (h[t+1]>=P)
        if not filled: t+=1; continue
        n_fill+=1; entry=P; te=t+1
        # after holding, try to exit with a resting order at the then-current price for up to exit_wait seconds
        tx=te+hold; X=c[tx]; exit_px=None
        for u in range(tx+1, tx+1+exit_wait):
            ok = (h[u]>X) if d==1 else (l[u]<X)
            if not strict: ok = (h[u]>=X) if d==1 else (l[u]<=X)
            if ok: exit_px=X; fee=2*maker_bps; tdone=u; break
        if exit_px is None:
            exit_px=c[tx+exit_wait]; fee=maker_bps+taker_bps; n_taker_exit+=1; tdone=tx+exit_wait
        r=d*(exit_px/entry-1)*1e4 - fee
        pnl.append(r); t=tdone+1
    pnl=np.array(pnl)
    return dict(signals=n_signal, fills=n_fill, fill_rate=n_fill/max(n_signal,1), taker_exits=n_taker_exit/max(n_fill,1),
                gross_bps=(pnl+0).mean() if len(pnl) else np.nan, trades=len(pnl),
                total_pct=pnl.sum()/1e2 if len(pnl) else 0, win=np.mean(pnl>0) if len(pnl) else np.nan)

print("BTCUSDT spot 1s, 7 days. Enter with a resting limit order at the last price when 1s net taker flow is extreme;")
print("hold, then exit with a resting order, falling back to a market order after exit_wait seconds. Fill = price traded THROUGH the order.\n")
print(f"{'k':>3}{'q':>5}{'hold':>5}{'wait':>5}{'maker':>6}{'taker':>6} | {'signals':>8}{'fill%':>7}{'trades':>7}{'takerX%':>8}{'win%':>6}{'net/trade bps':>14}{'week total':>11}")
for k,q,hold,wait in [(1,0.10,5,10),(1,0.10,15,10),(1,0.10,60,10),(1,0.05,15,10),(5,0.10,15,10),(1,0.10,15,30),(1,0.02,60,30)]:
    for mk,tk in [(0,5),(2,5),(-1,5)]:
        r=run(k,q,hold,wait,mk,tk)
        print(f"{k:3d}{q:5.2f}{hold:5d}{wait:5d}{mk:6.0f}{tk:6.0f} | {r['signals']:8d}{r['fill_rate']*100:7.1f}{r['trades']:7d}{r['taker_exits']*100:8.1f}{r['win']*100:6.1f}{r['gross_bps']:14.3f}{r['total_pct']:10.1f}%")
    print()
print("Same, but optimistic fills (filled when price merely TOUCHES the order), maker 0 / taker 5:")
for k,q,hold,wait in [(1,0.10,5,10),(1,0.10,15,10),(1,0.10,60,10)]:
    r=run(k,q,hold,wait,0,5,strict=False)
    print(f"{k:3d}{q:5.2f}{hold:5d}{wait:5d}{0:6.0f}{5:6.0f} | {r['signals']:8d}{r['fill_rate']*100:7.1f}{r['trades']:7d}{r['taker_exits']*100:8.1f}{r['win']*100:6.1f}{r['gross_bps']:14.3f}{r['total_pct']:10.1f}%")
