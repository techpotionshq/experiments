import numpy as np
import glob
def closes(p): return np.concatenate([np.loadtxt(f, delimiter=",", usecols=(4,)) for f in sorted(glob.glob(p))])
def stats(px, step_sec, fee):
    ret=np.r_[0.0, px[1:]/px[:-1]-1]; d=np.sign(np.r_[0.0,np.diff(px)])
    pos=np.zeros(len(px)); c=0
    for t in range(1,len(px)):
        if d[t-1]>0: c=1
        elif d[t-1]<0: c=0
        pos[t]=c
    turn=np.abs(np.diff(np.r_[0.0,pos])); yrs=len(px)*step_sec/31_557_600
    eq=np.cumprod(1+pos*ret-turn*fee/1e4); dd=(eq/np.maximum.accumulate(eq)-1).min()
    bh=px/px[0]; bdd=(bh/np.maximum.accumulate(bh)-1).min()
    per=ret*pos-turn*fee/1e4; sh=per.mean()/per.std()*np.sqrt(31_557_600/step_sec)
    bsh=ret[1:].mean()/ret[1:].std()*np.sqrt(31_557_600/step_sec)
    return dict(ann=eq[-1]**(1/yrs)-1, dd=dd, exp=pos.mean(), sharpe=sh, bh_ann=bh[-1]**(1/yrs)-1, bh_dd=bdd, bh_sharpe=bsh, trades=turn.sum()/2/yrs)
print(f"{'pair':9}{'bars':>5} | {'strategy @7.5bps':^34} | {'buy and hold':^26}")
print(f"{'':9}{'':>5} | {'ann%':>7}{'maxDD%':>8}{'sharpe':>7}{'in-mkt%':>8}{'trd/yr':>7} | {'ann%':>7}{'maxDD%':>8}{'sharpe':>7}")
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT']:
    px1=closes(f'h/{sym}-1h-*.csv')
    for k,lab in [(8,'8h'),(12,'12h'),(24,'1d')]:
        s=stats(px1[::k],3600*k,7.5)
        print(f"{sym:9}{lab:>5} | {s['ann']*100:7.1f}{s['dd']*100:8.1f}{s['sharpe']:7.2f}{s['exp']*100:8.1f}{s['trades']:7.0f} | {s['bh_ann']*100:7.1f}{s['bh_dd']*100:8.1f}{s['bh_sharpe']:7.2f}")
