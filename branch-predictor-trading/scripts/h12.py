import glob, numpy as np
def closes(p): return np.concatenate([np.loadtxt(f, delimiter=",", usecols=(4,)) for f in sorted(glob.glob(p))])
STEP=12*3600; PER_YR=31_557_600/STEP
def positions(d, mode, n):
    """d: bar directions. Position held in bar t decided from bars <= t-1."""
    T=len(d); pos=np.zeros(T)
    if mode in ('long','ls'):           # need n consecutive same-direction bars to switch
        state=0; run_dir=0; run_len=0
        for t in range(1,T):
            x=d[t-1]
            if x!=0:
                if x==run_dir: run_len+=1
                else: run_dir, run_len = x, 1
                if run_len>=n:
                    state = 1 if x>0 else (-1 if mode=='ls' else 0)
            pos[t]=state
    elif mode=='2bit':                   # saturating counter 0..3, long if >=2 else short
        c=2
        for t in range(1,T):
            if d[t-1]>0: c=min(3,c+1)
            elif d[t-1]<0: c=max(0,c-1)
            pos[t]=1 if c>=2 else -1
    elif mode=='2bit_long':
        c=2
        for t in range(1,T):
            if d[t-1]>0: c=min(3,c+1)
            elif d[t-1]<0: c=max(0,c-1)
            pos[t]=1 if c>=2 else 0
    return pos
def stats(px,pos,fee):
    ret=np.r_[0.0, px[1:]/px[:-1]-1]; turn=np.abs(np.diff(np.r_[0.0,pos])); yrs=len(px)/PER_YR
    per=pos*ret-turn*fee/1e4; eq=np.cumprod(1+per); dd=(eq/np.maximum.accumulate(eq)-1).min()
    return dict(ann=eq[-1]**(1/yrs)-1, gross=np.cumprod(1+pos*ret)[-1]**(1/yrs)-1, dd=dd, sh=per.mean()/per.std()*np.sqrt(PER_YR), trades=turn.sum()/2/yrs, exp=np.mean(pos!=0), lng=np.mean(pos>0))
RUNS=[('long',1,'long-only, 1 bar (baseline)'),('long',2,'long-only, 2-bar confirm'),('long',3,'long-only, 3-bar confirm'),('2bit_long',0,'long-only, 2-bit counter'),
      ('ls',1,'long-short, 1 bar'),('ls',2,'long-short, 2-bar confirm'),('ls',3,'long-short, 3-bar confirm'),('2bit',0,'long-short, 2-bit counter')]
if __name__=="__main__":
  print("12h bars, 2020-09 to 2026-08, fee 7.5 bps per side")
  for sym in ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT"]:
      px=closes(f'h/{sym}-1h-*.csv')[::12]; d=np.sign(np.r_[0.0,np.diff(px)])
      bh=stats(px,np.r_[0.0,np.ones(len(px)-1)],0)
      print(f"\n{sym}   buy&hold: {bh['ann']*100:+.1f}%/yr, maxDD {bh['dd']*100:.0f}%, sharpe {bh['sh']:.2f}")
      print(f"  {'strategy':30}{'gross/yr':>9}{'net/yr':>8}{'maxDD':>7}{'sharpe':>7}{'trd/yr':>7}{'long%':>7}")
      for mode,n,lab in RUNS:
          s=stats(px,positions(d,mode,n),7.5)
          print(f"  {lab:30}{s['gross']*100:+8.1f}%{s['ann']*100:+7.1f}%{s['dd']*100:6.0f}%{s['sh']:7.2f}{s['trades']:7.0f}{s['lng']*100:6.0f}%")
