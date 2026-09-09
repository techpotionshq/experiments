import numpy as np
from h12 import closes, positions
import h12
def stats(px,pos,fee,per_yr):
    ret=np.r_[0.0, px[1:]/px[:-1]-1]; turn=np.abs(np.diff(np.r_[0.0,pos])); yrs=len(px)/per_yr
    per=pos*ret-turn*fee/1e4; eq=np.cumprod(1+per); dd=(eq/np.maximum.accumulate(eq)-1).min()
    return eq[-1]**(1/yrs)-1, dd, per.mean()/per.std()*np.sqrt(per_yr)
SYMS=['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT']
data={s:closes(f'h/{s}-1h-*.csv') for s in SYMS}
print("Long-only confirm-N, net/yr at 7.5 bps (sharpe). Rows: N, columns: pair. 6 years.")
for k,lab in [(8,'8h'),(12,'12h'),(24,'1d')]:
    per_yr=31_557_600/(3600*k)
    print(f"\n--- {lab} bars ---   " + "".join(f"{s.replace('USDT',''):>18}" for s in SYMS) + f"{'avg sharpe':>12}")
    bh=[stats(data[s][::k], np.r_[0.0,np.ones(len(data[s][::k])-1)],0,per_yr) for s in SYMS]
    print(f"{'buy&hold':10}" + "".join(f"{a*100:+9.1f}% ({sh:4.2f})" for a,dd,sh in bh) + f"{np.mean([b[2] for b in bh]):12.2f}")
    for n in range(1,7):
        row=[]
        for s in SYMS:
            px=data[s][::k]; d=np.sign(np.r_[0.0,np.diff(px)])
            row.append(stats(px,positions(d,'long',n),7.5,per_yr))
        print(f"{'N='+str(n):10}" + "".join(f"{a*100:+9.1f}% ({sh:4.2f})" for a,dd,sh in row) + f"{np.mean([r[2] for r in row]):12.2f}")
print("\n12h, long-only N=3, split into halves (net/yr, maxDD, sharpe) vs buy&hold")
for s in SYMS:
    px=data[s][::12]; h=len(px)//2
    for lab,sl in [('2020-09..2023-08',slice(0,h)),('2023-08..2026-08',slice(h,None))]:
        p=px[sl]; d=np.sign(np.r_[0.0,np.diff(p)])
        a,dd,sh=stats(p,positions(d,'long',3),7.5,730.5); b,bdd,bsh=stats(p,np.r_[0.0,np.ones(len(p)-1)],0,730.5)
        print(f"{s:9}{lab:18} strategy {a*100:+7.1f}%  DD {dd*100:4.0f}%  sh {sh:5.2f}   |  hold {b*100:+7.1f}%  DD {bdd*100:4.0f}%  sh {bsh:5.2f}")
