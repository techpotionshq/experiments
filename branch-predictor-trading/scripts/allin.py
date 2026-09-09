import sys, numpy as np
sys.path.insert(0,'/home/user/experiments/branch-predictor-trading')
from sim import load_closes, resample, positions
D='/tmp/claude-0/-home-user-experiments/69046259-b071-508c-868a-510331d370af/scratchpad/k'
FEES=[0,0.5,1,2,5,7.5,10]
K=int(sys.argv[1]) if len(sys.argv)>1 else 5
print(f"All-in on up tick, all-out on down tick, {K}s bars, 7 days, $1000 start, compounded")
print(f"{'pair':9}{'up%':>6}{'hit%':>6}{'trades':>8}{'in-mkt%':>8}{'b/e bps':>8} |"+"".join(f"{'$@'+str(f):>9}" for f in FEES)+"  | bust@7.5bps")
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT']:
    px1,_=load_closes(D,sym); px=resample(px1,K)
    ret=np.r_[0.0, px[1:]/px[:-1]-1]
    direction=np.sign(np.r_[0.0,np.diff(px)])
    pos=positions(direction,'ramp_long',1)     # counter in [0,1] = all in / all out
    turn=np.abs(np.diff(np.r_[0.0,pos]))
    gross=(pos*ret).sum(); T=turn.sum()
    live=(direction!=0)&(pos!=0); hit=np.mean(direction[live]>0)
    row=f"{sym:9}{np.mean(ret[1:]>0)*100:6.1f}{hit*100:6.1f}{int(T/2):8d}{pos.mean()*100:8.1f}{gross/T*1e4:8.2f} |"
    bust=None
    for f in FEES:
        eq=1000*np.cumprod(1+pos*ret-turn*f/1e4)
        row+=f"{eq[-1]:9.2f}"
        if f==7.5:
            b=np.argmax(eq<1.0); bust=f"{b*K/3600:.1f} h" if eq[b]<1.0 else "never"
    print(row+f"  | {bust}")
    print(f"{'':9}buy&hold: ${1000*px[-1]/px[0]:.2f}")
