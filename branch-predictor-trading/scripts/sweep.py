import glob, sys, json, numpy as np
def closes(pattern):
    return np.concatenate([np.loadtxt(f, delimiter=",", usecols=(4,)) for f in sorted(glob.glob(pattern))])
def run(px, step_sec, fee_bps):
    ret=np.r_[0.0, px[1:]/px[:-1]-1]
    d=np.sign(np.r_[0.0, np.diff(px)])
    # all in after up, all out after down, hold on flat; decided on bar t-1, held during bar t
    pos=np.zeros(len(px)); c=0
    for t in range(1,len(px)):
        if d[t-1]>0: c=1
        elif d[t-1]<0: c=0
        pos[t]=c
    turn=np.abs(np.diff(np.r_[0.0,pos]))
    gross=(pos*ret).sum(); T=turn.sum()
    years=len(px)*step_sec/31_557_600
    out={'years':years,'trades_per_year':T/2/years,'be_bps':gross/T*1e4 if T else float('nan'),
         'bh_ann':(px[-1]/px[0])**(1/years)-1}
    for f in fee_bps:
        eq=np.cumprod(1+pos*ret-turn*f/1e4)
        out[f'net{f}']=eq[-1]**(1/years)-1 if eq[-1]>0 else -1.0
    return out
SRC=[('k','1s',1,[1,2,3,5,10,15,30,60]),('m','1m',60,[1,2,3,5,10,15,30,60,120,240,480]),('h','1h',3600,[1,2,4,6,8,12,24])]
FEES=[0,7.5]
res={}
for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT']:
    res[sym]=[]
    for d,tag,base,mults in SRC:
        px1=closes(f'{d}/{sym}-{tag}-*.csv')
        for k in mults:
            r=run(px1[::k], base*k, FEES); r['interval_s']=base*k; r['src']=tag; res[sym].append(r)
            print(f"{sym:9}{base*k:>7}s  {r['years']:5.2f}y  trades/yr {r['trades_per_year']:>10,.0f}  b/e {r['be_bps']:7.2f} bps  net@0 {r['net0']*100:>9.1f}%  net@7.5 {r['net7.5']*100:>8.1f}%  B&H {r['bh_ann']*100:>7.1f}%", flush=True)
json.dump(res, open('sweep.json','w'))
