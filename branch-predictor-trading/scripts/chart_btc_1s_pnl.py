import sys, numpy as np
sys.path.insert(0, '/home/user/experiments/branch-predictor-trading')
from sim import load_closes, positions
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D='/tmp/claude-0/-home-user-experiments/69046259-b071-508c-868a-510331d370af/scratchpad/k'
px,_=load_closes(D,'BTCUSDT')
ret=np.r_[0.0, px[1:]/px[:-1]-1]
direction=np.sign(np.r_[0.0, np.diff(px)])
hours=np.arange(len(px))/3600
series={}
for name,strat,n in [('1-bit, full size','1bit',0),('ramp, 20 steps','ramp',20),('streak, 5 steps','streak',5)]:
    pos=positions(direction,strat,n)
    turn=np.abs(np.diff(np.r_[0.0,pos]))
    series[name]=(np.cumsum(pos*ret)*100, np.cumsum(pos*ret-turn*1e-4)*100)
bh=(px/px[0]-1)*100

L='#fcfcfb'; T='#0b0b0b'; T2='#52514e'; cols=['#2a78d6','#eb6834','#1baf7a']
fig,axes=plt.subplots(1,2,figsize=(12,4.6),facecolor=L)
titles=['Gross, no fees','Net of a 1 bps fee per unit traded (Binance taker is 10 bps)']
for ax,idx,title in zip(axes,[0,1],titles):
    ax.set_facecolor(L)
    for (name,(g,n)),c in zip(series.items(),cols):
        y=(g if idx==0 else n)[::60]
        ax.plot(hours[::60],y,color=c,lw=2,label=name)
        ax.text(hours[-1]+1,y[-1],f"{y[-1]:+,.0f}%",color=T,fontsize=9,va='center')
    if idx==0:
        ax.plot(hours[::60],bh[::60],color=T2,lw=1.5,ls='--',label='buy and hold')
        ax.text(hours[-1]+1,bh[-1],f"{bh[-1]:+.1f}%",color=T2,fontsize=9,va='center')
    ax.axhline(0,color=T2,lw=0.8)
    ax.set_title(title,color=T,fontsize=11,loc='left')
    ax.set_xlabel('hours (2026-09-01 to 09-07)',color=T2); ax.set_ylabel('cumulative P&L, % of capital',color=T2)
    ax.set_xlim(0,hours[-1]+30)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.grid(axis='y',color='#e6e5e2',lw=0.6); ax.tick_params(colors=T2)
    ax.legend(frameon=False,fontsize=9,loc='upper left' if idx==0 else 'lower left')
fig.suptitle('BTC/USDT, 1-second bars: branch-predictor rules',color=T,fontsize=13,x=0.01,ha='left')
fig.tight_layout()
out='/tmp/claude-0/-home-user-experiments/69046259-b071-508c-868a-510331d370af/scratchpad/btc_1s_pnl.png'
fig.savefig(out,dpi=150,facecolor=L)
print(out)
