import sys, numpy as np
sys.path.insert(0,'/home/user/experiments/branch-predictor-trading')
from sim import load_closes, resample, positions
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='/tmp/claude-0/-home-user-experiments/69046259-b071-508c-868a-510331d370af/scratchpad/k'
px,_=load_closes(D,'BTCUSDT'); px=resample(px,5)
ret=np.r_[0.0, px[1:]/px[:-1]-1]; direction=np.sign(np.r_[0.0,np.diff(px)])
pos=positions(direction,'ramp_long',1); turn=np.abs(np.diff(np.r_[0.0,pos]))
h=np.arange(len(px))*5/3600
L='#fcfcfb'; T='#0b0b0b'; T2='#52514e'
fig,ax=plt.subplots(figsize=(11,4.8),facecolor=L); ax.set_facecolor(L)
for f,c,lab in [(0,'#2a78d6','no fee'),(0.5,'#1baf7a','0.5 bps fee'),(1,'#eda100','1 bps fee'),(7.5,'#eb6834','7.5 bps fee (Binance with BNB)')]:
    eq=1000*np.cumprod(1+pos*ret-turn*f/1e4); eq=np.maximum(eq,0.01)
    ax.plot(h,eq,color=c,lw=2,label=lab)
    ax.text(h[-1]+1,eq[-1],f"${eq[-1]:,.2f}" if eq[-1]>=0.01 else "$0",color=T,fontsize=9,va='center')
ax.plot(h,1000*px/px[0],color=T2,lw=1.5,ls='--',label='buy and hold'); ax.text(h[-1]+1,1000*px[-1]/px[0]*0.8,f"${1000*px[-1]/px[0]:,.0f}",color=T2,fontsize=9)
ax.set_yscale('log'); ax.set_ylim(0.01,5000)
ax.set_yticks([0.01,0.1,1,10,100,1000]); ax.set_yticklabels(['$0.01','$0.10','$1','$10','$100','$1,000'])
ax.set_title('BTC/USDT, 5-second bars: all in on an up tick, all out on a down tick. $1,000 start, 7 days',color=T,fontsize=11,loc='left')
ax.set_xlabel('hours',color=T2); ax.set_ylabel('account value (log scale)',color=T2)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.grid(axis='y',color='#e6e5e2',lw=0.6); ax.tick_params(colors=T2); ax.set_xlim(0,h[-1]+25)
ax.legend(frameon=False,fontsize=9,loc='lower left'); fig.tight_layout()
out='/tmp/claude-0/-home-user-experiments/69046259-b071-508c-868a-510331d370af/scratchpad/btc_5s_allin.png'; fig.savefig(out,dpi=150,facecolor=L); print(out)
