import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
res=json.load(open('sweep.json'))
L='#fcfcfb'; T='#0b0b0b'; T2='#52514e'
cols={'BTCUSDT':'#2a78d6','ETHUSDT':'#eb6834','SOLUSDT':'#1baf7a','DOGEUSDT':'#eda100'}
def pick(rows):
    # seconds from 1s data, minutes to 30m from 1m data, 1h and up from 1h data
    out=[]
    for r in rows:
        s=r['interval_s']
        if (s<60 and r['src']=='1s') or (60<=s<3600 and r['src']=='1m') or (s>=3600 and r['src']=='1h'): out.append(r)
    return sorted(out,key=lambda r:r['interval_s'])
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),facecolor=L)
ticks=[1,10,60,600,3600,4*3600,12*3600,86400]; labels=['1s','10s','1m','10m','1h','4h','12h','1d']
ax=axes[0]; ax.set_facecolor(L)
for sym,c in cols.items():
    rows=pick(res[sym]); x=[r['interval_s'] for r in rows]; y=[r['be_bps'] for r in rows]
    ax.plot(x,y,color=c,lw=2,marker='o',ms=4,label=sym.replace('USDT',''))
ax.axhline(7.5,color=T,lw=1,ls='--'); ax.text(90000,8.6,'Binance fee with BNB: 7.5 bps',color=T,fontsize=9,ha='right')
ax.axhline(0,color=T2,lw=0.8)
ax.set_xscale('log'); ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.set_yscale('symlog',linthresh=1); ax.set_yticks([-1,0,1,3,7.5,20,50]); ax.set_yticklabels(['-1','0','1','3','7.5','20','50'])
ax.set_title('Edge per trade: break-even fee in bps, by bar interval',color=T,fontsize=11,loc='left')
ax.set_xlabel('bar interval',color=T2); ax.set_ylabel('bps earned per unit traded (symlog)',color=T2)
ax.legend(frameon=False,fontsize=9,loc='lower right'); ax.grid(axis='y',color='#e6e5e2',lw=0.6)
ax=axes[1]; ax.set_facecolor(L)
for sym,c in cols.items():
    rows=[r for r in pick(res[sym]) if r['interval_s']>=3600]; x=[r['interval_s'] for r in rows]
    ax.plot(x,[max(r['net7.5'],-1)*100 for r in rows],color=c,lw=2,marker='o',ms=4,label=sym.replace('USDT','')+' strategy')
    ax.plot(x,[r['bh_ann']*100 for r in rows],color=c,lw=1.2,ls=':',label=sym.replace('USDT','')+' buy & hold')
ax.axhline(0,color=T2,lw=0.8)
ax.set_xscale('log'); ax.set_xticks([3600,2*3600,4*3600,8*3600,12*3600,86400]); ax.set_xticklabels(['1h','2h','4h','8h','12h','1d'])
ax.set_title('Net annual return at 7.5 bps, 2020-09 to 2026-08 (6 years)',color=T,fontsize=11,loc='left')
ax.set_xlabel('bar interval',color=T2); ax.set_ylabel('annualised return, %',color=T2)
ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower right'); ax.grid(axis='y',color='#e6e5e2',lw=0.6)
for a in axes:
    for s in ['top','right']: a.spines[s].set_visible(False)
    a.tick_params(colors=T2)
fig.suptitle('All in on an up bar, all out on a down bar: which interval?',color=T,fontsize=13,x=0.01,ha='left')
fig.tight_layout(); fig.savefig('interval_sweep.png',dpi=150,facecolor=L); print('ok')
