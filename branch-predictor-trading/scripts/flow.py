import glob, numpy as np
files=sorted(glob.glob('k/BTCUSDT-1s-*.csv'))
a=np.concatenate([np.loadtxt(f,delimiter=',',usecols=(4,5,9)) for f in files])  # close, volume, taker-buy volume
px,vol,buy=a[:,0],a[:,1],a[:,2]
sell=vol-buy; signed=buy-sell                    # net taker flow per second, in BTC
ret=np.r_[0.0, px[1:]/px[:-1]-1]
def roll(x,k): c=np.cumsum(np.r_[0.0,x]); return c[k:]-c[:-k]
print("BTCUSDT spot, 1s bars, 7 days. Signal = net taker flow (buy minus sell volume) over the last k seconds,")
print("known at the end of second t. Target = return over the next h seconds. Fee reference: 7.5 bps spot taker, 5 bps futures taker.\n")
print(f"{'k(s)':>5}{'h(s)':>5}{'corr':>8}{'hit% (top/bottom 10%)':>24}{'mean next ret, top 10%':>24}{'bottom 10%':>12}{'|edge| in bps':>15}")
for k in [1,5,15,60,300]:
    sig=np.r_[np.full(k-1,np.nan), roll(signed,k)]
    for h in [1,5,15,60,300,900]:
        fut=np.r_[roll(ret[1:],h), np.full(h,np.nan)]        # sum of returns t+1..t+h
        m=~np.isnan(sig)&~np.isnan(fut)
        s,f=sig[m],fut[m]
        c=np.corrcoef(s,f)[0,1]
        hi,lo=np.quantile(s,0.9),np.quantile(s,0.1)
        top,bot=f[s>=hi],f[s<=lo]
        hit=(np.mean(top[top!=0]>0)+np.mean(bot[bot!=0]<0))/2
        edge=(top.mean()-bot.mean())/2*1e4
        print(f"{k:5d}{h:5d}{c:8.3f}{hit*100:24.1f}{top.mean()*1e4:24.3f}{bot.mean()*1e4:12.3f}{edge:15.3f}")
