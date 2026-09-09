import glob, csv, numpy as np, datetime as dt
# depth snapshots: every 30s, levels at +-0.2%, 1..5% ; columns timestamp,percentage,depth(BTC),notional
snap={}
for f in sorted(glob.glob('bd/BTCUSDT-bookDepth-*.csv')):
    for r in csv.DictReader(open(f)):
        ts=int(dt.datetime.strptime(r['timestamp'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp())
        snap.setdefault(ts,{})[float(r['percentage'])]=float(r['depth'])
# futures 1m closes keyed by open-time second
px={}
for f in sorted(glob.glob('bd/BTCUSDT-1m-*.csv')):
    for row in csv.reader(open(f)):
        if row[0]=='open_time': continue
        px[int(row[0])//1000]=float(row[4])
pts=sorted(px); parr=np.array([px[t] for t in pts]); tarr=np.array(pts)
def price_at(t):  # close of the minute bar containing t (the last known price at end of that minute); use bar whose open<=t
    i=np.searchsorted(tarr,t,side='right')-1; return i
rows=[]
for ts,lv in sorted(snap.items()):
    if len(lv)<12: continue
    i=price_at(ts)
    if i<0 or i+60>=len(parr): continue
    imb={}
    for name,lvls in [('0.2%',[0.2]),('1%',[0.2,1.0]),('2%',[0.2,1.0,2.0]),('5%',[0.2,1,2,3,4,5])]:
        b=sum(lv[-l] for l in lvls); a=sum(lv[l] for l in lvls); imb[name]=(b-a)/(b+a)
    fut={h:parr[i+h]/parr[i]-1 for h in [1,5,15,60]}
    rows.append((imb,fut))
print(f"BTCUSDT futures, {len(rows):,} depth snapshots (every 30s) over 7 days.")
print("Signal = (bid depth - ask depth)/(bid+ask) within +-X% of mid. Target = return over next h minutes.\n")
print(f"{'band':>6}{'h(min)':>7}{'corr':>8}{'hit% top/bot 10%':>18}{'top10% mean':>13}{'bot10% mean':>13}{'|edge| bps':>12}")
for name in ['0.2%','1%','2%','5%']:
    s=np.array([r[0][name] for r in rows])
    for h in [1,5,15,60]:
        f=np.array([r[1][h] for r in rows]); c=np.corrcoef(s,f)[0,1]
        hi,lo=np.quantile(s,0.9),np.quantile(s,0.1); top,bot=f[s>=hi],f[s<=lo]
        hit=(np.mean(top[top!=0]>0)+np.mean(bot[bot!=0]<0))/2
        print(f"{name:>6}{h:7d}{c:8.3f}{hit*100:18.1f}{top.mean()*1e4:13.2f}{bot.mean()*1e4:13.2f}{(top.mean()-bot.mean())/2*1e4:12.2f}")
