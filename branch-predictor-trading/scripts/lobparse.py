"""Rebuild Bybit orderbook.200 stream (snapshot + deltas) and sample top-L levels every DT ms.
Output .npz: t (ms), bp,bq,ap,aq arrays of shape (N, L)."""
import json, sys, numpy as np
def parse(path, L=10, dt_ms=100):
    bids={}; asks={}; out=[]; next_t=None
    with open(path) as f:
        for line in f:
            m=json.loads(line); d=m['data']; ts=m['ts']
            if m['type']=='snapshot':
                bids={float(p):float(q) for p,q in d['b']}; asks={float(p):float(q) for p,q in d['a']}
            else:
                for p,q in d['b']:
                    p=float(p); q=float(q)
                    if q==0: bids.pop(p,None)
                    else: bids[p]=q
                for p,q in d['a']:
                    p=float(p); q=float(q)
                    if q==0: asks.pop(p,None)
                    else: asks[p]=q
            if next_t is None: next_t=ts - ts%dt_ms + dt_ms
            while ts>=next_t:
                b=sorted(bids.items(),reverse=True)[:L]; a=sorted(asks.items())[:L]
                if len(b)==L and len(a)==L:
                    out.append((next_t, [x[0] for x in b],[x[1] for x in b],[x[0] for x in a],[x[1] for x in a]))
                next_t+=dt_ms
    t=np.array([o[0] for o in out]); bp=np.array([o[1] for o in out]); bq=np.array([o[2] for o in out])
    ap=np.array([o[3] for o in out]); aq=np.array([o[4] for o in out])
    return t,bp,bq,ap,aq
if __name__=='__main__':
    src,dst=sys.argv[1],sys.argv[2]; L=int(sys.argv[3]) if len(sys.argv)>3 else 10; dt=int(sys.argv[4]) if len(sys.argv)>4 else 100
    t,bp,bq,ap,aq=parse(src,L,dt); np.savez_compressed(dst,t=t,bp=bp,bq=bq,ap=ap,aq=aq)
    mid=(bp[:,0]+ap[:,0])/2; spr=(ap[:,0]-bp[:,0])/mid*1e4
    print(f"{src}: {len(t):,} samples @ {dt}ms | median spread {np.median(spr):.2f} bps | spread==1 tick {np.mean((ap[:,0]-bp[:,0])<=1.0001*np.min(ap[:,0]-bp[:,0]))*100:.0f}% | mid moved between samples {np.mean(np.diff(mid)!=0)*100:.1f}%")
