"""Symmetric market maker with imbalance-based cancel, the real MM business.

Quote a bid at the best bid and an ask at the best ask, size 1 each. Earn the
spread on round trips. Use queue imbalance not to bet direction but to CANCEL
the toxic side:
  imb >= thr (mid about to tick up)   -> pull the ASK (don't sell into a rise)
  imb <= -thr (mid about to tick down) -> pull the BID (don't buy into a fall)
Hold inventory within +/- Qmax; stop quoting a side when the limit is hit.

Fill = price trades THROUGH the resting quote (best bid drops below our bid, or
best ask rises above our ask), the same honest convention as scripts/maker.py.
Book-only fills, so this is optimistic on queue position (assumes near front) and
conservative on price. Inventory is marked to mid at the end.

Usage: python3 scripts/makermm.py ob/WIF_*.npz --lvl 1 --thr 0.6 --qmax 20 [--nocancel]
"""
import sys, numpy as np

def load(paths):
    BP=[];BQ=[];AP=[];AQ=[]
    for p in paths:
        z=np.load(p); BP.append(z['bp']);BQ.append(z['bq']);AP.append(z['ap']);AQ.append(z['aq'])
    return (np.concatenate(BP),np.concatenate(BQ),np.concatenate(AP),np.concatenate(AQ))

def imbalance(bq,aq,L):
    b=bq[:,:L].sum(1); a=aq[:,:L].sum(1)
    return (b-a)/(b+a+1e-12)

def mm(bidp, askp, mid, imb, thr, Qmax, maker_bps, use_cancel):
    N=len(mid); f=maker_bps/1e4
    q=0; cash=0.0; nb=ns=0
    post_bid=None; post_ask=None
    for t in range(N):
        # fills against quotes posted at the previous step, judged by current best prices
        if post_bid is not None and bidp[t] < post_bid and q < Qmax:
            q+=1; cash-=post_bid*(1+f); nb+=1
        if post_ask is not None and askp[t] > post_ask and q > -Qmax:
            q-=1; cash+=post_ask*(1-f); ns+=1
        up = imb[t]>=thr; dn = imb[t]<=-thr
        if not use_cancel: up=dn=False
        post_bid = bidp[t] if (q<Qmax and not dn) else None
        post_ask = askp[t] if (q>-Qmax and not up) else None
    total = cash + q*mid[-1]                 # mark remaining inventory to mid
    return dict(total=total, fills=nb+ns, nb=nb, ns=ns, endq=q)

def main(paths, L, thr, Qmax):
    bp,bq,ap,aq=load(paths)
    bidp=bp[:,0]; askp=ap[:,0]; mid=(bidp+askp)/2
    imb=imbalance(bq,aq,L)
    spr=(askp-bidp)/mid*1e4
    N=len(mid); mavg=mid.mean(); cap=Qmax*mavg
    days=N/864000.0
    print(f"samples {N:,} (~{days:.1f} days) | L={L} thr={thr} Qmax={Qmax} | "
          f"median spread {np.median(spr):.2f} bps | capital ~= Qmax*price = {cap:,.0f} quote-ccy")
    for use_cancel,label in [(True,'imbalance-cancel ON'),(False,'always quote both (no cancel)')]:
        print(f"\n  --- {label} ---")
        print(f"  {'makerFee':>9} {'fills':>8} {'PnL/fill':>9} {'totalPnL':>12} {'return%':>9} {'return%/day':>11}")
        for mk in [10, 2, 0, -1, -2, -2.5]:
            r=mm(bidp, askp, mid, imb, thr, Qmax, mk, use_cancel)
            pnl_per_fill = (r['total']/max(r['fills'],1))/mavg*1e4
            ret = r['total']/cap*100
            print(f"  {mk:>8.1f}b {r['fills']:>8} {pnl_per_fill:>8.3f}b {r['total']:>12.2f} {ret:>8.1f} {ret/days:>10.2f}")
        print(f"    (buy fills {r['nb']}, sell fills {r['ns']}, end inventory {r['endq']})")

if __name__=='__main__':
    def opt(name,default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    L=int(opt('--lvl','1')); thr=float(opt('--thr','0.6')); Qmax=int(opt('--qmax','20'))
    skip=set()
    for name in ('--lvl','--thr','--qmax'):
        if name in sys.argv:
            i=sys.argv.index(name); skip.add(i); skip.add(i+1)
    args=[a for i,a in enumerate(sys.argv[1:],start=1) if i not in skip and not a.startswith('--')]
    main(args, L, thr, Qmax)
