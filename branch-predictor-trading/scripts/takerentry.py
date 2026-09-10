"""Buy at market, sell at limit. Taker entry, maker exit.

On a strong up-signal, BUY immediately at the ask (taker: you never miss the move
and there is no entry adverse selection). Then post a passive SELL limit at
entry*(1+target). It fills, maker, when the ask trades up through the target. If
it has not filled after `hold`, MARKET OUT at the bid (taker).

net/trade = (exit/entry - 1) in bps
            - taker_bps            (the market buy)
            - maker_bps or taker_bps on the exit, depending on how it left

Fill = price trades THROUGH the resting sell (best ask > target), the honest
convention from scripts/maker.py. Book-only, optimistic on queue position.

Usage: python3 scripts/takerentry.py ob/WIF_*.npz --lvl 1 --thr 0.7 --hold 100 --taker 10
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

def run(bidp, askp, sig, thr, hold, target_bps, taker_bps, maker_bps):
    N=len(bidp); f_t=taker_bps; f_m=maker_bps
    trades=[]      # (price_ret_bps, exit_fee_bps, maker_exit)
    t=0
    while t < N-1:
        if sig[t] < thr: t+=1; continue
        entry=askp[t]                          # market buy at the ask
        target=entry*(1+target_bps/1e4)
        xf=-1
        for u in range(t+1, min(t+1+hold, N)):
            if askp[u] > target: xf=u; break   # sell limit lifted -> maker exit at target
        if xf>=0:
            ret=(target/entry-1)*1e4; trades.append((ret, f_m, True)); t=xf+1
        else:
            uend=min(t+hold, N-1); ret=(bidp[uend]/entry-1)*1e4
            trades.append((ret, f_t, False)); t=uend+1
    if not trades: return None
    arr=np.array([(r, r - f_t - ef) for (r,ef,_) in trades])  # gross_price, net
    maker_rate=np.mean([m for (_,_,m) in trades])*100
    net=arr[:,1]
    return dict(n=len(trades), maker_exit_rate=maker_rate,
                gross_price_bps=arr[:,0].mean(), net_bps=net.mean(),
                win=np.mean(net>0)*100, compound=(np.prod(1+net/1e4)-1)*100)

def main(paths, L, thr, hold, taker_bps):
    bp,bq,ap,aq=load(paths)
    bidp=bp[:,0]; askp=ap[:,0]; mid=(bidp+askp)/2
    sig=imbalance(bq,aq,L)
    spr=(askp-bidp)/mid*1e4; N=len(mid)
    print(f"samples {N:,} | L={L} thr={thr} hold={hold*0.1:.1f}s | median spread {np.median(spr):.2f} bps | "
          f"market-buy pays {taker_bps} bps taker + half-spread on entry")
    for maker_bps,mlab in [(10,'Bybit maker'),(0,'free maker'),(-2,'rebate -2')]:
        print(f"\n  --- exit maker fee {maker_bps:+.0f} bps ({mlab}), taker-out {taker_bps} bps ---")
        print(f"  {'target':>7} {'trades':>7} {'makerExit%':>10} {'grossPx':>8} {'net/trade':>10} {'win%':>6} {'compound%':>10}")
        for tgt in [0,2.5,5,7.5,10,15,20]:
            r=run(bidp,askp,sig,thr,hold,tgt,taker_bps,maker_bps)
            if r is None: print(f"  {tgt:>7} none"); continue
            print(f"  {tgt:>6.1f}b {r['n']:>7} {r['maker_exit_rate']:>9.1f} {r['gross_price_bps']:>7.2f}b "
                  f"{r['net_bps']:>9.2f}b {r['win']:>5.1f} {r['compound']:>9.1f}")

if __name__=='__main__':
    def opt(name,default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    L=int(opt('--lvl','1')); thr=float(opt('--thr','0.7')); hold=int(opt('--hold','100')); taker=float(opt('--taker','10'))
    skip=set()
    for name in ('--lvl','--thr','--hold','--taker'):
        if name in sys.argv:
            i=sys.argv.index(name); skip.add(i); skip.add(i+1)
    args=[a for i,a in enumerate(sys.argv[1:],start=1) if i not in skip and not a.startswith('--')]
    main(args, L, thr, hold, taker)
