"""Maker version of the imbalance strategy on real Bybit order books.

Instead of crossing the spread as a taker, we POST passive limit orders and try
to EARN the spread, using the same queue-imbalance signal to pick the side.

  strong up-signal (imb >= thr) and flat -> post a passive BUY at the best bid B.
      leave it resting up to `patience` samples.
      FILL only if the market trades THROUGH it: best bid later drops below B.
      (that is exactly the adverse-selection case: you buy as price ticks down.)
      if price runs up first without filling you, cancel and miss it (realistic).
  once long at B -> post a passive SELL at the ask that stood at fill time (target
      = B + spread). Wait up to `hold` samples for best ask to trade through it.
      if it fills, you earned the spread minus two maker fees / plus two rebates.
      if it does not fill in time, MARKET OUT at the bid (taker fee), a real loss.

Fill = "price trades through the resting order", the honest convention from
scripts/maker.py. No trade prints in the archive, so we proxy fills from the
best-bid/ask path, which is conservative on price and optimistic on queue
position (assumes you are near the front of your level).

Usage: python3 scripts/makerbook.py ob/WIF_2026-09-01.npz [more days...] \
          --lvl 1 --thr 0.6 --patience 20 --hold 100 --short
Sweeps a grid of maker fees (Bybit spot base is +10 bps; rebate venues pay -2).
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

def simulate(bidp, askp, sig, thr, patience, hold, taker_bps, allow_short):
    """Return list of (entry, exit, side, maker_exit_bool) price pairs, no fees yet."""
    N=len(bidp)
    trades=[]              # (entry_px, exit_px, side, maker_exit)
    t=0
    while t < N-1:
        s=sig[t]
        side=0
        if s>=thr: side=1
        elif allow_short and s<=-thr: side=-1
        if side==0: t+=1; continue
        # --- try to get a passive entry fill ---
        if side==1:
            B=bidp[t]; target=askp[t]           # buy at bid, aim to sell at the ask that stood now
            tf=-1
            for u in range(t+1, min(t+1+patience, N)):
                if bidp[u] < B: tf=u; break      # traded through our bid -> filled
                if bidp[u] > B: break            # price ran up, our order never fills, cancel
            if tf<0: t+=1; continue
            entry=B
            # --- passive exit: sell at target, else taker-out at the bid ---
            xf=-1
            for u in range(tf+1, min(tf+1+hold, N)):
                if askp[u] > target: xf=u; break
            if xf>=0:
                trades.append((entry, target, 1, True)); t=xf+1
            else:
                uend=min(tf+hold, N-1); trades.append((entry, bidp[uend], 1, False)); t=uend+1
        else:
            A=askp[t]; target=bidp[t]            # sell at ask, aim to buy back at the bid that stood now
            tf=-1
            for u in range(t+1, min(t+1+patience, N)):
                if askp[u] > A: tf=u; break
                if askp[u] < A: break
            if tf<0: t+=1; continue
            entry=A
            xf=-1
            for u in range(tf+1, min(tf+1+hold, N)):
                if bidp[u] < target: xf=u; break
            if xf>=0:
                trades.append((entry, target, -1, True)); t=xf+1
            else:
                uend=min(tf+hold, N-1); trades.append((entry, askp[uend], -1, False)); t=uend+1
    return trades

def pnl_at_fee(trades, maker_bps, taker_bps):
    out=[]
    for entry, exitpx, side, maker_exit in trades:
        gross=side*(exitpx/entry - 1.0)*1e4
        fee = 2*maker_bps if maker_exit else (maker_bps + taker_bps)
        out.append(gross - fee)
    return np.array(out) if out else np.array([])

def main(paths, L, thr, patience, hold, taker_bps, allow_short):
    bp,bq,ap,aq=load(paths)
    bidp=bp[:,0]; askp=ap[:,0]; mid=(bidp+askp)/2
    sig=imbalance(bq,aq,L)
    spr=(askp-bidp)/mid*1e4
    N=len(mid)
    trades=simulate(bidp, askp, sig, thr, patience, hold, taker_bps, allow_short)
    nt=len(trades)
    if nt==0: print("no trades"); return
    maker_exits=np.mean([tr[3] for tr in trades])*100
    hold_ret=(mid[-1]/mid[0]-1)*100
    print(f"samples {N:,} | L={L} thr={thr} patience={patience*0.1:.1f}s hold={hold*0.1:.1f}s "
          f"{'LONG/SHORT' if allow_short else 'LONG-ONLY'}")
    print(f"median spread {np.median(spr):.2f} bps | buy-and-hold {hold_ret:+.1f}% | "
          f"trades {nt} | passive-exit rate {maker_exits:.0f}% (rest market out)")
    print(f"  {'makerFee':>9} {'net/trade':>10} {'win%':>6} {'compound%':>10}   (taker-out pays maker+{taker_bps:.0f} taker)")
    for mk in [10, 5, 2, 0, -1, -2, -2.5]:
        p=pnl_at_fee(trades, mk, taker_bps)
        comp=(np.prod(1+p/1e4)-1)*100
        tag=''
        if mk==10: tag='Bybit spot base'
        elif mk==2: tag='typical perp maker'
        elif mk==0: tag='free maker'
        elif mk==-2: tag='Kraken T12 / rebate'
        print(f"  {mk:>8.1f}b {p.mean():>9.2f}b {np.mean(p>0)*100:>5.1f} {comp:>9.1f}   {tag}")

if __name__=='__main__':
    def opt(name,default):
        return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default
    L=int(opt('--lvl','1')); thr=float(opt('--thr','0.6'))
    patience=int(opt('--patience','20')); hold=int(opt('--hold','100'))
    taker=float(opt('--taker','10')); allow_short='--short' in sys.argv
    skip=set()
    for name in ('--lvl','--thr','--patience','--hold','--taker'):
        if name in sys.argv:
            i=sys.argv.index(name); skip.add(i); skip.add(i+1)
    args=[a for i,a in enumerate(sys.argv[1:],start=1) if i not in skip and not a.startswith('--')]
    main(args, L, thr, patience, hold, taker, allow_short)
