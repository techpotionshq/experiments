"""Event-driven order-book strategy, the user's idea tested honestly.

Rule: 1-bit "buy all" gated by order-book strength, traded on signal, not on a clock.
  signal s_t = queue imbalance (bid qty - ask qty)/(bid qty + ask qty) over L levels.
  flat + s_t >= +thr  -> BUY at the ask (taker)
  long + s_t <= -thr  -> SELL at the bid (taker), go flat
  do nothing otherwise. No fixed 1s/10s/hour bar. Only buy and sell on signal.

Execution is honest: you cross the spread (buy ask, sell bid) and pay `fee` bps per
side. That captures the two real costs, the spread and the fee. We also print the
zero-fee, mid-to-mid P&L to separate "the signal has no edge" from "fees eat it".

Usage: python3 scripts/eventdriven.py ob/WIF_2026-09-01.npz [more days...] --lvl 1 --fee 10
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

def backtest(bid, ask, mid, sig, thr, fee_bps, allow_short=False):
    """State machine. Returns dict of stats. fee_bps per side on notional."""
    fee=fee_bps/1e4
    pos=0            # +1 long, 0 flat, -1 short
    entry_px=0.0
    trades=[]        # (ret_after_fee, ret_mid_zero_fee, hold_samples)
    entry_i=0
    N=len(mid)
    for t in range(N):
        s=sig[t]
        if pos==0:
            if s>=thr:
                pos=1; entry_px=ask[t]; entry_mid=mid[t]; entry_i=t
            elif allow_short and s<=-thr:
                pos=-1; entry_px=bid[t]; entry_mid=mid[t]; entry_i=t
        elif pos==1:
            if s<=-thr:
                exit_px=bid[t]
                gross=(exit_px/entry_px)-1.0
                net=gross-2*fee
                mid_ret=(mid[t]/entry_mid)-1.0
                trades.append((net,mid_ret,t-entry_i))
                pos=0
                if allow_short:
                    pos=-1; entry_px=bid[t]; entry_mid=mid[t]; entry_i=t
        elif pos==-1:
            if s>=thr:
                exit_px=ask[t]
                gross=(entry_px/exit_px)-1.0
                net=gross-2*fee
                mid_ret=(entry_mid/mid[t])-1.0
                trades.append((net,mid_ret,t-entry_i))
                pos=0
                if s>=thr:
                    pos=1; entry_px=ask[t]; entry_mid=mid[t]; entry_i=t
    if not trades:
        return None
    tr=np.array(trades)
    net=tr[:,0]; midr=tr[:,1]; hold=tr[:,2]
    comp_net=np.prod(1+net)-1.0
    comp_gross=np.prod(1+midr)-1.0
    return dict(
        n=len(tr),
        net_per_trade_bps=net.mean()*1e4,
        gross_mid_per_trade_bps=midr.mean()*1e4,
        win_rate=np.mean(net>0)*100,
        compound_net=comp_net*100,
        compound_gross_zero_fee=comp_gross*100,
        avg_hold_s=hold.mean()*0.1,       # samples are 100 ms
        median_hold_s=np.median(hold)*0.1,
    )

def main(paths, L, fee_bps):
    bp,bq,ap,aq=load(paths)
    bid=bp[:,0]; ask=ap[:,0]; mid=(bid+ask)/2
    sig=imbalance(bq,aq,L)
    spr=(ask-bid)/mid*1e4
    N=len(mid)
    hold_ret=(mid[-1]/mid[0]-1)*100
    print(f"samples {N:,} | levels {L} | median spread {np.median(spr):.2f} bps | "
          f"buy-and-hold over window {hold_ret:+.1f}% | fee {fee_bps} bps/side")
    medspr=np.median(spr)
    print(f"{'thr':>5} {'trades':>7} {'takerNet':>9} {'midEdge':>8} {'win%':>5} "
          f"{'hold_s':>7} {'beFee/side':>10} {'makerNet':>9}")
    print("  (takerNet = mid edge minus spread minus 2xfee, the realistic retail case)")
    print("  (beFee = max fee/side that breaks even AS A TAKER; makerNet = earn the spread, pay 2 bps maker, no rebate)")
    allow_short='--short' in sys.argv
    if allow_short: print("  (LONG/SHORT: flip to short on down-signal instead of going flat)")
    for thr in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]:
        r=backtest(bid,ask,mid,sig,thr,fee_bps,allow_short=allow_short)
        if r is None:
            print(f"{thr:>5} {'--- no trades ---':>7}"); continue
        mid_edge=r['gross_mid_per_trade_bps']
        be_fee=(mid_edge-medspr)/2.0                 # taker break-even fee per side
        maker_net=mid_edge+medspr-2*2.0              # earn spread instead of paying it, 2 bps maker fee, optimistic fill
        print(f"{thr:>5} {r['n']:>7} {r['net_per_trade_bps']:>8.1f}b {mid_edge:>7.2f}b "
              f"{r['win_rate']:>4.1f} {r['avg_hold_s']:>6.1f} {be_fee:>9.2f}b {maker_net:>8.2f}b")

if __name__=='__main__':
    def opt(name,default):
        if name in sys.argv:
            return sys.argv[sys.argv.index(name)+1]
        return default
    L=int(opt('--lvl','1')); fee=float(opt('--fee','10'))
    skip=set()
    for name in ('--lvl','--fee'):
        if name in sys.argv:
            i=sys.argv.index(name); skip.add(i); skip.add(i+1)
    args=[a for i,a in enumerate(sys.argv[1:],start=1) if i not in skip and not a.startswith('--')]
    main(args, L, fee)
