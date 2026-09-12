"""Trade-DRIVEN passive market maker: honest fills from real Bybit prints.

Fixes the book-only ambiguity (a shrinking best-quote size can be a trade OR a
cancel). Here fills are driven only by actual aggressive trades:
  resting BID at price p, joined behind `qa` base units:
     each aggressive SELL with price <= p consumes size; once cumulative consumed
     volume since we joined exceeds qa, our unit fills at p. A sell strictly
     below p (walked through our level) fills us outright.
  resting ASK at price p: symmetric with aggressive BUYs at price >= p.
Cancels never fill us. Queue position is conservative: on join / reprice we sit
at the BACK of the current resting size, and every completed fill re-joins at the
back. Decisions (which side to show) use the 100ms book; the static signal is the
top-3 queue imbalance. Maker fee per fill on notional (negative = rebate).
"""
import sys, argparse, csv, numpy as np


def load_trades(path):
    """Bybit spot trade archive: id,timestamp(ms),price,volume(base),side(buy/sell),rpi.
    side is the aggressor: 'sell' hits the bid, 'buy' hits the ask."""
    ts = []; side = []; sz = []; px = []
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            ts.append(float(row[1])); px.append(float(row[2])); sz.append(float(row[3])); side.append(row[4])
    ts = np.array(ts)                          # already ms
    sell = np.array([s.lower() == 'sell' for s in side])
    sz = np.array(sz); px = np.array(px)
    o = np.argsort(ts, kind='stable')
    return ts[o], sell[o], sz[o], px[o]


def run(npz, trades_csv, fee_bps=0.0, imb_thr=0.5, start_usd=100.0, order_usd=25.0,
        max_inv_usd=100.0, skew=True, reprice=True, mode='join'):
    z = np.load(npz)
    tb = z['t'].astype(np.int64)
    bp, bq, ap, aq = z['bp'][:, 0], z['bq'][:, 0], z['ap'][:, 0], z['aq'][:, 0]
    bq3, aq3 = z['bq'][:, :3].sum(1), z['aq'][:, :3].sum(1)
    N = len(bp); mid = (bp + ap) / 2.0
    I = (bq3 - aq3) / (bq3 + aq3 + 1e-12)
    d = np.diff(np.unique(np.round(ap, 10))); tick = np.median(d[d > 0])   # for 'inside' queue-jump mode

    tt, tsell, tsz, tpx = load_trades(trades_csv)
    # index of first trade >= each book time
    tidx = np.searchsorted(tt, tb)

    cash = start_usd; q = 0.0; fee = fee_bps / 1e4
    bid_px = np.nan; bid_qa = 0.0
    ask_px = np.nan; ask_qa = 0.0
    nb = na = 0
    eq = np.empty(N)

    for t in range(N - 1):
        unit = order_usd / mid[t]; qmax = max_inv_usd / mid[t]
        if skew:
            want_bid = (I[t] >= imb_thr or q < 0) and q < qmax
            want_ask = (I[t] <= -imb_thr or q > 0) and q > -qmax
        else:
            want_bid = q < qmax; want_ask = q > -qmax

        # target price + queue-ahead depend on mode. 'inside': when the spread is
        # wider than 1 tick, post 1 tick better than best -> alone at the front (qa=0).
        wide = (ap[t] - bp[t]) > 1.5 * tick
        if mode == 'inside' and wide:
            tbid, tbq = bp[t] + tick, 0.0
            task, taq = ap[t] - tick, 0.0
        else:
            tbid, tbq = bp[t], bq[t]
            task, taq = ap[t], aq[t]

        if want_bid:
            if np.isnan(bid_px) or (reprice and bid_px != tbid):
                bid_px = tbid; bid_qa = tbq
        else:
            bid_px = np.nan
        if want_ask:
            if np.isnan(ask_px) or (reprice and ask_px != task):
                ask_px = task; ask_qa = taq
        else:
            ask_px = np.nan

        # process trades in (tb[t], tb[t+1]]
        lo, hi = tidx[t], tidx[t + 1]
        for k in range(lo, hi):
            if not np.isnan(bid_px) and tsell[k] and tpx[k] <= bid_px + 1e-15:
                bid_qa -= tsz[k]
                if bid_qa <= 0:
                    cash -= bid_px * unit + abs(bid_px * unit) * fee
                    q += unit; nb += 1; bid_px = np.nan
            if not np.isnan(ask_px) and (not tsell[k]) and tpx[k] >= ask_px - 1e-15:
                ask_qa -= tsz[k]
                if ask_qa <= 0:
                    cash += ask_px * unit - abs(ask_px * unit) * fee
                    q -= unit; na += 1; ask_px = np.nan
        eq[t] = cash + q * mid[t]
    eq[-1] = cash + q * mid[-1]

    # honest flatten: cross the spread to close residual inventory + pay taker fee
    endinv_mid = q * mid[-1]
    taker = 10.0 / 1e4          # Bybit spot taker ~10 bps
    if q > 0:
        cash += q * bp[-1] * (1 - taker)      # sell into the bid
    elif q < 0:
        cash += q * ap[-1] * (1 + taker)      # buy back at the ask (q<0 -> pay)
    q = 0.0
    final = cash
    ret = (final / start_usd - 1) * 100
    fills = nb + na; hrs = (tb[-1] - tb[0]) / 1000 / 3600
    peak = eq[0]; maxdd = 0.0
    for e in eq:
        peak = max(peak, e); maxdd = min(maxdd, e / peak - 1)
    return dict(ret=ret, final=final, fills=fills, fills_per_hr=fills / hrs, nb=nb, na=na,
                maxdd=maxdd * 100, endinv_usd=endinv_mid, hrs=hrs, eq=eq, mid=mid, tb=tb)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('npz'); a.add_argument('trades')
    a.add_argument('--fee', type=float, default=0.0); a.add_argument('--thr', type=float, default=0.5)
    a.add_argument('--order', type=float, default=25.0); a.add_argument('--maxinv', type=float, default=100.0)
    a.add_argument('--start', type=float, default=100.0); a.add_argument('--noskew', action='store_true')
    g = a.parse_args()
    r = run(g.npz, g.trades, fee_bps=g.fee, imb_thr=g.thr, start_usd=g.start, order_usd=g.order,
            max_inv_usd=g.maxinv, skew=not g.noskew)
    print(f"{g.npz.split('/')[-1]:24} fee {g.fee:5.1f} thr {g.thr:.2f} skew {not g.noskew} | "
          f"ret {r['ret']:+7.2f}%  final ${r['final']:7.2f}  fills {r['fills']:5d} "
          f"({r['fills_per_hr']:4.0f}/hr b{r['nb']}/a{r['na']})  maxDD {r['maxdd']:6.1f}%  endInv ${r['endinv_usd']:+.2f}")
