"""Static imbalance-skewed passive market maker on Bybit L2 books (10 levels @100ms).

Improves the prior 'directional taker' framing. Here the model NEVER crosses the
spread: it rests one bid and/or one ask at the current best price, earns the
spread on completed round trips, and uses the STATIC queue-imbalance signal to
decide which side to show (skew) and inventory to decide which side to reduce.

Fill model (book-only, no trade prints -> conservative queue-reactive):
  A resting bid at price p (joined with `qa` units ahead of it in the queue):
    - best bid moves BELOW p            -> level traded through -> FILLED at p (toxic: price fell onto us)
    - best bid stays AT p, size shrinks -> shrink counts as executions+cancels ahead; qa -= shrink;
                                            when qa <= 0 -> FILLED at p (benign: a market sell reached us)
    - best bid moves ABOVE p            -> we are no longer best; cancel & reprice next step (no fill)
  Symmetric for a resting ask. One unit (order_usd notional) fills per side per step.
  'through' mode ignores the queue and fills only on trade-through (pessimistic bound).

Economics: maker fee charged per fill on notional (negative fee = rebate).
Inventory marked at mid; equity = cash + inventory*mid. Starts from `start_usd`.
"""
import sys, argparse, numpy as np


def run(npz, fee_bps=0.0, imb_thr=0.5, start_usd=100.0, order_usd=25.0,
        max_inv_usd=100.0, fillmode='queue', skew=True, reprice=True, verbose=False):
    z = np.load(npz)
    bp, bq, ap, aq = z['bp'][:, 0], z['bq'][:, 0], z['ap'][:, 0], z['aq'][:, 0]
    bq3, aq3 = z['bq'][:, :3].sum(1), z['aq'][:, :3].sum(1)
    N = len(bp)
    mid = (bp + ap) / 2.0
    I = (bq3 - aq3) / (bq3 + aq3 + 1e-12)      # top-3 queue imbalance (the static signal)

    cash = start_usd
    q = 0.0                                     # inventory in base units
    unit = order_usd / mid[0]                   # base units per order (kept ~constant notional)
    qmax = max_inv_usd / mid[0]

    # live orders: price (nan = none) and queue-ahead estimate
    bid_px = np.nan; bid_qa = 0.0
    ask_px = np.nan; ask_qa = 0.0
    fee = fee_bps / 1e4
    nfill_b = nfill_a = 0
    eq = np.empty(N)

    for t in range(N - 1):
        unit = order_usd / mid[t]
        qmax = max_inv_usd / mid[t]
        # ---- desired quoting (STATIC rule) ----
        if skew:
            want_bid = (I[t] >= imb_thr or q < 0) and q < qmax
            want_ask = (I[t] <= -imb_thr or q > 0) and q > -qmax
        else:                                   # symmetric baseline: always both, inventory-capped
            want_bid = q < qmax
            want_ask = q > -qmax

        # ---- (re)place / cancel bid ----
        if want_bid:
            if np.isnan(bid_px) or (reprice and bid_px != bp[t]):
                bid_px = bp[t]; bid_qa = bq[t]   # join back of current best queue
        else:
            bid_px = np.nan
        if want_ask:
            if np.isnan(ask_px) or (reprice and ask_px != ap[t]):
                ask_px = ap[t]; ask_qa = aq[t]
        else:
            ask_px = np.nan

        # ---- fills over t -> t+1 ----
        # BID
        if not np.isnan(bid_px):
            if bp[t + 1] < bid_px - 1e-15:                       # traded through -> filled
                fill = True
            elif fillmode == 'queue' and abs(bp[t + 1] - bid_px) < 1e-15:
                shrink = max(0.0, bq[t] - bq[t + 1]); bid_qa -= shrink
                fill = bid_qa <= 0
            else:
                fill = False
            if fill:
                cash -= bid_px * unit + abs(bid_px * unit) * fee
                q += unit; nfill_b += 1; bid_px = np.nan
        # ASK
        if not np.isnan(ask_px):
            if ap[t + 1] > ask_px + 1e-15:
                fill = True
            elif fillmode == 'queue' and abs(ap[t + 1] - ask_px) < 1e-15:
                shrink = max(0.0, aq[t] - aq[t + 1]); ask_qa -= shrink
                fill = ask_qa <= 0
            else:
                fill = False
            if fill:
                cash += ask_px * unit - abs(ask_px * unit) * fee
                q -= unit; nfill_a += 1; ask_px = np.nan

        eq[t] = cash + q * mid[t]
    eq[-1] = cash + q * mid[-1]

    # liquidate remaining inventory at mid, pay taker-ish (use mid, no fee bonus) for honesty
    final = cash + q * mid[-1]
    ret = (final / start_usd - 1) * 100
    fills = nfill_b + nfill_a
    hrs = N / 10 / 3600
    maxdd = 0.0; peak = eq[0]
    for e in eq:
        peak = max(peak, e); maxdd = min(maxdd, e / peak - 1)
    return dict(ret=ret, final=final, fills=fills, fills_per_hr=fills / hrs,
                maxdd=maxdd * 100, endinv_usd=q * mid[-1], hrs=hrs,
                nb=nfill_b, na=nfill_a, eq=eq, mid=mid)


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('npz')
    ap_.add_argument('--fee', type=float, default=0.0)
    ap_.add_argument('--thr', type=float, default=0.5)
    ap_.add_argument('--order', type=float, default=25.0)
    ap_.add_argument('--maxinv', type=float, default=100.0)
    ap_.add_argument('--start', type=float, default=100.0)
    ap_.add_argument('--fillmode', default='queue')
    ap_.add_argument('--noskew', action='store_true')
    args = ap_.parse_args()
    r = run(args.npz, fee_bps=args.fee, imb_thr=args.thr, start_usd=args.start,
            order_usd=args.order, max_inv_usd=args.maxinv, fillmode=args.fillmode,
            skew=not args.noskew)
    print(f"{args.npz.split('/')[-1]:28} fee {args.fee:5.1f}bps thr {args.thr:.2f} skew {not args.noskew} "
          f"fill={args.fillmode:6} | ret {r['ret']:+7.2f}%  final ${r['final']:7.2f}  "
          f"fills {r['fills']:6d} ({r['fills_per_hr']:5.0f}/hr b{r['nb']}/a{r['na']})  "
          f"maxDD {r['maxdd']:6.1f}%  endInv ${r['endinv_usd']:+.2f}")
