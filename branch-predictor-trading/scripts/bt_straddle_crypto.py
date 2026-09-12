"""Short ATM straddle on BTC and ETH, priced off Deribit's real DVOL index.

A short straddle sells a call and a put at the same (at-the-money) strike and
collects both premiums, profiting when the realized move stays inside the
breakevens (premium decay / short vol). Its entire P&L is the spread between the
IMPLIED vol you sell and the REALIZED vol that shows up, so you cannot backtest
it without real option prices. This uses Deribit's DVOL, the exchange's own
30-day implied-vol index for BTC and ETH (real market data, ~5.5 years), to
price the straddle, and settles it against the real price path.

Approximations (stated so they are not hidden):
  - ATM straddle only (no skew, no wings); priced with Black-Scholes at r=0,
    which is standard for crypto and exact enough for at-the-money.
  - DVOL is a 30-day constant-maturity index, so the 30-day roll matches its
    tenor exactly; shorter tenors reuse the 30-day number and are only indicative.
  - Deribit options are European and cash-settled, so held-to-expiry settlement
    is just the intrinsic |S_T - K|, no assignment/pin risk to model.
  - Fee ~9 bps of underlying notional per cycle (2 legs in, settlement out),
    Deribit's option taker + delivery schedule, rounded.

Usage: python3 scripts/bt_straddle_crypto.py [--dte 30] [--stop 2.0]
"""
import sys, math, numpy as np, pandas as pd
import dataio

FEE_FRAC = 0.0009            # ~9 bps of notional per full cycle (in + settle)
SQRT_YEAR = math.sqrt(365.0)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_straddle(S, K, sigma, T):
    """Black-Scholes call+put value at r=0. sigma annualized, T in years."""
    if T <= 0 or sigma <= 0:
        return abs(S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * _ncdf(d1) - K * _ncdf(d2)
    put = K * _ncdf(-d2) - S * _ncdf(-d1)
    return call + put


def run(currency, dte, stop_mult):
    px = dataio.binance_daily('BTCUSDT' if currency == 'BTC' else 'ETHUSDT')['close']
    dv = dataio.dvol_daily(currency)
    df = pd.concat([px.rename('S'), dv.rename('iv')], axis=1).dropna().sort_index()
    S = df['S'].values; IV = df['iv'].values / 100.0; idx = df.index
    N = len(S); T = dte / 365.0
    logret = np.diff(np.log(S))

    cycles = []            # (entry_date, premium_frac, iv0, rv, pnl_hold, pnl_stop)
    t0 = 0
    while t0 + dte < N:
        K = S[t0]; sig0 = IV[t0]
        prem = bs_straddle(S[t0], K, sig0, T) / K              # premium as frac of notional
        # realized vol over the cycle (annualized) and terminal intrinsic
        rv = logret[t0:t0 + dte].std() * SQRT_YEAR
        intrinsic = abs(S[t0 + dte] - K) / K
        pnl_hold = prem - intrinsic - FEE_FRAC
        # stop-managed: close early if the mark-to-market loss exceeds stop_mult * premium
        pnl_stop = None
        for u in range(t0 + 1, t0 + dte):
            Tr = (dte - (u - t0)) / 365.0
            v = bs_straddle(S[u], K, IV[u], Tr) / K            # current straddle value, frac
            if (v - prem) >= stop_mult * prem:                 # loss breached the stop
                pnl_stop = prem - v - FEE_FRAC * 1.5
                break
        if pnl_stop is None:
            pnl_stop = pnl_hold
        cycles.append((idx[t0], prem, sig0, rv, pnl_hold, pnl_stop))
        t0 += dte

    C = pd.DataFrame(cycles, columns=['date', 'prem', 'iv0', 'rv', 'pnl_hold', 'pnl_stop']).set_index('date')
    per_yr = 365.0 / dte
    return C, per_yr


def summ(C, per_yr, col):
    r = C[col].values
    total = r.sum() * 100                              # additive P&L in units of notional
    mean = r.mean() * 100
    sh = r.mean() / r.std() * math.sqrt(per_yr) if r.std() > 0 else 0
    ann = r.mean() * per_yr * 100
    win = np.mean(r > 0) * 100
    worst = r.min() * 100
    # drawdown on the additive cumulative curve
    eq = np.cumsum(r); dd = (eq - np.maximum.accumulate(eq)).min() * 100
    return dict(n=len(r), ann=ann, sh=sh, win=win, mean=mean, worst=worst, total=total, dd=dd)


def main():
    def opt(n, d): return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    dte = int(opt('--dte', '30')); stop_mult = float(opt('--stop', '2.0'))
    print("SHORT ATM STRADDLE on BTC and ETH, premium priced off real Deribit DVOL.")
    print("scripts/bt_straddle_crypto.py . Sell call+put ATM, collect premium, settle vs the")
    print(f"real move. {dte}-day cycles, held to expiry; a stop-managed variant closes if the")
    print(f"loss exceeds {stop_mult:.0f}x the premium. P&L is in units of underlying notional.\n")
    print(f"  {'':6}{'cyc':>5}{'meanIV':>8}{'meanRV':>8}{'IV-RV':>7}  | "
          f"{'variant':<12}{'ann%':>7}{'Sharpe':>8}{'win%':>6}{'avg%':>7}{'worst%':>8}{'maxDD%':>8}{'total%':>8}")
    for cur in ['BTC', 'ETH']:
        C, per_yr = run(cur, dte, stop_mult)
        iv = C['iv0'].mean() * 100; rv = C['rv'].mean() * 100
        for col, tag in [('pnl_hold', 'held-to-exp'), ('pnl_stop', f'stop {stop_mult:.0f}x')]:
            s = summ(C, per_yr, col)
            head = f"  {cur:6}{s['n']:>5}{iv:>7.1f}%{rv:>7.1f}%{iv-rv:>+6.1f}" if col == 'pnl_hold' else " " * 35
            print(f"{head}  | {tag:<12}{s['ann']:>6.1f}%{s['sh']:>8.2f}{s['win']:>5.0f}%"
                  f"{s['mean']:>+6.2f}%{s['worst']:>+7.1f}%{s['dd']:>7.1f}%{s['total']:>+7.0f}%")
    # per-year for BTC held-to-expiry
    print("\n  per-year P&L (units of notional), held-to-expiry:")
    for cur in ['BTC', 'ETH']:
        C, per_yr = run(cur, dte, stop_mult)
        py = C['pnl_hold'].groupby(C.index.year).sum() * 100
        print(f"    {cur}:  " + "  ".join(f"{y}: {py[y]:+.0f}%" for y in py.index))
    print("\n  Read: mean IV above mean RV is the vol risk premium the short straddle harvests;")
    print("  a negative worst-cycle shows the fat left tail (a big move overwhelms the premium).")


if __name__ == '__main__':
    main()
