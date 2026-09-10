"""Range / squeeze breakout, ported from the forex version to crypto and stocks.

The classic forex range-breakout idea: price coils in a tight range, then breaks
out; you enter in the breakout direction, put a stop the other side, and ride it
until it stalls. This is the same rule on daily bars for a crypto basket and a US
stock basket, on real OHLC, net of fees.

Rules (per asset, event-driven, no look-ahead):
  range top/bottom = highest high / lowest low over the prior `lookback` bars
    (shifted one bar, so today's bar is not in its own range).
  squeeze filter   = only take a breakout when the range width (top-bottom)/close
    sits in the bottom `squeeze_q` of its own trailing history, i.e. the range was
    unusually tight (a coil). squeeze_q=1.0 disables the filter.
  entry  = buy-stop at the range top (sell-stop at the bottom if shorts allowed);
    a gap through the level fills at the open.
  stop   = entry -/+ atr_mult * ATR(atr_n).
  exit   = stop hit intrabar, OR close beyond the opposite `exit_lb` channel
    (a trailing Donchian exit), OR a `max_hold`-bar time stop.
  fee    = fee_bps per side, charged on entry and exit.
Portfolio = equal weight across the basket (each name trades independently).

Usage: python3 scripts/bt_rangebreak.py [--lookback 20] [--exit 10] [--atr 14]
         [--atrmult 2.0] [--squeeze 0.5] [--maxhold 60] [--shorts]
"""
import sys, numpy as np, pandas as pd
import dataio


def atr_wilder(H, L, C, n):
    prev = np.roll(C, 1); prev[0] = C[0]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev), np.abs(L - prev)))
    out = np.full(len(tr), np.nan)
    if len(tr) <= n:
        return out
    out[n] = tr[1:n + 1].mean()
    for t in range(n + 1, len(tr)):
        out[t] = (out[t - 1] * (n - 1) + tr[t]) / n
    return out


def backtest_asset(O, H, L, C, lookback, exit_lb, atr_n, atr_mult,
                   squeeze_q, max_hold, fee_bps, allow_short):
    N = len(C); fee = fee_bps / 1e4
    s = pd.Series
    ph = s(H).rolling(lookback).max().shift(1).values      # range top (prior bars)
    pl = s(L).rolling(lookback).min().shift(1).values      # range bottom
    xh = s(H).rolling(exit_lb).max().shift(1).values       # trailing exit channel
    xl = s(L).rolling(exit_lb).min().shift(1).values
    atr = atr_wilder(H, L, C, atr_n)
    width = (ph - pl) / C
    wq = s(width).rolling(100, min_periods=30).quantile(squeeze_q).values
    squeeze_ok = (width <= wq) if squeeze_q < 1.0 else np.ones(N, bool)

    ret = np.zeros(N)
    pos = 0; entry = stop = 0.0; held = 0
    trades = []; exposure = 0
    for t in range(1, N):
        if np.isnan(ph[t]) or np.isnan(atr[t]):
            continue
        if pos == 0:
            took = None
            if squeeze_ok[t] and H[t] >= ph[t]:                    # break up -> long
                entry = O[t] if O[t] >= ph[t] else ph[t]
                pos, stop, held, took = 1, entry - atr_mult * atr[t], 0, 'L'
            elif allow_short and squeeze_ok[t] and L[t] <= pl[t]:   # break down -> short
                entry = O[t] if O[t] <= pl[t] else pl[t]
                pos, stop, held, took = -1, entry + atr_mult * atr[t], 0, 'S'
            if took:
                # did the same bar also stop us out?
                exited, xp = _check_exit(pos, O[t], H[t], L[t], C[t], stop, xh[t], xl[t], held, max_hold)
                base = entry
                if exited:
                    ret[t] = pos * (xp / base - 1) - 2 * fee
                    trades.append(pos * (xp / entry - 1) - 2 * fee); pos = 0
                else:
                    ret[t] = pos * (C[t] / base - 1) - fee
                exposure += 1
            continue
        # already in a position
        held += 1
        exited, xp = _check_exit(pos, O[t], H[t], L[t], C[t], stop, xh[t], xl[t], held, max_hold)
        if exited:
            ret[t] = pos * (xp / C[t - 1] - 1) - fee
            trades.append(_trade_ret(pos, entry, xp, fee)); pos = 0
        else:
            ret[t] = pos * (C[t] / C[t - 1] - 1)
        exposure += 1
    return ret, trades, exposure


def _check_exit(pos, O, H, L, C, stop, xh, xl, held, max_hold):
    if pos == 1:
        if L <= stop:
            return True, min(O, stop) if O < stop else stop
        if held >= max_hold or (not np.isnan(xl) and C <= xl):
            return True, C
    else:
        if H >= stop:
            return True, max(O, stop) if O > stop else stop
        if held >= max_hold or (not np.isnan(xh) and C >= xh):
            return True, C
    return False, 0.0


def _trade_ret(pos, entry, exitp, fee):
    return pos * (exitp / entry - 1) - 2 * fee


def stats(ret, bpy, label):
    ret = np.asarray(ret); eq = np.cumprod(1 + ret)
    n = len(ret); cagr = eq[-1] ** (bpy / n) - 1 if n else 0
    sd = ret.std(); sh = ret.mean() / sd * np.sqrt(bpy) if sd > 0 else 0
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(label=label, cagr=cagr * 100, sharpe=sh, maxdd=dd * 100)


def portfolio(panel, fee_bps, P):
    rets = []; ntr = 0; wins = 0; expo = 0; bars = 0; holds = []
    for sym, df in panel.items():
        O, H, L, C = (df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        r, tr, ex = backtest_asset(O, H, L, C, P['lookback'], P['exit'], P['atr'],
                                   P['atrmult'], P['squeeze'], P['maxhold'], fee_bps, P['shorts'])
        rets.append(pd.Series(r, index=df.index)); ntr += len(tr)
        wins += sum(1 for x in tr if x > 0); holds += tr; expo += ex; bars += len(r)
    R = pd.concat(rets, axis=1).fillna(0.0)
    port = R.mean(axis=1)                              # equal weight, daily rebalanced
    hold = pd.concat([df['close'].pct_change() for df in panel.values()], axis=1).mean(axis=1).fillna(0.0)
    info = dict(n=R.shape[1], ntr=ntr, win=wins / ntr * 100 if ntr else 0,
                avg=np.mean(holds) * 100 if holds else 0, expo=expo / bars * 100 if bars else 0)
    return port, hold, info


def run_market(name, panel, bpy, fee_bps, P):
    port, hold, info = portfolio(panel, fee_bps, P)
    st = stats(port.values, bpy, 'range breakout'); hd = stats(hold.values, bpy, 'buy & hold')
    wk = (1 + port).groupby(pd.Grouper(freq='W')).prod() - 1
    print(f"\n=== {name} | range breakout {'(long/short)' if P['shorts'] else '(long only)'} | "
          f"{info['n']} names | {port.index.min().date()}..{port.index.max().date()} | fee {fee_bps} bps/side ===")
    print(f"  params: lookback {P['lookback']}  exit {P['exit']}  ATR {P['atr']}x{P['atrmult']}  "
          f"squeeze q{P['squeeze']}{' (off)' if P['squeeze'] >= 1 else ''}  maxhold {P['maxhold']}")
    print(f"  {st['label']:<16} CAGR {st['cagr']:>7.1f}%  Sharpe {st['sharpe']:>5.2f}  maxDD {st['maxdd']:>7.1f}%")
    print(f"  {hd['label']:<16} CAGR {hd['cagr']:>7.1f}%  Sharpe {hd['sharpe']:>5.2f}  maxDD {hd['maxdd']:>7.1f}%")
    print(f"  trades {info['ntr']}  win {info['win']:.0f}%  avg/trade {info['avg']:+.2f}%  "
          f"time in market {info['expo']:.0f}%  weekly+ {np.mean(wk > 0) * 100:.0f}%")
    # per-year net (strategy / hold)
    py = (1 + port).groupby(port.index.year).prod() - 1
    ph = (1 + hold).groupby(hold.index.year).prod() - 1
    print("  per-year net (breakout / hold): " +
          "  ".join(f"{y}: {py[y]*100:+.0f}%/{ph[y]*100:+.0f}%" for y in py.index))
    return st, hd


def sweep_line(panel, bpy, fee_bps, P):
    port, hold, info = portfolio(panel, fee_bps, P)
    st = stats(port.values, bpy, '')
    return st, info


def main():
    def opt(n, d): return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    P = dict(lookback=int(opt('--lookback', '20')), exit=int(opt('--exit', '10')),
             atr=int(opt('--atr', '14')), atrmult=float(opt('--atrmult', '2.0')),
             squeeze=float(opt('--squeeze', '1.0')), maxhold=int(opt('--maxhold', '60')),
             shorts=('--shorts' in sys.argv))
    crypto = {s.replace('USDT', ''): dataio.binance_daily(s) for s in dataio.CRYPTO}
    stocks = {t: dataio.yahoo_daily(t) for t in dataio.STOCKS}
    markets = [('CRYPTO basket', crypto, 365, 10), ('US STOCKS basket', stocks, 252, 2)]
    print("RANGE BREAKOUT ported from forex to crypto and stocks (real OHLC, net of fees).")
    print("scripts/bt_rangebreak.py . Donchian range breakout, ATR stop, trailing-channel exit.")
    print("Default is the PLAIN breakout (no squeeze filter); see the squeeze comparison below.\n")
    print("########## headline: plain breakout, long only ##########")
    for name, panel, bpy, fee in markets:
        run_market(name, panel, bpy, fee, P)

    print("\n########## squeeze filter on vs off (does the forex 'coil' filter transfer?) ##########")
    print(f"  {'market':<18}{'squeeze':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}")
    for name, panel, bpy, fee in markets:
        for sq in [0.3, 0.5, 1.0]:
            st, info = sweep_line(panel, bpy, fee, dict(P, squeeze=sq))
            tag = f"q{sq}" + (" off" if sq >= 1 else "")
            print(f"  {name:<18}{tag:>9}{st['cagr']:>7.0f}%{st['sharpe']:>8.2f}{st['maxdd']:>7.0f}%{info['ntr']:>8}")

    print("\n########## long only vs long/short ##########")
    print(f"  {'market':<18}{'side':>11}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
    for name, panel, bpy, fee in markets:
        for sh, tag in [(False, 'long only'), (True, 'long/short')]:
            st, _ = sweep_line(panel, bpy, fee, dict(P, shorts=sh))
            print(f"  {name:<18}{tag:>11}{st['cagr']:>7.0f}%{st['sharpe']:>8.2f}{st['maxdd']:>7.0f}%")

    print("\n########## lookback robustness (plain, long only) ##########")
    print(f"  {'market':<18}{'lookback':>10}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
    for name, panel, bpy, fee in markets:
        for lb, ex in [(10, 5), (20, 10), (35, 15), (55, 20)]:
            st, _ = sweep_line(panel, bpy, fee, dict(P, lookback=lb, exit=ex))
            print(f"  {name:<18}{lb:>10}{st['cagr']:>7.0f}%{st['sharpe']:>8.2f}{st['maxdd']:>7.0f}%")


if __name__ == '__main__':
    main()
