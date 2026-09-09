#!/usr/bin/env python3
"""Branch-predictor trading on 1-second crypto bars.

Idea under test: treat each tick's direction like a CPU branch outcome.
  * 1-bit predictor : go the way the last tick went, full size.
  * ramp(N)         : saturating counter in [-N, N]. Each up tick adds one
                      step, each down tick removes one. Position = counter/N,
                      so money is added "bit by bit" instead of all at once.
  * streak(N)       : keep adding a step while the direction repeats, and
                      reset to one step in the new direction when the loop
                      breaks (the "do the same until something breaks it" rule).
  * long-only ramp  : same counter clipped to [0, N] (never short).

Position for bar t is decided from bars <= t-1 (no lookahead) and earns the
close-to-close return of bar t. Costs are charged per unit of turnover.

Usage: python3 sim.py --data DIR [--symbols BTCUSDT,ETHUSDT] [--intervals 1,5,15,60]
DIR holds Binance kline CSVs named SYMBOL-1s-YYYY-MM-DD.csv, downloadable from
https://data.binance.vision/data/spot/daily/klines/SYMBOL/1s/SYMBOL-1s-YYYY-MM-DD.zip
"""
import argparse, glob, math, os, sys
import numpy as np


def load_closes(data_dir, symbol):
    files = sorted(glob.glob(os.path.join(data_dir, f"{symbol}-1s-*.csv")))
    if not files:
        sys.exit(f"no files for {symbol} in {data_dir}")
    closes = []
    for f in files:
        a = np.loadtxt(f, delimiter=",", usecols=(4,), dtype=np.float64)
        closes.append(a)
    return np.concatenate(closes), len(files)


def resample(px, k):
    return px[::k] if k > 1 else px


def positions(direction, strat, n):
    """direction: array of -1/0/+1 per bar. Returns position array (fraction of capital)
    where pos[t] is held during bar t, decided from direction[t-1]."""
    T = len(direction)
    pos = np.zeros(T)
    if strat == "1bit":
        d = direction[:-1].copy()
        # a flat tick keeps the previous prediction
        for t in range(1, len(d)):
            if d[t] == 0:
                d[t] = d[t - 1]
        pos[1:] = d
        return pos
    c = 0
    if strat == "ramp":            # saturating counter, long-short
        lo, hi = -n, n
        for t in range(1, T):
            c = min(hi, max(lo, c + direction[t - 1]))
            pos[t] = c / n
    elif strat == "ramp_long":     # saturating counter, long only
        for t in range(1, T):
            c = min(n, max(0, c + direction[t - 1]))
            pos[t] = c / n
    elif strat == "streak":        # grow while direction repeats, reset on break
        d, k = 0, 0
        for t in range(1, T):
            x = direction[t - 1]
            if x == 0:
                pass
            elif x == d:
                k = min(n, k + 1)
            else:
                d, k = x, 1
            pos[t] = d * k / n
    else:
        raise ValueError(strat)
    return pos


def evaluate(px, pos, cost_bps_list):
    ret = np.zeros(len(px))
    ret[1:] = px[1:] / px[:-1] - 1
    turnover = np.abs(np.diff(np.r_[0.0, pos])).sum()
    gross = (pos * ret).sum()
    direction = np.sign(ret)
    live = (direction != 0) & (pos != 0)
    hit = np.mean(np.sign(pos[live]) == direction[live]) if live.any() else float("nan")
    breakeven_bps = gross / turnover * 1e4 if turnover > 0 else float("nan")
    nets = {c: gross - turnover * c / 1e4 for c in cost_bps_list}
    return dict(gross=gross, turnover=turnover, hit=hit, breakeven_bps=breakeven_bps, nets=nets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT")
    ap.add_argument("--intervals", default="1,5,15,60")
    ap.add_argument("--levels", default="2,5,10,20")
    ap.add_argument("--costs", default="0,1,2,5,10")
    ap.add_argument("--min-move-bps", type=float, default=0.0,
                    help="moves smaller than this count as flat (filters 1-tick bid/ask bounce)")
    a = ap.parse_args()
    costs = [float(c) for c in a.costs.split(",")]
    levels = [int(x) for x in a.levels.split(",")]

    hdr = f"{'strategy':14}{'hit%':>7}{'gross%':>9}{'turn(x)':>9}{'b/e bps':>9} |" + "".join(f"{'net@'+str(int(c)):>9}" for c in costs)
    for sym in a.symbols.split(","):
        px1, ndays = load_closes(a.data, sym)
        for k in [int(x) for x in a.intervals.split(",")]:
            px = resample(px1, k)
            ret = px[1:] / px[:-1] - 1
            direction = np.sign(np.diff(np.r_[px[0], px]))
            if a.min_move_bps > 0:
                small = np.abs(np.r_[0.0, ret]) * 1e4 < a.min_move_bps
                direction[small] = 0
            flat = np.mean(direction[1:] == 0)
            nz = ret[ret != 0]
            ac = np.corrcoef(ret[1:], ret[:-1])[0, 1]
            bh = px[-1] / px[0] - 1
            print(f"\n=== {sym} @ {k}s bars: {len(px):,} bars over {ndays} days | flat ticks {flat*100:.1f}% "
                  f"| median |move| {np.median(np.abs(nz))*1e4:.4f} bps | lag1 autocorr {ac:+.3f} | buy&hold {bh*100:+.2f}%")
            print(hdr)
            runs = [("1bit", 0)] + [(s, n) for n in levels for s in ("ramp", "ramp_long", "streak")]
            for strat, n in runs:
                pos = positions(direction, strat, n)
                r = evaluate(px, pos, costs)
                name = strat if n == 0 else f"{strat}({n})"
                print(f"{name:14}{r['hit']*100:7.1f}{r['gross']*100:9.2f}{r['turnover']:9.0f}{r['breakeven_bps']:9.2f} |"
                      + "".join(f"{r['nets'][c]*100:9.1f}" for c in costs))


if __name__ == "__main__":
    main()
