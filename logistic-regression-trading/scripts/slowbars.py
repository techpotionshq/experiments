"""Slow-bar static strategies on crypto: where the money actually is for a $100
retail account. The microstructure logistic is unmonetizable (see maker results);
here the SAME idea (a static logistic on past returns) is applied to slow bars
where the per-trade move clears retail fees.

Long-only (crypto drifts up; the short leg loses, per prior work). Three rules:
  hold     : buy and hold (benchmark)
  confirm3 : long after 3 consecutive up bars, flat after 3 consecutive down bars
  logit    : static logistic on [r1,r2,r3,mom6,vol6] -> P(next bar up); long if >0.5.
             Coefficients FIT ONCE on the first half, FROZEN, applied to the second
             half (true out-of-sample). Never re-fit online -> static.
$100 start, fee charged on turnover each time the position changes.
"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def fit_logit(Xtr, ytr):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)).fit(Xtr, ytr)


def load_1h(path):
    # Binance kline csv: open_time,open,high,low,close,volume,close_time,...
    df = pd.read_csv(path, header=None, usecols=[0, 4], names=['t', 'close'])
    df = df.sort_values('t').drop_duplicates('t')
    return df['t'].to_numpy(), df['close'].to_numpy(dtype=float)


def resample(t, c, k):
    # bucket every k hourly bars, take the last close of each bucket
    n = (len(c) // k) * k
    return t[k - 1:n:k], c[k - 1:n:k]


def metrics(equity, bars_per_year):
    eq = np.asarray(equity)
    rets = np.diff(eq) / eq[:-1]
    yrs = len(eq) / bars_per_year
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if eq[-1] > 0 else -1
    peak = np.maximum.accumulate(eq); dd = (eq / peak - 1).min()
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(bars_per_year) if rets.std() > 0 else 0
    return cagr * 100, dd * 100, sharpe


def backtest(pos, ret, fee_bps, start=100.0):
    # pos[i] applied to ret[i] (ret[i] = close[i+1]/close[i]-1); charge fee on |pos change|
    fee = fee_bps / 1e4
    eq = [start]; p_prev = 0.0
    for i in range(len(ret)):
        turn = abs(pos[i] - p_prev)
        e = eq[-1] * (1 - turn * fee) * (1 + pos[i] * ret[i])
        eq.append(e); p_prev = pos[i]
    return np.array(eq)


def run(path, k, fee_bps, hours_per_bar=None):
    t, c = load_1h(path)
    tb, cb = resample(t, c, k)
    logret = np.diff(np.log(cb))
    ret = cb[1:] / cb[:-1] - 1                     # forward simple return per bar
    N = len(ret)
    bpy = 365 * 24 / k                             # bars per year

    # features at bar i (known at close i), predict sign(ret[i])
    def feat(i):
        r1 = logret[i - 1]; r2 = logret[i - 2]; r3 = logret[i - 3]
        mom6 = logret[i - 6:i].sum(); vol6 = logret[i - 6:i].std() + 1e-9
        return [r1, r2, r3, mom6, vol6]
    idx = np.arange(7, N)                           # need 6 bars of history, and ret[i] exists
    X = np.array([feat(i) for i in idx]); y = (ret[idx] > 0).astype(int)

    out = {}
    # buy & hold
    out['hold'] = backtest(np.ones(N), ret, 0.0)
    # confirm3: long after 3 up bars, flat after 3 down bars
    pos = np.zeros(N); state = 0
    for i in range(3, N):
        up3 = logret[i - 1] > 0 and logret[i - 2] > 0 and logret[i - 3] > 0
        dn3 = logret[i - 1] < 0 and logret[i - 2] < 0 and logret[i - 3] < 0
        if up3: state = 1
        elif dn3: state = 0
        pos[i] = state
    out['confirm3'] = backtest(pos, ret, fee_bps)
    # logit: fit on first half of idx, apply out-of-sample on second half
    cut = len(idx) // 2
    lr = fit_logit(X[:cut], y[:cut])
    pos_l = np.zeros(N)
    p = lr.predict_proba(X[cut:])[:, 1]
    for j, i in enumerate(idx[cut:]):
        pos_l[i] = 1.0 if p[j] > 0.5 else 0.0

    # logit v2: REGIME features (trend filter) instead of raw last-bar returns.
    # target = up over next 3 bars (smoother, more persistent than single-bar sign).
    csum = np.concatenate([[0.0], np.cumsum(logret)])   # csum[i] = sum logret[:i]
    sma = lambda w: np.array([cb[max(0, i - w + 1):i + 1].mean() for i in range(len(cb))])
    s10, s30 = sma(10), sma(30)

    def feat2(i):                                       # known at close of bar i (index into cb)
        fracup = (logret[i - 10:i] > 0).mean()
        return [cb[i] / (s10[i] + 1e-12) - 1, cb[i] / (s30[i] + 1e-12) - 1,
                fracup, logret[i - 6:i].sum(), logret[i - 12:i].sum(),
                logret[i - 12:i].std() + 1e-9]
    # cb has N+1 entries (len(ret)=N). bar i in cb maps to ret index i (ret[i]=cb[i+1]/cb[i]-1)
    idx2 = np.arange(30, N - 3)
    X2 = np.array([feat2(i) for i in idx2])
    fwd3 = np.array([(cb[i + 3 + 1] / cb[i + 1] - 1) > 0 for i in idx2]).astype(int)  # up over next 3 bars
    cut2 = len(idx2) // 2
    lr2 = fit_logit(X2[:cut2], fwd3[:cut2])
    p2 = lr2.predict_proba(X2[cut2:])[:, 1]
    pos_l2 = np.zeros(N)
    for j, i in enumerate(idx2[cut2:]):
        pos_l2[i] = 1.0 if p2[j] > 0.55 else 0.0        # confidence gate to cut whipsaw
    # for a fair comparison restrict all curves to the OOS window
    oos0 = idx2[cut2]                       # common OOS start (v2 needs more warmup)
    out['logit_oos'] = backtest(pos_l[oos0:], ret[oos0:], fee_bps)
    out['logit2_oos'] = backtest(pos_l2[oos0:], ret[oos0:], fee_bps)
    out['hold_oos'] = backtest(np.ones(N - oos0), ret[oos0:], 0.0)
    out['confirm3_oos'] = backtest(pos[oos0:], ret[oos0:], fee_bps)
    return out, bpy, N, oos0


if __name__ == '__main__':
    path = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    fee = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5
    out, bpy, N, oos0 = run(path, k, fee)
    name = path.split('/')[-1].split('_')[0]
    print(f"\n{name}  {k}h bars  fee {fee}bps  ({N} bars, full history)")
    print(f"{'strategy':14} {'final$/100':>10} {'CAGR%':>7} {'maxDD%':>7} {'Sharpe':>7}")
    for kk in ['hold', 'confirm3']:
        cg, dd, sh = metrics(out[kk], bpy)
        print(f"{kk:14} {out[kk][-1]:>10.2f} {cg:>7.1f} {dd:>7.1f} {sh:>7.2f}")
    print(f"-- out-of-sample second half only ({N - oos0} bars) --")
    for kk in ['hold_oos', 'confirm3_oos', 'logit_oos', 'logit2_oos']:
        cg, dd, sh = metrics(out[kk], bpy)
        print(f"{kk:14} {out[kk][-1]:>10.2f} {cg:>7.1f} {dd:>7.1f} {sh:>7.2f}")
