"""Survey candidate Bybit spot pairs for maker-strategy suitability.
For each pair (one day, 10-level book @100ms) measure the quantities that decide
whether a passive, imbalance-skewed maker can make money:
  - price, tick size in bps (large tick => strong queue-imbalance signal, wide spread to earn)
  - median spread in bps and in ticks
  - how often the mid moves between 100ms samples (need some, not too much)
  - micro mean-reversion: lag-1 autocorr of 100ms mid returns (negative => bid/ask bounce to harvest)
  - imbalance -> next-move accuracy (the static signal's raw edge)
  - toxicity split: P(order run over | imbalance against) vs (imbalance with) -- can we cancel the bad fills?
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lobparse import parse

def imb(bq, aq, n=1):
    b = bq[:, :n].sum(1); a = aq[:, :n].sum(1)
    return (b - a) / (b + a + 1e-12)

def survey(npz):
    z = np.load(npz)
    bp, bq, ap, aq = z['bp'], z['bq'], z['ap'], z['aq']
    mid = (bp[:, 0] + ap[:, 0]) / 2
    px = np.median(mid)
    ticks = np.diff(np.unique(np.round(ap[:, 0], 12)))
    tick = np.median(ticks[ticks > 0])
    tick_bps = tick / px * 1e4
    spread = (ap[:, 0] - bp[:, 0])
    spread_bps = np.median(spread / mid * 1e4)
    spread_ticks = np.median(spread / tick)
    moved = np.mean(np.diff(mid) != 0) * 100
    r = np.diff(np.log(mid))
    r = r[r != 0] if (r != 0).any() else r
    # lag-1 autocorr of raw 100ms log-returns (include zeros: microstructure)
    rr = np.diff(np.log(mid))
    ac1 = np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 10 else np.nan
    # imbalance -> next mid move
    I = imb(bq, aq, 1)
    N = len(mid); nxt = np.zeros(N); last = 0
    for t in range(N - 2, -1, -1):
        if mid[t + 1] != mid[t]:
            last = np.sign(mid[t + 1] - mid[t])
        nxt[t] = last
    ok = (nxt != 0) & (I != 0)
    acc = np.mean((I[ok] > 0) == (nxt[ok] > 0)) * 100
    # confident-only
    conf = np.abs(I) > 0.5
    okc = ok & conf
    accc = np.mean((I[okc] > 0) == (nxt[okc] > 0)) * 100 if okc.any() else np.nan
    # toxicity: resting bid at bp[t]; is it run over (best bid falls below it) within 30 samples (3s)?
    H = 30
    runover_bid = np.zeros(N, bool)
    for t in range(N - H):
        runover_bid[t] = (bp[t + 1:t + 1 + H, 0] < bp[t, 0] - 1e-12).any()
    bad = I < -0.5   # sellers dominant -> bid in danger
    good = I > 0.5
    tox_bad = np.mean(runover_bid[:N - H][bad[:N - H]]) * 100 if bad[:N - H].any() else np.nan
    tox_good = np.mean(runover_bid[:N - H][good[:N - H]]) * 100 if good[:N - H].any() else np.nan
    return dict(px=px, tick_bps=tick_bps, spread_bps=spread_bps, spread_ticks=spread_ticks,
                moved=moved, ac1=ac1, acc=acc, accc=accc, tox_bad=tox_bad, tox_good=tox_good, n=N)

if __name__ == '__main__':
    hdr = f"{'pair':12} {'price':>10} {'tick_bps':>8} {'spr_bps':>7} {'spr_tk':>6} {'mv%':>5} {'ac1':>7} {'imb%':>5} {'imbC%':>5} {'toxBad':>6} {'toxGood':>7}"
    print(hdr); print('-' * len(hdr))
    for arg in sys.argv[1:]:
        name = arg.split('/')[-1].replace('.npz', '')
        s = survey(arg)
        print(f"{name:12} {s['px']:>10.6g} {s['tick_bps']:>8.2f} {s['spread_bps']:>7.2f} {s['spread_ticks']:>6.2f} "
              f"{s['moved']:>5.1f} {s['ac1']:>7.3f} {s['acc']:>5.1f} {s['accc']:>5.1f} {s['tox_bad']:>6.1f} {s['tox_good']:>7.1f}")
