"""Generate the two headline charts.
  1. slowbars $100 out-of-sample equity curve on SOL 8h: hold vs confirm3 vs logit.
  2. maker fill-model reality: book-only queue-reactive fantasy vs honest trade-driven
     fills, per pair, at zero fee (the whole edge lives in the gap).
Usage: python3 charts.py <klines_dir> <lob_data_dir> <out_dir>
  klines_dir: has <SYM>_1h.csv (Binance 1h)
  lob_data_dir: has <SYM>_<DATE>.npz (Bybit L2) and <SYM>_spot_<DATE>.csv (Bybit spot trades)
"""
import sys, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slowbars import run as slowrun
from marketmaker import run as bookrun
from mmtrades import run as traderun

KL, LOB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DATE = sys.argv[4] if len(sys.argv) > 4 else '2025-12-30'
os.makedirs(OUT, exist_ok=True)

# ---- chart 1: SOL 8h $100 equity, OOS ----
out, bpy, N, oos0 = slowrun(f'{KL}/SOLUSDT_1h.csv', 8, 7.5)
fig, ax = plt.subplots(figsize=(9, 5.2))
for key, lab, col in [('hold_oos', 'buy & hold', '#888'),
                      ('confirm3_oos', '3-bar confirmation (static, no ML)', '#0a7'),
                      ('logit_oos', 'static logistic on returns', '#c33')]:
    ax.plot(out[key], label=f"{lab}  ->  ${out[key][-1]:,.0f}", color=col, lw=1.8)
ax.axhline(100, color='k', lw=0.6, ls=':')
ax.set_title('SOL 8h bars, $100 start, out-of-sample half (2021-2025), 7.5 bps/side')
ax.set_ylabel('account value ($)'); ax.set_xlabel('8-hour bars')
ax.legend(loc='upper left', frameon=False); ax.set_yscale('log')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/sol_8h_100usd.png', dpi=110); plt.close(fig)
print('wrote sol_8h_100usd.png')

# ---- chart 2: maker fill-model reality per pair ----
pairs = ['WIFUSDT', 'FLOKIUSDT', 'DOGEUSDT', 'XRPUSDT', 'BTCUSDT']
book0, trade0 = [], []
for P in pairs:
    npz = f'{LOB}/{P}_{DATE}.npz'; tr = f'{LOB}/{P}_spot_{DATE}.csv'
    book0.append(bookrun(npz, fee_bps=0.0, imb_thr=0.5, fillmode='queue')['ret'])
    trade0.append(traderun(npz, tr, fee_bps=0.0, imb_thr=0.5)['ret'])
x = np.arange(len(pairs)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.bar(x - w / 2, book0, w, label='book-only queue model (counts cancels as fills)', color='#c33')
ax.bar(x + w / 2, trade0, w, label='honest fills from real trades', color='#0a7')
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([p.replace('USDT', '') for p in pairs])
ax.set_ylabel('one-day return at ZERO fee (%)')
ax.set_title('Passive maker: the "edge" is an artifact of the fill model')
ax.legend(loc='upper right', frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/maker_fill_reality.png', dpi=110); plt.close(fig)
print('wrote maker_fill_reality.png')
