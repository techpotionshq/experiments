# Branch-predictor trading: findings

Starting idea: treat each price tick like a CPU branch outcome. A 1-bit predictor
goes the way the last tick went; a 2-bit saturating counter needs two misses to
flip; "keep doing the same thing until the loop breaks" is a streak rule. Add
money one step per confirming tick instead of all at once. Test it on volatile
crypto at 1 to 5 second intervals, then on order-book data at a small venue.

Everything below was measured, not argued. Scripts are in `scripts/`, raw
outputs in `results/`, charts in `results/charts/`, and a GPU notebook in
`notebooks/`. Data is not committed; every script downloads what it needs from
public Binance and Bybit archives.

## One-paragraph verdict

The persistence the idea relies on is real at two horizons: seconds and days.
At seconds it is worth 0.1 to 0.5 basis points per trade against fees of 5 to
10 bps, and no way of acting on it (market orders, limit orders, order-book
features, a neural network) survives that gap. At days it pays: a long-only
rule on 12-hour bars with a 3-bar confirmation returns roughly what buy-and-hold
returns over six years with drawdowns 10 to 25 points shallower. Order-book
queue imbalance is the strongest predictor tested (61 to 75% on the next mid
move) and it beat DeepLOB, but it is a market maker's cancel signal, only worth
money on a venue that pays rebates.

## 1. Daily bars: the naive predictor loses

`sim.py` was first run on FRED daily closes (S&P 500, Nasdaq, Dow, EUR/USD,
USD/JPY, oil, BTC). Daily lag-1 autocorrelation is zero or negative in every
liquid market today, so the 1-bit rule loses before costs. The Nasdaq looked
profitable only because the 1971 to 1990 index carried stale prices from
non-trading stocks; the effect inverted once the market became liquid.

## 2. Seconds: the trend is real, the fee kills it

Binance spot 1-second klines, 2026-09-01 to 09-07, BTC, ETH, SOL, DOGE.

| BTC, 1s bars, 7 days | hit % | gross | turnover | break-even fee |
|---|---|---|---|---|
| 1-bit, full size | 28.1 | +255% | 458,313x capital | 0.06 bps |
| ramp, 5 steps | 51.6 | +101% | 57,303x | 0.18 bps |
| ramp, 20 steps | 51.3 | +26% | 15,236x | 0.17 bps |
| streak, 5 steps | 28.1 | +129% | 123,213x | 0.10 bps |

BTC and ETH have positive 1-second autocorrelation (+0.11, +0.09). Gross
returns look enormous. The break-even fee, gross divided by turnover, sits at
0.06 to 0.2 bps against Binance's 7.5 to 10 bps. Ramping money in bit by bit
scales gross and turnover down together and leaves the edge per trade
unchanged. SOL and DOGE have no 1-second autocorrelation and lose even at zero
fee. The effect is gone by 15-second bars.

The user's exact rule, all in after an up tick and all out after a down tick on
5-second bars, compounded from $1,000: doubles with no fee, ends under $1 after
about 38 hours at Binance's 7.5 bps, on every pair (`results/allin_5s.txt`,
`results/charts/btc_5s_allin.png`).

## 3. Which interval? 12 hours

`scripts/sweep.py`, 1 second to 1 day, using 7 days of 1s data, 2 years of 1m
data and 6 years of 1h data (`results/interval_sweep.txt`,
`results/charts/interval_sweep.png`).

| interval | BTC | ETH | SOL | DOGE | trades/yr |
|---|---|---|---|---|---|
| 1s to 1m | 0.0 to 0.2 bps | 0.0 to 0.2 | -0.2 to 0.1 | 0.0 | 120k to 6M |
| 5m to 30m | -0.1 to -0.4 | -0.1 to -0.4 | -0.2 to -0.7 | 0.0 to -0.5 | 4.5k to 27k |
| 1h | 0.2 | 0.8 | 1.6 | 0.3 | 2,300 |
| 4h | 1.1 | 3.8 | 5.0 | 5.6 | 570 |
| 8h | 3.5 | 7.1 | 4.7 | 21.8 | 285 |
| 12h | 11.5 | 16.3 | 15.0 | 37.7 | 188 |
| 1d | 10.4 | 5.0 | 28.9 | 68.1 | 96 |

Edge per trade in bps. A small positive edge under a minute that fees eat, a
dead zone from 2 to 30 minutes where prices mean-revert, then a climb from 1
hour. 12 hours is the first interval where all four coins clear a 7.5 bps fee.

## 4. Confirmation and shorting on 12h bars

`scripts/h12.py`, six years, four coins, 7.5 bps per side
(`results/h12_confirmation_longshort.txt`).

**Long-short is a disaster.** Every variant loses 15 to 80% a year with 95 to
100% drawdowns. Crypto drifts up hard; the short leg is wrong more often than
right and pays the fee either way. DOGE 3-bar long-short lost more than 100% in
one 12-hour bar in 2021.

**Long-only with 3-bar confirmation is the best thing found.**

| 12h, long-only, 3-bar confirm | net/yr | max DD | Sharpe | trades/yr | hold/yr | hold DD | hold Sharpe |
|---|---|---|---|---|---|---|---|
| BTC | +31.6% | -64% | 0.89 | 24 | +37.6% | -77% | 0.85 |
| ETH | +42.3% | -59% | 0.94 | 23 | +33.8% | -81% | 0.76 |
| SOL | +56.2% | -85% | 0.94 | 25 | +67.0% | -96% | 1.02 |
| DOGE | +60.4% | -85% | 0.95 | 24 | +72.1% | -93% | 0.90 |

Robustness (`results/robustness_confirmN.txt`): confirmation 1 to 6 at 8h,
12h and 1d is a bumpy surface. 2 bars at 12h is bad on all coins, 3 is good on
all; at 1d it flips. Split into two 3-year halves the rule beat holding on
return in 4 of 8 cases, on drawdown in 8 of 8, on Sharpe in 5 of 8. Read it as
"a slow trend filter with 1.5 to 3 days of confirmation roughly matches holding
with less pain", not as a tuned edge. It is time-series momentum.

## 5. Order flow and order book on BTC

`scripts/flow.py`: net taker flow over the last second predicts the next 5 to
60 seconds at 58 to 60% hit rate, worth 0.1 to 0.44 bps, decaying to nothing by
5 minutes (`results/flow_1s_btc.txt`).

`scripts/book.py`: Binance futures depth imbalance within 0.2% to 5% of mid,
sampled every 30 seconds, predicts nothing at 1 to 60 minutes: correlations 0.00
to 0.05, hit rates 50 to 52% (`results/bookdepth_30s_btc.txt`). The predictive
imbalance lives at the best quotes at millisecond resolution.

## 6. Limit orders do not fix it

`scripts/maker.py`: rest a limit order at the last price when 1-second flow is
extreme, hold, exit with a resting order, market-out after a wait. A resting
order counts as filled only if price trades through it.

| hold | fill rule | fill rate | price P&L per trade, zero fees | net at 0 maker / 5 taker |
|---|---|---|---|---|
| 5s | through | 42% | -0.19 bps | -1.78 bps |
| 15s | through | 42% | -0.10 bps | -1.76 bps |
| 60s | through | 41% | -0.07 bps | -1.76 bps |
| 15s | touch (optimistic) | 88% | +0.31 bps | +0.21 bps |

With honest fills the trades lose before any fee: the resting buy fills exactly
when the price dips against the signal. That is adverse selection and it
consumes the whole 0.3 bps edge. Only a maker rebate of about 1 bps per side
turns it positive on paper. No venue pays a retail account that: MEXC's 0%
maker is web/app only (API orders pay 6 to 8 bps), Binance/OKX/Bybit/Kraken
perps are 2 bps maker, Hyperliquid 1.5 bps with rebates from $5M per 14 days.

## 7. Small venue and DeepLOB

Bybit publishes free 200-level order books daily for hundreds of spot pairs
since 2025-04-29 at `quote-saver.bycsi.com/orderbook/spot/<SYMBOL>/`.
WIF/USDT was used as the small pair (tick 4.96 bps, spread equals one tick 90%
of the time, mid moves every 4 seconds) and BTC/USDT as the control.

`scripts/lobparse.py` rebuilds the book and samples every 100 ms.
`scripts/lobmodels.py` compares four predictors; `scripts/nextmove.py` asks the
large-tick question, when the mid next moves, which way
(`results/deeplob_wif_eval.txt`).

| model | 3-class macro-F1 | next-move direction | most confident 20% |
|---|---|---|---|
| queue imbalance sign | 33.4 | 61.0% | 67% at extreme imbalance |
| logistic, 9 features | 33.7 | 60.9% | 70.6% |
| gradient boosting | 35.5 | 51.8% | 64.1% |
| DeepLOB, 4 epochs on CPU | 46.0 | 56.1% | 60.3% |

On BTC the same imbalance sign hits 63.4% and logistic reaches 75.3% on its
confident fifth. DeepLOB learned the 3-class label best, as the papers report,
and was worse than a one-line imbalance rule on the question a maker cares
about. Its confident calls had a realised edge of -0.48 bps. It was
under-trained; `notebooks/deeplob_bybit_colab.ipynb` runs the same comparison
on a GPU with any Bybit pair and more epochs.

Fill conditioning on WIF: a resting bid is run over within 5 seconds 26% of the
time, 36% when imbalance is strongly negative, 15% when strongly positive.
Imbalance is a good cancel signal. Against Bybit's 10 bps spot maker fee it is
not a business; the venues that pay for it are Kraken's rebate pairs at Tier 12
(-2 bps maker, $10M/month) and Bitget's liquidity incentive program.

## 8. Venues

Live best bid/ask recorded from public websockets for 8 minutes
(`scripts/recorder.py`, `results/venue_leadlag.txt`). Bybit leads Gate and
Bitget by about 100 ms and Hyperliquid by about 800 ms; Kraken is simultaneous
with Bybit. Hyperliquid's spread is 10x the others. Binance and OKX websockets
were unreachable from the recording host.

## 9. What the literature says

Cont, Kukanov and Stoikov (order flow imbalance drives short-horizon price
changes, linearly, with slope inversely proportional to depth); Gould and
Bonart (queue imbalance predicts the next mid move, strongest for large-tick
instruments); LOBCAST benchmark of 15 deep LOB models (all degrade sharply on
new data); Briola, Bartolucci and Aste (forecasting power depends on the
asset's microstructure and does not translate into actionable signals); TLOB
(beats DeepLOB by about 1 F1 on Bitcoin, and a plain MLP matches it). Every
number in this repo agrees with them.

What quants do with the signal: market makers post passive quotes and use
imbalance to skew and cancel them, earning spread plus rebates on $5M to $50M
of capital with colocation. Retail-scale quants run funding-rate carry (10 to
30% a year quoted), basis trades (5 to 15%), being fastest in a small market,
and slow trend rules like section 4.

## Ranking

1. Long-only, 12h bars, 3-bar confirmation. Net positive at retail fees. Matches
   holding with shallower drawdowns. Not a tuned edge.
2. Logistic regression on queue imbalance and order-flow features for the next
   mid move. Strongest predictor per line of code. Pays only as a maker with
   rebates.
3. Everything fast and taker: real signals, each smaller than the cost of acting
   on them.

## Reproduce

```
# daily and 1-second experiments
python3 sim.py --data <dir of Binance 1s kline CSVs> --symbols BTCUSDT --intervals 1,5,15,60
python3 scripts/sweep.py          # expects k/ (1s), m/ (1m), h/ (1h) kline CSV folders in cwd
python3 scripts/h12.py; python3 scripts/robust.py; python3 scripts/dd.py
python3 scripts/flow.py; python3 scripts/book.py; python3 scripts/maker.py
# order books (Bybit archive)
python3 scripts/lobparse.py <SYMBOL>-ob200.data out.npz 10 100
python3 scripts/lobmodels.py dayA.npz dayB.npz dayC.npz 10 5
python3 scripts/nextmove.py dayC.npz
# live quotes
python3 scripts/recorder.py 480
```

Kline zips: `data.binance.vision/data/spot/{daily,monthly}/klines/<SYMBOL>/<1s|1m|1h>/`.
Futures depth snapshots: `data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/`.
Scripts use relative folder names as in the session (`k/`, `m/`, `h/`, `bd/`, `ob/`, `rec/`).
