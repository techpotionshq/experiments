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

The 1-bit hit rate in the table (28%) looks paradoxical next to that positive
autocorrelation, but the two measure different things. At 1-second resolution
most ticks are one-unit bid/ask-bounce reversals, so the raw direction sign
flips more often than it repeats (low hit rate), while the rarer, larger moves
persist (positive return autocorrelation and positive gross). The gross is
carried by a few big directional runs, not by being right often, which is
exactly why the tiny break-even fee kills it.

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

## 10. The fee-robust build: vol-targeted trend basket

The question "make something that profits despite a 10 bps maker fee" has one
answer, and it is not in the order book. At 10 bps/side a round trip pays 20 bps
in fees; the order-book signal is worth 4 to 8 bps per move, so no fast strategy
clears it (sections 5 to 7, and the event-driven and maker sims in
`results/eventdriven_wif_btc.txt`, `results/maker_wif_btc.txt`,
`results/takerentry_wif.txt`). The only way to make the fee irrelevant is to
trade rarely at a horizon where moves are hundreds of bps, so 20 bps is noise.

`scripts/bt_trend.py` builds that: long-flat (no shorting), daily bars, a 30-day
trend filter, an 8-day momentum entry with 2-bar confirmation (the branch
predictor's saturating-counter idea, here to cut turnover), volatility targeting
to 60% annualized, equal-weight across BTC/ETH/SOL/DOGE/BNB/XRP. Binance spot
1h klines resampled to 24h, 2021-01 to 2026-08 (`results/trend_basket.txt`).

| 24h trend basket, 10 bps/side | CAGR | Sharpe | max DD |
|---|---|---|---|
| strategy, net of fees | +30.2% | 1.20 | -30.5% |
| buy-and-hold basket | +82.7% | 1.15 | -77.8% |

It gives up raw return (hold rode the 2021 melt-up) but beats hold on Sharpe and
runs less than half the drawdown, and it protects in the bad years: 2022 -19% vs
hold -68%, 2025 +5% vs hold -17%, 2026 flat vs hold -19%. Turnover is ~26
unit-trades/yr, so the fee costs only 2.6%/yr. It still returns 23.7% at Sharpe
~1 even at 30 bps/side, triple the Bybit maker fee. That is what "profit despite
the fee" looks like: the fee is ~8% of gross, not 300%. It is time-series
momentum (crisis-alpha / drawdown protection), not a secret edge, and out of
sample (2024+) it degrades honestly to Sharpe 0.73 while still halving hold's
drawdown.

## 11. What actually works on alts and memes: cross-sectional trend

`scripts/bt_search.py` searches four families over a 31-coin universe (majors,
mid-cap alts, memes) on daily bars, 2021-2026, all net of fee, weekly rebalance
for the cross-sectional books (`results/strategy_search.txt`). The answer is not
mean reversion (section, `results/meanrev_memes.txt`, loses) and not
short-term reversal (XSREV, -29%/yr). It is trend, in two forms that both beat
buy-and-hold net of 10 bps:

| net of 10 bps/side | CAGR | Sharpe | max DD | worst week |
|---|---|---|---|---|
| cross-sectional momentum (top 25%, weekly) | +38.9% | 0.81 | -86% | -42% |
| Donchian breakout (30d high / 15d low) | +47.5% | 0.88 | -89% | -38% |
| **breakout + market-regime filter + vol target** | **+47.1%** | **1.05** | **-61%** | **-22%** |
| equal-weight buy-and-hold | +30.0% | 0.74 | -87% | -49% |

The winner adds two risk overlays from section 10: a **market-regime filter**
(go fully to cash when the equal-weight index is below its 30-day average) and
**volatility targeting** (scale the book to 50% annualized vol, no leverage).
Together they cut the drawdown from -89% to -61% and halve the worst week, while
keeping ~47% CAGR and lifting Sharpe to 1.05. It beats hold on CAGR, Sharpe and
drawdown at once. It is robust: Sharpe 0.84 to 1.15 across lookbacks 20 to 55,
and still Sharpe 0.85 at 30 bps/side (triple the fee), because weekly rebalance
keeps turnover low.

Honest limits: it is still a volatile alt book, drawdown -61% and only 31% of
weeks positive (median week 0%), so it is lumpy trend, not steady weekly income.
It protects in down years (2022 -36% vs hold -73%) and beat hold in 2024 (+133%
vs +88%), but lost in 2025 (-41%) and 2026 (-16%): alt trend has been poor
lately. The universe is survivorship-biased (only coins still listed), which
flatters any long-only backtest, though breakout/momentum that buys strength is
less exposed to it than dip-buying. Parameters were picked looking at the full
sample, so treat the level as optimistic and the ranking of families as the
robust result.

## 12. Stocks: the same idea, cleaner (and the cross-asset verdict)

The trend/momentum family was carried over to US equities (Yahoo daily, 30 large
caps + SPY, 2021-2026, ~2 bps): `results/stocks_momentum.txt`,
`stocks_bit_logistic.txt`, `news_spike_study.txt`, `weekly_flip.txt`.

- **Cross-sectional momentum works better in stocks:** top-5 by 30-day momentum
  + regime filter (equal-weight, no vol target) = CAGR 29%, Sharpe 1.38, maxDD
  -20%, vs SPY 11% / 0.71 / -25%. Higher
  Sharpe than the crypto basket (1.05) with a third of the drawdown, near-zero
  fees, $100-tradeable via fractional shares.
- **1-bit and logistic fail here too:** daily direction 50.9% (coin flip);
  logistic 51.7% vs a 51.6% base rate. Same short-horizon-is-noise result.
- **News-spike reaction** (proxy: >2 sigma on >2x volume): down-spikes revert
  ~+1.8% over 10d, edge ~+0.7%/trade but 51-54% win, survivorship-biased. The one
  place an ML/LLM layer earns its keep: reading the news *text* to tell an
  overreaction from a fundamental break. Needs a news feed to test properly.
- **Weekly flip:** buy last week's 5 strongest, rotate Friday = CAGR 21%, Sharpe
  0.89, 53% positive weeks. The best real-edge weekly-trading strategy found, but
  still ~half red weeks. No strategy gave reliable weekly gains.

Full cross-asset synthesis and the practical playbook are in **`FINDINGS.md`**.

## Ranking (all experiments, both asset classes)

1. **US stock cross-sectional momentum** (top-5 by 30-day momentum, regime
   filter, equal-weight, no vol target, section 12). Best risk-adjusted result
   found: Sharpe 1.38, -20% DD, near-zero
   fees, $100-friendly. Momentum's home market.
2. Vol-targeted crypto trend basket / cross-sectional breakout (sections 10-11).
   Same DNA, beats hold net of 10 bps, but -30 to -61% drawdowns.
3. Long-only 12h 3-bar confirmation (section 4). Net positive at retail fees;
   matches holding with shallower drawdowns. Same family.
4. Logistic / queue imbalance for the next mid move (sections 5-7). Strongest
   predictor per line of code. Pays only as a rebated maker, not for retail.
5. News-spike reversion on stocks. Small real edge (~0.7%/trade), noisy; wants an
   ML news-text classifier to be worth trading.
6. Everything fast and taker, and next-tick prediction (1-bit, logistic on price):
   real signals or coin flips, each smaller than the cost of acting on them.

**No strategy tested produces reliable weekly gains.** The best weekly-positive
rate for any real edge was 53-55%.

*Method note: all Sharpe ratios use a zero cash rate (excess-of-zero, not
excess-of-cash); at 2021-2026 rates, subtract ~0.2-0.3 to compare a single
Sharpe against holding cash. Positions are always taken the bar after the
signal (no look-ahead), and long-only levels on still-listed universes are
survivorship-influenced, so treat the ranking of families as the robust result
and the exact levels as optimistic. See `FINDINGS.md` for the full method notes.*

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
# fee-robust trend basket (section 10): 1h klines in k1h/, resampled to 24h
python3 scripts/bt_trend.py --interval 24H --slow 30 --fast 8 --confirm 2 --targetvol 0.6 --fee 10
# order-book execution sims (section 5-7 follow-ups): needs ob/*.npz from lobparse
python3 scripts/eventdriven.py ob/WIF_*.npz --lvl 1 --fee 10      # taker both legs
python3 scripts/makermm.py    ob/WIF_*.npz --lvl 1 --thr 0.6     # symmetric maker, imbalance cancel
python3 scripts/makerbook.py  ob/WIF_*.npz --thr 0.6             # directional maker
python3 scripts/takerentry.py ob/WIF_*.npz --thr 0.7            # buy market, sell limit
# cross-sectional search + small-pool + live (sections 11-12): k1h/ kmeme/ kalt/ kline folders
python3 scripts/bt_search.py  --fee 10 --top 0.25 --lookback 30 --regime 30
python3 scripts/bt_topk.py    --lookback 30 --fee 10 --live --K 5   # small pool + live picks
python3 scripts/bt_meanrev.py --interval 1H --entry 2.0 --trend 168 --fee 10   # meme mean reversion (loses)
python3 scripts/sim_top3.py; python3 scripts/live_signal.py
# stocks (section 12): stocks/*.json from Yahoo chart API
python3 scripts/bt_stocks.py        # cross-sectional momentum vs SPY
python3 scripts/bt_stocks_bit.py    # 1-bit + logistic (coin flip)
python3 scripts/bt_newsspike.py     # news-spike reaction
python3 scripts/bt_weeklyflip.py    # weekly momentum vs reversal
```

Kline zips: `data.binance.vision/data/spot/{daily,monthly}/klines/<SYMBOL>/<1s|1m|1h>/`.
Futures depth snapshots: `data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/`.
Stocks: `query1.finance.yahoo.com/v8/finance/chart/<TICKER>?range=5y&interval=1d`.
Scripts use relative folder names as in the session (`k/`, `m/`, `h/`, `bd/`, `ob/`,
`rec/`, `k1h/`, `kmeme/`, `kalt/`, `stocks/`).

**See `FINDINGS.md` for the full cross-asset synthesis, the ranking of every
strategy tested, and the practical playbook.**
