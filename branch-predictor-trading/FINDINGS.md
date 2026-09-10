# Findings: what actually works, across crypto and stocks

This is the full synthesis of the whole investigation, from the original
branch-predictor idea through order books, makers, trend, mean reversion, memes,
stocks, machine-learning classifiers, news spikes, and weekly flipping. Every
number here was measured on real data, not argued. Scripts are in `scripts/`,
raw outputs in `results/`. `README.md` has the detailed crypto write-up
(sections 1-11); this file adds the stock work and the cross-asset verdict.

## The one rule that explains every result

**The edge is never in predicting the next tick. It is in riding a
medium-horizon trend across a basket, and it is only worth money when the fee is
small relative to the move you capture.**

Everything below is a corollary of that sentence.

- Short-horizon direction prediction (next second, next bar) is a coin flip in
  both crypto and stocks. Order books push it to ~60% on crypto but the move is
  smaller than the fee.
- Medium-horizon momentum (weeks to months), diversified across a basket, is a
  real, documented premium in both asset classes.
- A round trip costs ~2x the fee. Any strategy whose per-trade edge is smaller
  than that loses, no matter how clever the signal.

## Final verdict, one line

The best thing we found is **cross-sectional momentum on US stocks** (hold the
top 5 by 30-day momentum, re-ranked weekly, and go to cash when the market is in
a downtrend; equal-weighted): Sharpe ~1.38,
~29% CAGR, -20% drawdown, net of fees, and it fits a $100 account with fractional
shares. The crypto version of the same idea works too but with triple the
drawdown. Nothing we tested delivers reliable weekly gains; the best weekly-
positive rate for any real-edge strategy was 53-55%.

---

## Part A. Crypto, fast strategies: real signals, killed by fees

Detailed in README sections 1-9. Summary of what was measured:

| experiment | finding | verdict |
|---|---|---|
| 1-bit / 2-bit / streak on 1s bars | +0.1 to 0.5 bps/trade edge vs 5-10 bps fee | dead after fees |
| interval sweep 1s to 1d | edge grows with horizon; 12h first to clear 7.5 bps | slow works |
| order flow (taker) 1s | 58-60% hit, 0.1-0.44 bps, decays by 5 min | < fee |
| depth imbalance 30s | predicts nothing at 1-60 min | no edge |
| queue imbalance / logistic / DeepLOB (next mid move) | 61 / 61 / 56% direction; DeepLOB worst on the maker question | signal real, tiny |

The order-book signal is the strongest predictor per line of code (~61% next-move
direction on a large-tick pair), but the realised move is 4-8 bps and a taker
round trip costs ~20-25 bps.

### Order-book execution follow-ups (this session)

Tested every way to actually harvest the imbalance signal on Bybit spot books
(WIF large-tick, BTC control). `results/eventdriven_wif_btc.txt`,
`maker_wif_btc.txt`, `takerentry_wif.txt`.

| execution | per-trade result | why |
|---|---|---|
| taker in / taker out (event-driven, hold to opposite signal) | gross +3 to +8 bps, **net -18 to -23 bps** | 20 bps fees + spread |
| directional maker (post bid because bullish) | net negative, fills only when wrong | adverse selection |
| symmetric maker + imbalance cancel | **break-even at ~-1 bps rebate**; +1.3 bps/fill at -2 rebate | the real MM business |
| buy market, sell limit (taker in, maker out) | **gross negative**, net -14 bps | caps winners, keeps losers |

The only order-book structure with a pulse is a **rebated market maker** on a
large-tick pair, which needs a venue that pays a maker rebate (Kraken Tier 12
-2 bps at $10M/mo, Bitget incentives), not a retail account. The imbalance signal
is worth ~0.4 bps/fill of adverse-selection reduction there.

---

## Part B. The fee-robust builds: slow trend

### Crypto trend basket (README section 10, `results/trend_basket.txt`)
Vol-targeted trend following, long-flat, daily bars, 6-coin basket:
**+30% CAGR, Sharpe 1.20, -30% DD net of 10 bps**, vs hold +83% / 1.15 / -78%.
Fee drag only 2.6%/yr because turnover is ~26 trades/yr. Still profitable at
30 bps/side. Beats hold on Sharpe and halves its drawdown.

### Crypto cross-sectional (README section 11, `results/strategy_search.txt`)
Search over 31 coins: mean reversion and reversal lose; **trend wins**. Best:
Donchian breakout + market-regime filter + vol target, **+47% CAGR, Sharpe 1.05,
-61% DD net of 10 bps**, beating equal-weight hold on all three.

### Small pool for a $100 account (`results/topk_small_pool.txt`)
Holding only the top-K strongest each week keeps positions tradeable:
K=3 CAGR 53% / Sharpe 1.07 ($33/coin), K=5 37% / 0.88 ($20/coin), K=8 44% / 1.00
($12/coin). Comparable to the full basket while fitting $100.

### Top-3 account simulation from $100 (`results/sim_top3.txt`)
$100 -> $1,122 over 5.7 years (11.2x, 53% CAGR), **but** -63% max drawdown and
all gains are 2021-2024: from the 2024 peak of $2,508 it fell to $1,122
(2025 -26%, 2026 -40%). Timing dominates: start-2021 is 11x, start-2025 is
down ~55%.

### Single-coin ranking (`results/single_coin_ranking.txt`)
The basket edge does not transfer cleanly to one coin. Best single coins for the
time-series trend: SOL (Sharpe 1.61), BNB/PEPE/NEAR/FLOKI/DOGE ~1.0-1.2; many
alts negative. This is hindsight/survivorship; the point of the basket is that
it holds whichever coin is trending without you pre-picking it.

---

## Part C. Stocks: the same idea, cleaner

Yahoo daily data, 30 large-cap US names + SPY, 2021-2026, ~2 bps
(`results/stocks_momentum.txt`, `stocks_bit_logistic.txt`).

### Cross-sectional momentum works, and better than in crypto
| strategy (net ~2 bps) | CAGR | Sharpe | max DD |
|---|---|---|---|
| top-5 momentum + regime (equal-weight, no vol target) | 29.3% | **1.38** | **-20.4%** |
| top-3 momentum + regime | 34.3% | 1.29 | -27.5% |
| equal-weight all 30 | 15.6% | 0.89 | -29.6% |
| SPY buy & hold | 11.3% | 0.71 | -25.4% |

Higher Sharpe than the crypto version (1.05) with a third of the drawdown,
near-zero fees, and $100-tradeable via fractional shares. Momentum was
discovered in equities (Jegadeesh-Titman 1993); this is its home.

### 1-bit and logistic fail here too
Daily direction is a coin flip: yesterday predicts today **50.9%**; logistic on
5 lagged returns **51.7%** overall (vs 51.6% base rate of up-days), 53.1% on its
confident fifth. Same failure as crypto order-book next-move, confirming the
short-horizon-is-noise rule across asset classes.

### News-spike reaction (`results/news_spike_study.txt`)
Using a >2 sigma move on >2x volume as a news proxy: down-spikes (bad news)
revert (+1.8% over 10d on the per-stock average), up-spikes barely drift. As a
net-of-fee strategy, buying spikes earns ~+0.7%/trade over 10 days in either
direction, but with 51-54% win rates and -16 to -27% worst trades. Real but
small and noisy, and survivorship-biased (bankruptcies that never bounced are
absent). This is the one place an ML/LLM layer would genuinely add value:
reading the news **text** to separate overreactions (buy) from fundamental
breaks (avoid), which price alone cannot do. Not backtestable here without a
news feed.

### Weekly flipping (`results/weekly_flip.txt`)
For those who want to trade every week: weekly momentum (buy last week's 5
strongest, rotate each Friday) is the best real-edge weekly strategy found:
**CAGR 21%, Sharpe 0.89, 53% of weeks positive**. Weekly reversal (buy last
week's losers) is weaker (13% / 0.57). Both are ~half red weeks with -9 to -11%
worst weeks: a slight weekly lean, not income.

---

## Cross-asset conclusions

1. **Next-tick prediction is dead everywhere.** 1-bit, logistic, order-book
   next-move: coin flips net of cost, crypto and stocks alike.
2. **Medium-horizon cross-sectional momentum is the durable edge**, in both
   markets, and cleaner in stocks (higher Sharpe, lower drawdown, near-zero fees).
3. **Fees decide everything fast.** The order-book signal is real but smaller
   than a taker crossing; only a rebated maker clears it, which retail cannot be.
4. **No reliable weekly gains exist.** Best weekly-positive rate for any real
   edge was 53-55%. Weekly returns are dominated by noise in every strategy.
5. **Survivorship and in-sample bias flatter every long-only backtest here.**
   Levels are optimistic; the *ranking of families* is the robust takeaway.

## The practical playbook

- **Best algo:** US stock cross-sectional momentum, top-5 by 30-day return,
  re-ranked weekly, cash when the equal-weight index is below its 30-day average,
  equal-weighted (no vol targeting). Sharpe ~1.38, ~29% CAGR, -20% DD. Run it as
  `bt_stocks.py` (the crypto small-pool variant `bt_topk.py` adds 50% vol
  targeting on top; the stock book does not).
- **Best weekly-flip:** buy last week's 5 strongest stocks, rotate every Friday
  (`bt_weeklyflip.py`). ~21% CAGR, 53% green weeks. Trades weekly, fits $100.
- **What to avoid:** anything fast and taker; mean reversion / dip-buying on
  memes (falling knives); single-coin bets; leverage above ~2x (a -60% drawdown
  becomes liquidation).
- **Expectation setting on $100:** ~+$0.40 in an average week, ~-$9 in a bad
  week, roughly half of weeks red, ~+$25 in a good year. A learning stake, not
  income.

## Reproduce

```
# crypto (needs ob/ from lobparse, k1h/ kmeme/ kalt/ kline folders)
python3 scripts/eventdriven.py ob/WIF_*.npz --lvl 1 --fee 10
python3 scripts/makermm.py    ob/WIF_*.npz --lvl 1 --thr 0.6
python3 scripts/bt_trend.py   --interval 24H --slow 30 --fast 8 --confirm 2 --targetvol 0.6 --fee 10
python3 scripts/bt_search.py  --fee 10 --top 0.25 --lookback 30 --regime 30
python3 scripts/bt_topk.py    --lookback 30 --fee 10 --live --K 5
python3 scripts/bt_meanrev.py --interval 1H --entry 2.0 --trend 168 --fee 10
# stocks (needs stocks/*.json from Yahoo chart API)
python3 scripts/bt_stocks.py
python3 scripts/bt_stocks_bit.py
python3 scripts/bt_newsspike.py
python3 scripts/bt_weeklyflip.py
```

Data is not committed; every script downloads or reads public archives.
Kline zips: `data.binance.vision`. Order books: `quote-saver.bycsi.com`.
Stocks: `query1.finance.yahoo.com/v8/finance/chart/<TICKER>`.

## Method notes

- **Sharpe uses a zero cash rate** (excess-of-zero, not excess-of-cash). At
  2021-2026 US cash rates, subtract roughly 0.2-0.3 to compare any single Sharpe
  against holding cash. The comparison *between* a strategy and its buy-and-hold
  baseline is unaffected, since both use the same convention.
- **Positions are taken on the bar after the signal** in every backtest
  (`.shift(1)` on the weight, `direction[t-1]` in `sim.py`), so there is no
  look-ahead. Fees are charged per side on realized turnover.
- **Long-only backtests run on still-listed universes** (30 current large-cap
  stocks, coins still trading), so levels are survivorship-influenced and read as
  optimistic; the robust takeaway is the *ranking of strategy families*, not the
  exact CAGR/Sharpe.

## Disclaimer

This is measurement, not investment advice. Every profitable result is
in-sample and survivorship-influenced; past performance is not predictive.
Trade only money you can afford to lose, and paper-trade any strategy before
risking real capital.
