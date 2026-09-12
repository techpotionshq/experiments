# Logistic-regression trading: can $100 make money, kept static?

Follow-up to `branch-predictor-trading`. That work ranked a logistic regression on
order-book queue imbalance as the strongest predictor per line of code (60 to 75%
on the next mid move) but flagged it as "pays only as a maker with rebates." This
folder answers the practical question: start with $100, keep the model static (fit
once, freeze, no online learning, no neural net), and find a pair we can actually
make money in.

Short answer: the microstructure logistic cannot be monetised by a retail account,
and moving the same static idea to slow bars, the logistic is beaten by a
three-line trend rule with no fitted parameters. The thing that makes money from
$100 is that trend rule, long-only, on 8-hour bars. The best pair is SOL:
$100 becomes about $843 out-of-sample over the 2021 to 2025 test half, Sharpe 1.73,
with a shallower drawdown than buy and hold.

Everything below was measured. Data is not committed; scripts download public
Bybit order books/trades and Binance klines. Dates used: Bybit L2 and spot trades
2025-12-30, Binance 1h klines 2021 to 2025.

## 1. Picking a pair: the survey (`scripts/survey.py`, `results/pair_survey.txt`)

Seven Bybit spot pairs, one day of 10-level book at 100 ms.

| pair | tick (bps) | spread (bps) | mid moves | imbalance -> next move | run-over if imbalance against / with |
|---|---|---|---|---|---|
| WIF | 34.8 | 34.8 | 0.1% | 75% (83% confident) | 3.8% / 0.0% |
| PEPE | 24.3 | 24.3 | 0.1% | 68% (72%) | 4.0% / 0.0% |
| BONK | 12.9 | 12.9 | 0.2% | 64% (74%) | 12.9% / 0.1% |
| FLOKI | 2.5 | 2.6 | 2.3% | 55% (57%) | 19.7% / 10.6% |
| DOGE | 0.8 | 0.8 | 3.4% | 62% (69%) | 37.5% / 12.2% |
| XRP | 0.5 | 0.5 | 1.9% | 62% (73%) | 34.3% / 6.4% |
| BTC | 0.01 | 0.5 | 2.6% | 62% (69%) | 34.1% / 10.9% |

On paper WIF is the dream: an 83% confident directional signal, a 35 bps spread to
earn, and resting orders almost never run over. Large-tick coins (WIF, PEPE, BONK)
have the strongest imbalance signal and the least toxic flow. This agrees with the
literature (queue imbalance is strongest for large-tick instruments).

## 2. The maker trap: the "edge" is a fill-model artifact

A passive maker never crosses the spread. It rests a bid and/or an ask, earns the
spread on completed round trips, and uses the static imbalance signal to skew and
to reduce inventory. `scripts/marketmaker.py` first sims this with a book-only fill
model (a resting order fills when its level trades through, or when the best-quote
size shrinks). It looks incredible:

| WIF, book-only queue model | fee 0 bps | fee 2 | fee 10 |
|---|---|---|---|
| one-day return | +49.5% | +39.9% | +1.5% |
| fills/day | 1923 | 1923 | 1923 |

Then replace the fill model with real trades. `scripts/mmtrades.py` fills a resting
order only when actual aggressive prints reach its price and clear the size ahead of
it in the queue (a cancel never fills you). The same pair, same rule:

| WIF, honest trade-driven fills | fee 0 bps | fee 10 |
|---|---|---|
| one-day return | +0.09% | +0.04% |
| fills/day | 2 | 2 |

The whole +50% was the book-only model counting queue cancels as fills. Honestly,
WIF fills twice a day. The best-bid queue is about $15,000 deep, a $25 order sits
behind roughly 114 median trades, and the spread is exactly one tick 100% of the
time so there is no room to post inside and jump the queue. The 35 bps spread is
only reachable by whoever already holds queue priority (fast, colocated). See
`results/charts/maker_fill_reality.png`.

Every pair, honest fills, at **zero fee** (`results/maker_verdict.txt`):

| pair | WIF | PEPE | BONK | FLOKI | DOGE | XRP | BTC |
|---|---|---|---|---|---|---|---|
| fills/day | 2 | 0 | 0 | 247 | 1199 | 575 | 963 |
| return | +0.1% | 0 | 0 | -0.3% | -1.5% | -0.9% | -1.6% |

The large-tick coins do not fill (queues too deep). The small-tick coins fill but
lose even at zero fee: the fills are adversely selected, so the round trips lose
before any fee is charged, and every run ends stuck long. Fees only make it worse
(DOGE is -31% a day at Bybit's 10 bps maker). The one genuine algo improvement,
posting one tick inside the spread to jump the queue when it is wider than a tick,
makes it worse everywhere, because becoming the best quote means being the first to
be run over. There is no pair, no fee, no rebate at which the static passive maker
is positive. Fillability and low toxicity are anti-correlated across the tick
spectrum, and no pair has both.

## 3. Slow bars: the same static idea where it can pay

The imbalance signal is real; it just lives at 100 ms, where you cannot act on it
for less than it costs. Move the same static logistic to slow bars, where a single
bar's move clears retail fees. `scripts/slowbars.py`, long-only (crypto drifts up;
the short leg loses), $100 start, 7.5 bps per side, coefficients fit on the first
half of 2021 to 2025 and frozen for the out-of-sample second half.

Three rules: `hold` (benchmark), `confirm3` (long after 3 consecutive up bars, flat
after 3 down bars, no fitted parameters), `logit` (static logistic on
`[r1, r2, r3, mom6, vol6]`, long if P(next bar up) > 0.5).

Out-of-sample, final value of $100 (`results/slowbars_oos.txt`):

| pair | interval | hold | confirm3 | logit |
|---|---|---|---|---|
| BTC | 8h | $290 (Sh 1.17) | $260 (Sh 1.30) | $110 (Sh 0.28) |
| ETH | 12h | $162 (Sh 0.63) | $230 (Sh 0.97) | $87 (Sh 0.10) |
| **SOL** | **8h** | **$583 (Sh 1.25)** | **$843 (Sh 1.73)** | **$73 (Sh -0.04)** |
| DOGE | 8h | $180 (Sh 0.72) | $295 (Sh 0.99) | $51 (Sh -0.30) |

The static logistic on returns is the worst rule on the board. It loses money
out-of-sample on three of four pairs. Predicting the direction of the next single
bar is close to a coin flip; the edge is in persistence, not in next-bar direction,
and the three-line confirmation rule captures persistence directly. A fancier
logistic on regime features (SMA ratios, fraction of recent bars up) with a
confidence gate cuts drawdown hard but sits in cash most of the time and does not
beat `confirm3` either. See `results/charts/sol_8h_100usd.png`.

## The answer

Pair SOL, 8-hour bars, 3-bar confirmation, long-only. Static, no fitted parameters,
no ML. $100 becomes about $843 out-of-sample (roughly 2.5 years), Sharpe 1.73,
max drawdown -44% against buy and hold's -64%. It beats holding on both return and
drawdown on SOL, and beats holding out-of-sample on ETH and DOGE too. It is
time-series momentum, the same finding as the parent repo, now with the $100 framing
and the explicit result that the logistic regression does not help. "No ML needed"
turned out to be not just allowed but correct: the ML-flavored rule is the one that
loses.

Honest caveats: one 50/50 split, five years, coins that survived (SOL, DOGE). This
is a slow trend filter that roughly matches holding with less pain, not a tuned
edge, and it will underperform holding in a straight-up bull leg. It is not a
market-neutral money machine; it is long crypto with a drawdown brake.

## Ranking

1. 3-bar confirmation, long-only, 8h bars, on SOL. Positive out-of-sample, beats
   holding risk-adjusted, static, zero fitted parameters.
2. Buy and hold. Higher raw return on some pairs, much deeper drawdowns.
3. Static logistic regression, slow bars. Makes money on some pairs but is beaten
   by rule 1 everywhere and loses out-of-sample on most.
4. Passive maker on the microstructure logistic. Negative on every pair at every
   fee once fills come from real trades. The strongest predictor in the study, and
   the only rule that cannot be turned into money at retail scale.

## Reproduce

```
# order books + spot trades (Bybit), one day
python3 scripts/lobparse.py <DATE>_<SYM>_ob200.data out.npz 10 100      # from quote-saver.bycsi.com/orderbook/spot/<SYM>/
python3 scripts/survey.py *.npz                                          # pair survey
python3 scripts/marketmaker.py <SYM>.npz --fee 0                         # book-only fill model (optimistic)
python3 scripts/mmtrades.py <SYM>.npz <SYM>_spot_<DATE>.csv --fee 0      # honest trade-driven fills; trades from public.bybit.com/spot/<SYM>/
# slow bars (Binance 1h klines from data.binance.vision, resampled)
python3 scripts/slowbars.py <SYM>_1h.csv 8 7.5                           # interval 8h, fee 7.5 bps/side
python3 scripts/charts.py <klines_dir> <lob_dir> results/charts
```
