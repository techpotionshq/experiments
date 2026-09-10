# Two carried-over strategies on crypto and stocks: range breakout and short straddle

Two strategies from other markets, tested here on crypto and US stocks with the
same measured-not-argued discipline as the rest of the repo:

1. **Range breakout**, a forex strategy, ported to a crypto basket and a US stock
   basket on daily bars (`scripts/bt_rangebreak.py`, `results/rangebreak.txt`).
2. **Short ATM straddle**, an options premium-decay strategy, tested on BTC and
   ETH with real Deribit implied vol (`scripts/bt_straddle_crypto.py`,
   `results/straddle_crypto.txt`).

Data is fetched and cached by `scripts/dataio.py` (Binance daily OHLC, Yahoo
daily OHLC, Deribit DVOL), nothing committed, no keys, no live orders. All
Sharpe ratios use a zero cash rate, positions are taken with no look-ahead, and
the universes are still-listed (survivorship-influenced), same caveats as
`FINDINGS.md`.

## Scope note on the straddle

A short straddle's entire P&L is the spread between the **implied** vol you sell
and the **realized** vol that shows up, so it cannot be backtested without real
option data. On crypto that means Deribit, and Deribit only publishes its DVOL
implied-vol index for **BTC and ETH** (also the only two crypto with deep listed
options). So the straddle is BTC/ETH only, on purpose: those are the names where
the premium is real rather than assumed. US-stock options history is paywalled,
so the straddle is not run on stocks here.

---

## Part 1. Range breakout (crypto and stocks)

The classic forex idea: price sits in a range, breaks out, and you enter in the
breakout direction with a stop the other side and ride it until it stalls. Rules
(`scripts/bt_rangebreak.py`, per asset, event-driven, no look-ahead):

- **Range** = highest high / lowest low over the prior `lookback` (20) bars.
- **Entry** = a stop order at the range edge (a gap through it fills at the open).
- **Stop** = entry -/+ `atr_mult` (2.0) x ATR(14).
- **Exit** = stop hit, or the close crosses back through the opposite `exit_lb`
  (10) channel (a trailing Donchian exit), or a 60-bar time stop.
- **Squeeze filter** (optional) = only take breakouts out of an unusually tight
  range. Fee 10 bps/side crypto, 2 bps/side stocks. Equal-weight basket.

### Headline (plain breakout, long only, net of fees)

| market | strategy | CAGR | Sharpe | max DD | vs buy & hold |
|---|---|---|---|---|---|
| **crypto** (6 coins, 2021-2026) | range breakout | **82.6%** | **1.60** | **-47.4%** | hold 82.7% / 1.15 / -77.8% |
| **US stocks** (29 names, 2020-2026) | range breakout | 6.9% | 0.89 | **-8.9%** | hold 19.4% / 1.09 / -29.2% |

**Crypto: same return as buy-and-hold, far better risk.** The breakout matches
hold's CAGR (82.6% vs 82.7%) while lifting Sharpe from 1.15 to 1.60 and cutting
the drawdown from -78% to -47%, because it is in the market only ~41% of the time
and steps aside in the crashes (2025 +12% vs hold -17%, 2026 +10% vs -19%). Low
win rate (39%) with large average winners (+14.6%/trade): a real trend-rider.

**Stocks: a defensive underperformer.** It cuts the drawdown to a third of
hold's (-9% vs -29%) but only earns 7% CAGR against hold's 19% in a strong equity
bull, and its Sharpe (0.89) sits just under hold (1.09). Breakouts on individual
large-cap stocks whipsaw more and the market simply trended up, so sitting out
41% of the time mostly cost return here. It would look better in a sideways or
bear equity regime, which this 2020-2026 sample does not contain.

### Three findings that fall out of the sweep

- **The forex "squeeze/coil" filter does not transfer to crypto.** Requiring the
  range to be tight before a breakout *lowers* the crypto result (Sharpe 1.27 at
  q0.5 vs 1.60 with the filter off); on stocks it is roughly neutral. The plain
  Donchian breakout is better, so it is the default.
- **Long/short is worse than long only, again.** Adding shorts drops crypto to
  Sharpe 0.97 and turns the stock book slightly negative (-0.04), matching the
  repo's earlier result that shorting these up-drifting markets is a loser.
- **It is robust to the lookback.** Crypto Sharpe 1.02-1.60 across lookbacks
  20-55, stock Sharpe 0.89-0.95, drawdowns stable, so the edge is the mechanism
  (ride breakouts, stop out fast), not a tuned parameter.

---

## Part 2. Short ATM straddle on BTC and ETH (real DVOL)

Sell an at-the-money call and put, collect both premiums, and profit if the
realized move stays inside the breakevens. `scripts/bt_straddle_crypto.py` sells
a 30-day ATM straddle each cycle, prices the premium with Black-Scholes at the
**real DVOL implied vol** (Deribit's 30-day index, so the tenor matches exactly),
and settles it against the real BTC/ETH move. Deribit options are European and
cash-settled, so settlement is just the intrinsic `|S_T - K|`, no assignment to
model. Fee ~9 bps of notional per cycle. A stop-managed variant closes early if
the mark-to-market loss exceeds 2x the premium.

### Result (30-day cycles, held to expiry, net of fees, P&L in units of notional)

| underlying | mean IV | mean RV | IV - RV | win % | ann. return | Sharpe | worst cycle | max DD |
|---|---|---|---|---|---|---|---|---|
| **BTC** | 62.3% | 51.0% | **+11.3** | 62% | 12.9% | **0.28** | -43.7% | -65.1% |
| **ETH** | 76.1% | 69.1% | +7.0 | 65% | 6.3% | 0.11 | -49.7% | -91.9% |

**The vol risk premium is real, but it is a bad risk-adjusted trade naked.**
Implied vol sits above realized (+11 points on BTC, +7 on ETH), so selling it
wins most months (62-65%). But the payoff is the textbook short-vol shape:
many small wins and rare huge losses. A single month can lose 44-50% of notional,
the equity curve draws down 65% (BTC) to 92% (ETH), and the Sharpe is only
0.11-0.34. You are picking up premiums in front of a steamroller.

**A stop helps BTC and hurts ETH.** Cutting losers at 2x premium lifts BTC to
14.9% annualized and trims the worst cycle to -33%, but on ETH it just locks in
losses (0% annualized), because ETH's moves gap through the stop. There is no
clean fix: the tail is the strategy.

**Honest limits:** ATM only (no skew or wings, which real straddle sellers use to
cap the tail), 30-day tenor only (DVOL is 30-day; shorter-dated straddles need
shorter-dated IV, not modeled here), no intra-day delta hedging (a real short-vol
desk hedges the delta, which changes the risk profile substantially), and no
margin/liquidation modeling (a -50% cycle could be a margin call before expiry).
So read this as "does the crypto vol risk premium exist and what does harvesting
it naked look like" (yes, and it is ugly), not as a turnkey income strategy.

---

## Verdict

- **Range breakout is a keeper on crypto**, as a risk-reduced way to hold the
  basket: hold's return at two-thirds the drawdown and a much higher Sharpe. It
  is the same family as the trend/momentum strategies that win everywhere else in
  this repo, which is the recurring lesson: on crypto, ride medium-horizon trends
  long-only and get out of the way in crashes.
- **Range breakout on stocks is defense, not offense**, in this bull sample.
- **The short straddle confirms the crypto vol risk premium is real but not
  worth harvesting naked.** High win rate, low Sharpe, ruinous tail. The version
  a professional would run (defined-risk with wings, delta-hedged, skew-aware) is
  a different, more capital-intensive strategy than "short the straddle and
  collect theta", and needs full option-chain data to test.

## Reproduce

```
python3 scripts/dataio.py all          # cache Binance / Yahoo / Deribit DVOL
python3 scripts/bt_rangebreak.py       # crypto + stocks breakout, full report
python3 scripts/bt_straddle_crypto.py  # BTC/ETH short straddle on real DVOL
```

Sources: `data.binance.vision` (crypto OHLC), `query2.finance.yahoo.com` (stock
OHLC), `deribit.com/api/v2/public/get_volatility_index_data` (DVOL). Data is
cached under `datacache/` and is git-ignored.

## Disclaimer

Measurement, not investment advice. Every result is in-sample and
survivorship-influenced, and the straddle in particular carries tail risk that a
backtest understates. Paper-trade before risking real capital.
