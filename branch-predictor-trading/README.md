# Branch-predictor trading on 1-second crypto bars

Test of the idea: treat each price tick like a CPU branch outcome and trade it
with the standard branch-predictor rules, adding money one step at a time
instead of all at once.

Strategies (position for bar t is decided from bars up to t-1, no lookahead):

| name | rule | CPU analogue |
|---|---|---|
| `1bit` | go the way the last tick went, full size | 1-bit predictor |
| `ramp(N)` | counter in [-N, N]; up tick +1, down tick -1; position = counter/N | N-level saturating counter |
| `ramp_long(N)` | same counter clipped to [0, N], never short | |
| `streak(N)` | add one step per repeated direction, reset to one step when it breaks | "do the same until the loop breaks" |

Data: Binance spot 1-second klines, 2026-09-01 to 2026-09-07 (604,800 bars per
pair), BTC, ETH, SOL, DOGE against USDT. Also resampled to 5s, 15s, 60s.

## Result

The intuition about short intervals is right, and it still cannot make money.

**1. At 1 second, BTC and ETH do trend.** Lag-1 autocorrelation of 1s returns is
+0.11 (BTC) and +0.09 (ETH). That is the opposite sign of daily data. Before
costs the 1-bit rule "earns" +255% (BTC) and +348% (ETH) in seven days.

**2. The edge per trade is around 0.1 to 0.2 basis points.** The `b/e bps`
column is gross profit divided by turnover, i.e. the fee per unit traded at
which the strategy breaks even. Every configuration on every pair sits between
-0.3 and +0.6 bps. Binance spot taker fee is 10 bps (7.5 with BNB), maker fee
is 10 bps at base tier. Even at 1 bps all-in, every strategy loses many times
its capital in a week.

**3. Adding money bit by bit does not change the edge.** `ramp(20)` cuts BTC
turnover from 458,000x to 15,000x and gross from 255% to 26%. Break-even stays
at 0.17 bps. The ramp scales the bet down, it does not make each bet better.
Same for `streak`: it is a leveraged version of the same signal.

**4. The effect is BTC/ETH specific and dies past 5 seconds.** SOL and DOGE
have no 1s autocorrelation and lose even before costs. BTC at 15s is +0.01,
ETH at 15s is -0.09. By 60s bars every rule is roughly zero gross.

**5. Why BTC and ETH trend at 1s.** Their tick size is tiny relative to price
(0.01 on 78,000, i.e. 0.0013 bps) and the book is deep, so a large order walks
the price up one tick per second. The signal is order flow autocorrelation.
Market makers with fee rebates and colocation already harvest it; a taker
paying any fee cannot.

Two caveats that make the gross numbers an upper bound: trades are assumed to
fill at the bar close (in reality you get the next ask or bid, and the
"continuation" in last-trade prices is partly bid/ask alternation), and one
week of data is one regime.

## Key table (7 days, BTC/USDT, gross and net of a fee per unit turnover)

| bars | strategy | lag-1 ac | hit % | gross | turnover | b/e bps | net @ 1 bps | net @ 10 bps |
|---|---|---|---|---|---|---|---|---|
| 1s | 1bit | +0.113 | 28.1 | +255% | 458,313x | 0.06 | -4,328% | -45,576% |
| 1s | ramp(5) | | 51.6 | +101% | 57,303x | 0.18 | -472% | -5,629% |
| 1s | ramp(20) | | 51.3 | +26% | 15,236x | 0.17 | -126% | -1,497% |
| 1s | streak(5) | | 28.1 | +129% | 123,213x | 0.10 | -1,103% | -12,192% |
| 5s | 1bit | +0.128 | 48.0 | +161% | 88,571x | 0.18 | -725% | -8,696% |
| 15s | 1bit | +0.010 | 52.6 | +55% | 31,993x | 0.17 | -265% | -3,144% |
| 60s | 1bit | +0.039 | 51.3 | +14% | 9,411x | 0.15 | -80% | -927% |
| 60s | ramp(20) | | 49.7 | -3.6% | 473x | -0.76 | -8% | -51% |

The 28% hit rate on 1s BTC is real: most non-flat 1s moves are one-tick
bid/ask bounces that reverse, and the gross comes from the rarer multi-tick
runs. With moves under 1 bps treated as flat the hit rate is 53% and the
break-even rises to 0.55 bps, still far below any retail fee.

Full output for all pairs and bar sizes: `results/grid.txt`, and with the 1 bps
minimum-move filter: `results/grid_min1bps.txt`.

## Reproduce

```
mkdir -p data && cd data
for d in 01 02 03 04 05 06 07; do
  curl -O https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2026-09-$d.zip
done
unzip -o '*.zip' && cd ..
python3 sim.py --data data --symbols BTCUSDT --intervals 1,5,15,60 --levels 5,20
python3 sim.py --data data --symbols BTCUSDT --intervals 1 --min-move-bps 1
```

Only numpy is required.

## What would have to be true for this to work

- A fee of under 0.1 bps per unit traded, meaning a maker rebate, not a fee.
- Fills at the bar close, which means being first in queue at the touch.
- The 1s autocorrelation persisting once you are the one adding to the flow.

That is a market-making business, not a retail strategy. The branch-predictor
rules are a reasonable trend filter, but the horizon where crypto trends
enough to pay retail fees is hours to weeks, not seconds.
