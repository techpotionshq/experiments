"""Small-pool version: hold only the top-K strongest coins, sized to fit $100.

Instead of holding every coin in breakout (29 names, ~$3.45 each, untradeable on
$100), rank the universe each week by trailing momentum and hold only the top K,
equal weight, gated by the market regime and vol-targeted. K=5 -> $20 a coin,
K=3 -> $33 a coin, both above exchange minimums.

Backtests a range of K net of fee, reports weekly $ on $100, and prints the LIVE
top-K holdings as of the latest bar. No orders placed.

Usage: python3 scripts/bt_topk.py --lookback 30 --fee 10 [--live]
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import bt_search as X
import live_signal as L

def topk_weights(close, lookback, K):
    mom=close.pct_change(lookback)
    reb=(close.index.weekday==0)
    W=pd.DataFrame(0.0, index=close.index, columns=close.columns)
    valid=close.notna() & mom.notna()
    for d in close.index[reb]:
        row=mom.loc[d][valid.loc[d]].dropna()
        if len(row)<3: continue
        picks=row.sort_values(ascending=False).index[:K]
        W.loc[d, picks]=1.0/len(picks)
    Wv=W.values.copy(); Wv[~reb,:]=np.nan
    return pd.DataFrame(Wv, index=W.index, columns=W.columns).ffill().fillna(0.0)

def build_panel():
    syms=sorted({__import__('os').path.basename(f).split('-1h-')[0]
                 for fol in L.FOLD_1H for f in __import__('glob').glob(f'{fol}/*-1h-*.csv')})
    panel={s:L.load_hist(s) for s in syms}
    panel={k:v for k,v in panel.items() if v is not None and len(v)>60}
    return pd.DataFrame(panel).sort_index()

def main():
    def opt(n,d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
    lb=int(opt('--lookback','30')); fee=float(opt('--fee','10'))/1e4
    close=build_panel(); ret=close.pct_change().fillna(0.0)
    eww=close.notna().astype(float).div(close.notna().sum(axis=1).replace(0,np.nan),axis=0)
    mkt=(1+(ret*eww).sum(axis=1)).cumprod(); regime=mkt>mkt.rolling(30).mean()
    print(f"universe {close.shape[1]} coins, {close.index.min().date()} to {close.index.max().date()}, fee {fee*1e4:.0f} bps/side\n")
    print(f"{'pool K':>7}{'$/coin':>8}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'wkPos%':>8}{'avg wk $100':>12}{'worst wk $':>11}")
    for K in [3,5,8,10]:
        W=topk_weights(close, lb, K)
        net=X.settle(W, ret, fee, regime, 0.50)
        p=X.perf(net)
        wk=((1+net).resample('W').prod()-1).dropna()
        print(f"{K:>7}{100/K:>7.0f}${p['cagr']:>7.0f}%{p['sharpe']:>8.2f}{p['maxdd']:>7.0f}%"
              f"{p['wk_pos']:>7.0f}%{wk.mean()*100:>10.2f}${wk.min()*100:>10.2f}$")
    if '--live' in sys.argv:
        K=int(opt('--K','5'))
        mom=close.pct_change(lb).iloc[-1]
        live=close.notna().iloc[-1] & mom.notna()
        ranked=mom[live].sort_values(ascending=False)
        risk_on=regime.iloc[-1]
        print(f"\n=== LIVE top-{K} as of {close.index.max().date()} | regime {'RISK-ON' if risk_on else 'RISK-OFF (CASH)'} ===")
        if not risk_on:
            print(">>> RISK-OFF: hold 100% cash, no positions. <<<"); return
        picks=ranked.index[:K]
        print(f"HOLD these {K} strongest (by {lb}d return), ${100/K:.0f} each on spot / ${200/K:.0f} exposure each at 2x:")
        for c in picks:
            print(f"   {c.replace('USDT',''):<10} {lb}d return {ranked[c]*100:+.0f}%   -> ${100/K:.0f} spot")

if __name__=='__main__': main()
