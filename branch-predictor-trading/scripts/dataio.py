"""Data fetchers for the range-breakout and short-straddle studies.

Everything here downloads from public archives and caches to `datacache/`
(git-ignored), so the backtests run end-to-end with no pre-staged folders:

  - Binance monthly 1d klines (full OHLC) from data.binance.vision
  - Yahoo daily OHLC from query2.finance.yahoo.com (needs a browser UA)
  - Deribit DVOL, the real 30-day implied-vol index, from the Deribit API

No API keys, no live orders, nothing committed.
"""
import os, io, json, time, zipfile, urllib.request, datetime as dt
import numpy as np, pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datacache')
os.makedirs(CACHE, exist_ok=True)
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'


def _get(url, timeout=30, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e; time.sleep(2 ** i)
    raise last


# ---------- Binance daily OHLC (crypto) ----------
def _months(start='2021-01', end=None):
    s = dt.date(int(start[:4]), int(start[5:7]), 1)
    e = dt.date.today() if end is None else dt.date(int(end[:4]), int(end[5:7]), 1)
    out = []
    while s <= e:
        out.append(f"{s.year:04d}-{s.month:02d}")
        s = dt.date(s.year + (s.month == 12), (s.month % 12) + 1, 1)
    return out


def binance_daily(sym, start='2021-01'):
    """Full daily OHLCV for a Binance spot symbol, cached as one parquet-ish csv."""
    cache = os.path.join(CACHE, f'binance_{sym}_1d.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df
    frames = []
    for m in _months(start):
        url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/1d/{sym}-1d-{m}.zip"
        try:
            raw = _get(url, timeout=30, tries=2)
        except Exception:
            continue  # month not published yet
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            a = np.loadtxt(z.open(z.namelist()[0]), delimiter=',', usecols=(0, 1, 2, 3, 4, 5))
        except Exception:
            continue
        if a.ndim == 1:
            a = a[None, :]
        frames.append(a)
    if not frames:
        return None
    a = np.concatenate(frames)
    t = a[:, 0].astype('float64')
    t = np.where(t > 1e15, t / 1e3, np.where(t < 1e12, t * 1e3, t))  # ms/us -> ms
    idx = pd.to_datetime(t, unit='ms', utc=True).tz_convert(None).normalize()
    df = pd.DataFrame({'open': a[:, 1], 'high': a[:, 2], 'low': a[:, 3],
                       'close': a[:, 4], 'volume': a[:, 5]}, index=idx)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(cache)
    return df


# ---------- Yahoo daily OHLC (stocks) ----------
def yahoo_daily(ticker, rng='6y'):
    cache = os.path.join(CACHE, f'yahoo_{ticker}.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={rng}&interval=1d")
    d = json.loads(_get(url, timeout=30))
    res = d['chart']['result'][0]
    ts = pd.to_datetime(res['timestamp'], unit='s', utc=True).tz_convert(None).normalize()
    q = res['indicators']['quote'][0]
    df = pd.DataFrame({'open': q['open'], 'high': q['high'], 'low': q['low'],
                       'close': q['close'], 'volume': q['volume']}, index=ts).dropna()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    time.sleep(0.4)  # be gentle, Yahoo rate-limits cloud IPs
    df.to_csv(cache)
    return df


# ---------- Deribit DVOL (real 30-day implied vol index) ----------
def dvol_daily(currency='BTC'):
    cache = os.path.join(CACHE, f'dvol_{currency}.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)['dvol']
    rows = []
    end = int(time.time() * 1000)
    for _ in range(6):  # page backwards, 1000 daily rows per call
        url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency={currency}&start_timestamp=1577836800000&end_timestamp={end}"
               f"&resolution=86400")
        d = json.loads(_get(url, timeout=40))
        data = d.get('result', {}).get('data', [])
        if not data:
            break
        rows = data + rows
        first_ts = data[0][0]
        if first_ts <= 1577836800000 + 86400000 or len(data) < 1000:
            break
        end = first_ts - 86400000
    if not rows:
        return None
    seen = {}
    for r in rows:
        seen[r[0]] = r[4]  # close DVOL
    idx = pd.to_datetime(sorted(seen), unit='ms').normalize()
    s = pd.Series([seen[t] for t in sorted(seen)], index=idx, name='dvol')
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.to_frame().to_csv(cache)
    return s


CRYPTO = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT']
STOCKS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'JPM', 'V',
          'MA', 'UNH', 'HD', 'PG', 'COST', 'XOM', 'CVX', 'JNJ', 'WMT', 'BAC',
          'KO', 'PEP', 'CRM', 'ADBE', 'NFLX', 'AMD', 'INTC', 'ORCL', 'DIS']


if __name__ == '__main__':
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'crypto'):
        for s in CRYPTO:
            df = binance_daily(s)
            print(f"{s:10} {None if df is None else len(df)} bars "
                  f"{'' if df is None else df.index.min().date()}..{'' if df is None else df.index.max().date()}")
    if what in ('all', 'stocks'):
        for t in STOCKS + ['SPY']:
            df = yahoo_daily(t)
            print(f"{t:6} {len(df)} bars {df.index.min().date()}..{df.index.max().date()}")
    if what in ('all', 'dvol'):
        for c in ['BTC', 'ETH']:
            s = dvol_daily(c)
            print(f"DVOL {c}: {len(s)} days {s.index.min().date()}..{s.index.max().date()}")
