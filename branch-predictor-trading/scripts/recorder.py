"""Record best bid/ask (and top levels where cheap) from several venues for N seconds. One JSONL file per venue:
   {"t": recv_time_ms, "b": best_bid, "a": best_ask, "bq": bid_qty, "aq": ask_qty}"""
import asyncio, json, os, sys, time, websockets
DUR=int(sys.argv[1]) if len(sys.argv)>1 else 480
proxy=os.environ.get('HTTPS_PROXY')
V={
 'binance':  ('wss://stream.binance.com:9443/ws/btcusdt@bookTicker', None),
 'bybit':    ('wss://stream.bybit.com/v5/public/spot', {"op":"subscribe","args":["orderbook.1.BTCUSDT"]}),
 'gate':     ('wss://api.gateio.ws/ws/v4/', {"time":int(time.time()),"channel":"spot.book_ticker","event":"subscribe","payload":["BTC_USDT"]}),
 'kraken':   ('wss://ws.kraken.com/v2', {"method":"subscribe","params":{"channel":"ticker","symbol":["BTC/USD"],"event_trigger":"bbo"}}),
 'bitget':   ('wss://ws.bitget.com/v2/ws/public', {"op":"subscribe","args":[{"instType":"SPOT","channel":"books1","instId":"BTCUSDT"}]}),
 'hyperliquid':('wss://api.hyperliquid.xyz/ws', {"method":"subscribe","subscription":{"type":"bbo","coin":"BTC"}}),
}
def parse(name,m):
    d=json.loads(m)
    try:
        if name=='binance': return float(d['b']),float(d['a']),float(d['B']),float(d['A'])
        if name=='bybit':
            x=d.get('data');
            if not x or not x.get('b') or not x.get('a'): return None
            return float(x['b'][0][0]),float(x['a'][0][0]),float(x['b'][0][1]),float(x['a'][0][1])
        if name=='gate':
            x=d.get('result');
            if not x or 'b' not in x: return None
            return float(x['b']),float(x['a']),float(x['B']),float(x['A'])
        if name=='kraken':
            if d.get('channel')!='ticker': return None
            x=d['data'][0]; return float(x['bid']),float(x['ask']),float(x['bid_qty']),float(x['ask_qty'])
        if name=='bitget':
            x=d.get('data');
            if not x: return None
            x=x[0]; return float(x['bids'][0][0]),float(x['asks'][0][0]),float(x['bids'][0][1]),float(x['asks'][0][1])
        if name=='hyperliquid':
            if d.get('channel')!='bbo': return None
            b,a=d['data']['bbo']; return float(b['px']),float(a['px']),float(b['sz']),float(a['sz'])
    except Exception: return None
async def rec(name,url,sub):
    out=open(f'rec/{name}.jsonl','w'); n=0; t_end=time.time()+DUR
    while time.time()<t_end:
        try:
            async with websockets.connect(url, open_timeout=15, proxy=proxy, ping_interval=20) as ws:
                if sub: await ws.send(json.dumps(sub))
                last=None
                while time.time()<t_end:
                    m=await asyncio.wait_for(ws.recv(), timeout=30)
                    p=parse(name,m)
                    if p and p!=last:
                        last=p; out.write(json.dumps({"t":int(time.time()*1000),"b":p[0],"a":p[1],"bq":p[2],"aq":p[3]})+"\n"); n+=1
        except Exception as e:
            open('rec/errors.log','a').write(f'{name} {time.time():.0f} {type(e).__name__}: {str(e)[:100]}\n'); await asyncio.sleep(2)
    out.close(); print(f'{name}: {n} quote updates')
async def main(): await asyncio.gather(*[rec(k,*v) for k,v in V.items()])
asyncio.run(main())
