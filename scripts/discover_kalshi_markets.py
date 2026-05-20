"""
Discover Kalshi BTC market series tickers and field shapes.
Run: python3 scripts/discover_kalshi_markets.py
"""
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from btc_kalshi_system.execution.raw_http_client import KalshiRawClient
import json

c = KalshiRawClient()

# 1. List ALL BTC-related series
print("=== ALL BTC SERIES ===")
try:
    r = c._request("GET", "/trade-api/v2/series?status=active&limit=100")
    btc_series = [
        s for s in r.get("series", [])
        if "BTC" in s.get("ticker", "").upper() or "BTC" in s.get("title", "").upper()
    ]
    for s in btc_series:
        print(json.dumps(s, indent=2))
    if not btc_series:
        print("(no BTC series found in active series)")
except Exception as e:
    print(f"Series endpoint error: {e}")

# 2. Probe KXBTC (daily)
print("\n=== KXBTC MARKETS (first 3) ===")
try:
    r = c._request("GET", "/trade-api/v2/markets?series_ticker=KXBTC&status=open&limit=3")
    for m in r.get("markets", []):
        print(json.dumps(m, indent=2))
except Exception as e:
    print(f"KXBTC probe error: {e}")

# 3. Try known 15-min / hourly BTC series tickers
for ticker in ["BTCUSD", "BTCD", "BTCH", "BTCM", "KXBTCH", "KXBTCM", "BTCUSDM", "BTCUSDH"]:
    try:
        r = c._request("GET", f"/trade-api/v2/markets?series_ticker={ticker}&status=open&limit=2")
        markets = r.get("markets", [])
        if markets:
            print(f"\n=== {ticker} MARKETS ===")
            for m in markets:
                print(json.dumps(m, indent=2))
    except Exception:
        pass

# 4. Search by keyword in market titles
print("\n=== MARKETS CONTAINING 'bitcoin' OR 'btc' (title search, limit 10) ===")
try:
    r = c._request("GET", "/trade-api/v2/markets?status=open&limit=200")
    for m in r.get("markets", []):
        title = (m.get("title") or "").lower()
        ticker = (m.get("ticker") or "").upper()
        if "bitcoin" in title or "btc" in ticker:
            # Print only key fields to keep output concise
            summary = {
                "ticker": m.get("ticker"),
                "series_ticker": m.get("series_ticker"),
                "title": m.get("title"),
                "status": m.get("status"),
                "close_time": m.get("close_time"),
                "expiration_time": m.get("expiration_time"),
                "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"),
                "strike_price": m.get("strike_price"),
                "result_at_open": m.get("result_at_open"),
                "open_price": m.get("open_price"),
            }
            print(json.dumps(summary, indent=2))
except Exception as e:
    print(f"Broad search error: {e}")
