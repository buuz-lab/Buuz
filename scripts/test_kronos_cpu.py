"""
Smoke-test Kronos inference in isolation -- outside the async main loop.
Run this BEFORE starting main.py to confirm the model loads and infers
without segfaulting.

Usage:
    cd ~/Kronos\\ V2
    python3 scripts/test_kronos_cpu.py
"""
import sys
import os
# Ensure the project root (parent of scripts/) is on sys.path so kronos_model is importable
# when this script is invoked as `python3 scripts/test_kronos_cpu.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

print("Step 1 — importing torch …")
import torch
torch.set_num_threads(1)
print(f"  torch version : {torch.__version__}")
print(f"  MPS available : {torch.backends.mps.is_available()}")
print(f"  CUDA available: {torch.cuda.is_available()}")

print("\nStep 2 — loading Kronos model to CPU …")
from kronos_model import Kronos, KronosPredictor, KronosTokenizer

model     = Kronos.from_pretrained("NeoQuasar/Kronos-small",        map_location="cpu")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", map_location="cpu")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
print(f"  model device  : {next(model.parameters()).device}")
print("  ✓ model loaded")

print("\nStep 3 — building synthetic 50-candle OHLCV dataframe …")
np.random.seed(42)
n = 50
price = 100_000 + np.cumsum(np.random.randn(n) * 200)
idx   = pd.date_range("2026-01-01", periods=n, freq="5min")
df    = pd.DataFrame({
    "open":   price - np.abs(np.random.randn(n) * 50),
    "high":   price + np.abs(np.random.randn(n) * 100),
    "low":    price - np.abs(np.random.randn(n) * 100),
    "close":  price,
    "volume": np.zeros(n),
    "amount": np.zeros(n),
}, index=idx)

x_ts = df.index.to_series().reset_index(drop=True)
y_ts = pd.Series([df.index[-1] + pd.Timedelta(minutes=5)])

print("\nStep 4 — single MC inference path (this is the crash point) …")
result = predictor.predict(df, x_ts, y_ts, pred_len=1, T=1.0, top_p=0.9,
                           sample_count=1, verbose=False)
predicted_close = float(result["close"].iloc[0])
print(f"  predicted close: ${predicted_close:,.2f}")
print("  ✓ inference succeeded — no segfault")

print("\nStep 5 — 5 MC paths to confirm stability …")
closes = []
for i in range(5):
    r = predictor.predict(df, x_ts, y_ts, pred_len=1, T=1.0, top_p=0.9,
                          sample_count=1, verbose=False)
    closes.append(float(r["close"].iloc[0]))
    print(f"  path {i+1}: ${closes[-1]:,.2f}")

threshold = float(np.mean(price))
prob = float(np.mean(np.array(closes) > threshold))
print(f"\n  P(close > ${threshold:,.0f}) over 5 paths = {prob:.2f}")
print("\n✓ ALL STEPS PASSED — Kronos is safe to run in main.py")
