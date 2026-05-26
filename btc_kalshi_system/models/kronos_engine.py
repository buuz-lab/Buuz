import numpy as np
import pandas as pd
from loguru import logger

from btc_kalshi_system.data.feature_store import FeatureStore

_MIN_CANDLES = 10


class KronosEngine:
    """
    Wraps KronosPredictor for BRTI resolution-window forecasting.

    Usage:
        engine = KronosEngine()
        prob = engine.run_monte_carlo(store, n_paths=100, threshold=76548.76)
    """

    def __init__(self, model_name: str = "NeoQuasar/Kronos-small") -> None:
        self._model_name = model_name
        self._predictor = None  # lazy-loaded on first call

    def preload(self) -> None:
        """Eagerly load the model.

        Call this BEFORE starting the asyncio event loop (e.g. in KronosV2.__init__).
        Loading inside a worker thread while WebSocket feeds run concurrently triggers
        a segfault on Apple Silicon because PyTorch's Accelerate-framework initialisation
        is not safe under concurrent macOS kqueue I/O.  Loading here — single-threaded,
        no event loop — is unconditionally safe.
        """
        self._load()

    def _load(self) -> None:
        if self._predictor is not None:
            return
        import torch
        from kronos_model import Kronos, KronosPredictor, KronosTokenizer

        # set_num_threads MUST come before any torch operation (including from_pretrained).
        # In the test script this was called first and inference succeeded; calling it
        # after from_pretrained races with Accelerate's internal thread pool init.
        torch.set_num_threads(1)

        logger.info("KronosEngine: loading model weights to CPU …")
        model = Kronos.from_pretrained(self._model_name, map_location="cpu")
        tokenizer = KronosTokenizer.from_pretrained(
            "NeoQuasar/Kronos-Tokenizer-base", map_location="cpu"
        )
        self._predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        logger.info("KronosEngine: model ready on CPU")

    def run_monte_carlo(
        self,
        store: FeatureStore,
        n_paths: int = 100,
        threshold: float = 76548.76,
        candle_freq: str = "5min",
    ) -> float:
        """
        Pull the last 400 BRTI candles at candle_freq resolution, run n_paths MC
        inference paths, return P(predicted_close > threshold) at the next candle.

        candle_freq controls the prediction horizon:
          "5min"  → predicts next 5-min close  (default, original behaviour)
          "15min" → predicts next 15-min close (aligned with 15-min Kalshi markets)
          "1h"    → predicts next 1-hour close (aligned with 1-hour Kalshi markets)

        Raises ValueError if fewer than _MIN_CANDLES candles are available.
        """
        _FREQ_DELTA = {"5min": pd.Timedelta(minutes=5), "15min": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
        if candle_freq not in _FREQ_DELTA:
            raise ValueError(f"Unsupported candle_freq: {candle_freq!r}. Use '5min', '15min', or '1h'.")

        df = store.get_ohlcv(candle_freq)
        if df is None or len(df) < _MIN_CANDLES:
            raise ValueError(
                f"Insufficient OHLCV data: need >={_MIN_CANDLES} {candle_freq} candles, "
                f"got {0 if df is None else len(df)}"
            )

        df = df.tail(400)
        if len(df) < 400:
            logger.warning(f"KronosEngine: only {len(df)} {candle_freq} candles available (recommend >=400)")

        self._load()

        # predict() requires Series types, not bare Timestamps
        x_timestamp = df.index.to_series().reset_index(drop=True)
        y_timestamp = pd.Series([df.index[-1] + _FREQ_DELTA[candle_freq]])

        # sample_count>1 averages paths internally — call n_paths times with sample_count=1
        predicted_closes = []
        for _ in range(n_paths):
            row = self._predictor.predict(
                df, x_timestamp, y_timestamp,
                pred_len=1, T=1.0, top_p=0.9, sample_count=1,
                verbose=False,
            )
            predicted_closes.append(float(row["close"].iloc[0]))
        predicted_closes = np.array(predicted_closes)

        prob = float(np.mean(predicted_closes > threshold))

        print(f"\n{'='*55}")
        print(f"Kronos MC Inference — {self._model_name}")
        print(f"Input:  {len(df)} × 5-min candles  ({df.index[0]} → {df.index[-1]})")
        print(f"Target: {y_timestamp.iloc[0]}")
        print(f"MC paths: {n_paths}")
        print(f"Predicted close — min=${predicted_closes.min():,.2f}  "
              f"mean=${predicted_closes.mean():,.2f}  max=${predicted_closes.max():,.2f}")
        print(f"Threshold: ${threshold:,.2f}")
        print(f"P(close > threshold) = {prob:.4f}  ({prob*100:.1f}%)")
        print(f"{'='*55}\n")

        return prob
