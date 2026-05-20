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

    def _load(self) -> None:
        if self._predictor is not None:
            return
        import torch
        from kronos_model import Kronos, KronosPredictor, KronosTokenizer

        # map_location="cpu" forces weights to land on CPU during torch.load() inside
        # from_pretrained(), BEFORE KronosPredictor gets a chance to set the device.
        # Without this, PyTorchModelHubMixin loads to MPS on Apple Silicon, and the
        # subsequent .to("cpu") move triggers a segfault in torch.multinomial.
        logger.info("KronosEngine: loading model weights directly to CPU …")
        model = Kronos.from_pretrained(self._model_name, map_location="cpu")
        tokenizer = KronosTokenizer.from_pretrained(
            "NeoQuasar/Kronos-Tokenizer-base", map_location="cpu"
        )
        # Single-threaded to avoid Accelerate-framework threading conflicts on macOS.
        torch.set_num_threads(1)
        self._predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        logger.info("KronosEngine: model ready on CPU")

    def run_monte_carlo(
        self,
        store: FeatureStore,
        n_paths: int = 100,
        threshold: float = 76548.76,
    ) -> float:
        """
        Pull the last 400 5-min BRTI candles, run n_paths MC inference paths,
        return P(predicted_close > threshold) at the next 5-min resolution window.

        Raises ValueError if fewer than _MIN_CANDLES 5-min candles are available.
        """
        df = store.get_ohlcv("5min")
        if df is None or len(df) < _MIN_CANDLES:
            raise ValueError(
                f"Insufficient OHLCV data: need >={_MIN_CANDLES} 5-min candles, "
                f"got {0 if df is None else len(df)}"
            )

        df = df.tail(400)
        if len(df) < 400:
            logger.warning(f"KronosEngine: only {len(df)} 5-min candles available (recommend >=400)")

        self._load()

        # predict() requires Series types, not bare Timestamps
        x_timestamp = df.index.to_series().reset_index(drop=True)
        y_timestamp = pd.Series([df.index[-1] + pd.Timedelta(minutes=5)])

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
