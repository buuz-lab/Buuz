import numpy as np
import pandas as pd

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
        from kronos_model import Kronos, KronosPredictor, KronosTokenizer
        model = Kronos.from_pretrained(self._model_name)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self._predictor = KronosPredictor(model, tokenizer, max_context=512)

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
        self._load()

        x_timestamp = df.index[-1]
        y_timestamp = x_timestamp + pd.Timedelta(minutes=5)

        samples = self._predictor.predict(
            df,
            x_timestamp,
            y_timestamp,
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=n_paths,
        )

        predicted_closes = samples["close"].values
        prob = float(np.mean(predicted_closes > threshold))

        print(f"\n{'='*55}")
        print(f"Kronos MC Inference — {self._model_name}")
        print(f"Input:  {len(df)} × 5-min candles  ({df.index[0]} → {x_timestamp})")
        print(f"Target: {y_timestamp}")
        print(f"MC paths: {n_paths}")
        print(f"Predicted close — min=${predicted_closes.min():,.2f}  "
              f"mean=${predicted_closes.mean():,.2f}  max=${predicted_closes.max():,.2f}")
        print(f"Threshold: ${threshold:,.2f}")
        print(f"P(close > threshold) = {prob:.4f}  ({prob*100:.1f}%)")
        print(f"{'='*55}\n")

        return prob
