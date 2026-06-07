import os

import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

_MIN_SAMPLES = 150  # 7 inputs × ~20 samples/feature = 140 minimum; 150 adds margin

_REGIME_ENCODING: dict[str, float] = {
    "trending_up":      1.0,
    "trending_down":   -1.0,
    "ranging":          0.3,   # 53.7% historical WR — mildly trustworthy
    "high_uncertainty": -0.3,  # 45.2% historical WR — mildly penalised
}


def _encode_regime(regime: str | None) -> float:
    return _REGIME_ENCODING.get(regime or "", 0.0)


class Calibrator:
    """
    Quadratic-logistic probability calibrator.

    Fits LogisticRegression on [raw, raw², regime_score?, edge?] features.
    edge = abs(raw_prob - market_price) — how far our signal is from Kalshi
    pricing at entry. Two trades with the same raw_prob but different edges
    have very different Kelly implications; this feature captures that.

    Pass-through when n_samples < _MIN_SAMPLES (not enough data to fit reliably).
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._passthrough: bool = True
        self._n_samples: int = 0
        self._prev_brier: float | None = None
        self._regime_aware: bool = False
        self._edge_aware: bool = False
        self._disagreement_aware: bool = False
        self._volatility_aware: bool = False
        self._spread_aware: bool = False

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def _build_X(
        self,
        raw: np.ndarray,
        regime_scores: np.ndarray | None,
        edges: np.ndarray | None,
        disagreements: np.ndarray | None = None,
        volatilities: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
    ) -> np.ndarray:
        cols = [raw, raw ** 2]
        if regime_scores is not None:
            cols.append(regime_scores)
        if edges is not None:
            cols.append(edges)
        if disagreements is not None:
            cols.append(disagreements)
        if volatilities is not None:
            cols.append(volatilities)
        if spreads is not None:
            cols.append(spreads)
        return np.column_stack(cols)

    def fit(
        self,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        regimes: np.ndarray | None = None,
        edges: np.ndarray | None = None,
        disagreements: np.ndarray | None = None,
        volatilities: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
    ) -> "Calibrator":
        """
        raw_probs     : model output probabilities (regime_prob for Phase 3c)
        outcomes      : binary win/loss labels
        regimes       : DeepSeek regime strings — regime-conditional calibration
        edges         : abs(raw_prob - market_price) at trade time
        disagreements : abs(regime_prob - kronos_raw_15min) — signal agreement check.
                        When the two signals diverge, compress confidence more.
        volatilities  : brti_volatility_1h — high vol = less reliable 15-min snapshot
        spreads       : kalshi_spread_normalized — wide spread = noisier edge signal
        """
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        n = len(raw_probs)
        self._n_samples = n

        use_regime       = regimes       is not None and len(regimes)       == n
        use_edge         = edges         is not None and len(edges)         == n
        use_disagreement = disagreements is not None and len(disagreements) == n
        use_volatility   = volatilities  is not None and len(volatilities)  == n
        use_spread       = spreads       is not None and len(spreads)       == n

        regime_scores    = np.array([_encode_regime(r) for r in regimes], dtype=float) if use_regime else None
        edge_arr         = np.asarray(edges,         dtype=float) if use_edge         else None
        disagreement_arr = np.asarray(disagreements, dtype=float) if use_disagreement else None
        volatility_arr   = np.asarray(volatilities,  dtype=float) if use_volatility   else None
        spread_arr       = np.asarray(spreads,       dtype=float) if use_spread       else None

        n_holdout = max(20, n // 5)
        n_train = n - n_holdout
        if n_train < _MIN_SAMPLES:
            self._passthrough = True
            return self

        def _split(arr): return (arr[n_holdout:], arr[:n_holdout]) if arr is not None else (None, None)

        raw_train, raw_holdout   = raw_probs[n_holdout:], raw_probs[:n_holdout]
        y_train,   y_holdout     = outcomes[n_holdout:],  outcomes[:n_holdout]
        reg_tr,    reg_ho        = _split(regime_scores)
        edg_tr,    edg_ho        = _split(edge_arr)
        dis_tr,    dis_ho        = _split(disagreement_arr)
        vol_tr,    vol_ho        = _split(volatility_arr)
        spr_tr,    spr_ho        = _split(spread_arr)

        X_train   = self._build_X(raw_train,   reg_tr, edg_tr, dis_tr, vol_tr, spr_tr)
        X_holdout = self._build_X(raw_holdout, reg_ho, edg_ho, dis_ho, vol_ho, spr_ho)

        new_model = LogisticRegression(max_iter=1000)
        new_model.fit(X_train, y_train)
        holdout_preds = np.clip(new_model.predict_proba(X_holdout)[:, 1], 0.0, 1.0)
        holdout_brier = float(np.mean((holdout_preds - y_holdout) ** 2))
        passthrough_holdout_brier = float(np.mean((raw_holdout - y_holdout) ** 2))

        prev_model = self._model
        prev_passthrough = self._passthrough
        beats_passthrough = holdout_brier < passthrough_holdout_brier
        beats_prev = self._prev_brier is None or holdout_brier < self._prev_brier

        if beats_passthrough and beats_prev:
            # Direction sanity guard: in the most trusted context (trending_up, low
            # disagreement, calm vol, tight spread), calibrated output must not invert.
            guard_raw = np.array([0.75])
            guard_reg = np.array([_encode_regime("trending_up")]) if use_regime else None
            guard_edg = np.array([0.15])  if use_edge         else None
            guard_dis = np.array([0.0])   if use_disagreement else None  # full agreement
            guard_vol = np.array([0.0])   if use_volatility   else None  # calm market
            guard_spr = np.array([0.0])   if use_spread       else None  # tight spread
            guard_X = self._build_X(guard_raw, guard_reg, guard_edg, guard_dis, guard_vol, guard_spr)
            guard_val = float(np.clip(new_model.predict_proba(guard_X)[:, 1], 0.0, 1.0)[0])
            direction_ok = guard_val >= 0.5
            if not direction_ok:
                logger.warning(
                    f"Calibrator direction guard: transform(0.75, trending_up, edge=0.15, dis=0)"
                    f"={guard_val:.4f} < 0.5 — inverted signal detected. Forcing passthrough."
                )
            if direction_ok:
                self._model = new_model
                self._passthrough = False
                self._prev_brier = holdout_brier
                self._regime_aware       = use_regime
                self._edge_aware         = use_edge
                self._disagreement_aware = use_disagreement
                self._volatility_aware   = use_volatility
                self._spread_aware       = use_spread
            else:
                self._model = prev_model
                self._passthrough = prev_passthrough
                if prev_passthrough:
                    self._prev_brier = passthrough_holdout_brier
        else:
            logger.warning(
                f"Calibrator: holdout Brier {holdout_brier:.4f} vs passthrough "
                f"{passthrough_holdout_brier:.4f}"
                + (f", prev {self._prev_brier:.4f}" if self._prev_brier is not None else "")
                + " — reverting"
            )
            self._model = prev_model
            self._passthrough = prev_passthrough
            if prev_passthrough:
                self._prev_brier = passthrough_holdout_brier

        return self

    def transform(
        self,
        raw_prob: float,
        regime: str | None = None,
        edge: float | None = None,
        disagreement: float | None = None,
        volatility: float | None = None,
        spread: float | None = None,
    ) -> float:
        """
        edge         : abs(raw_prob - market_price) at inference time
        disagreement : abs(regime_prob - kronos_raw_15min) — 0 = full agreement
        volatility   : brti_volatility_1h at inference time
        spread       : kalshi_spread_normalized at inference time
        Missing values default to 0.0 (neutral/best-case context).
        """
        if self._passthrough or self._model is None:
            return float(raw_prob)
        raw = np.array([raw_prob])
        reg = np.array([_encode_regime(regime)]) if self._regime_aware else None
        edg = np.array([edge         if edge         is not None else 0.0]) if self._edge_aware         else None
        dis = np.array([disagreement if disagreement is not None else 0.0]) if self._disagreement_aware else None
        vol = np.array([volatility   if volatility   is not None else 0.0]) if self._volatility_aware   else None
        spr = np.array([spread       if spread       is not None else 0.0]) if self._spread_aware       else None
        X = self._build_X(raw, reg, edg, dis, vol, spr)
        return float(np.clip(self._model.predict_proba(X)[0, 1], 0.0, 1.0))

    def brier_score(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> float:
        raw_probs = np.asarray(raw_probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        calibrated = np.array([self.transform(float(p)) for p in raw_probs])
        return float(np.mean((calibrated - outcomes) ** 2))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model": self._model,
            "passthrough": self._passthrough,
            "n_samples": self._n_samples,
            "prev_brier": self._prev_brier,
            "regime_aware":       self._regime_aware,
            "edge_aware":         self._edge_aware,
            "disagreement_aware": self._disagreement_aware,
            "volatility_aware":   self._volatility_aware,
            "spread_aware":       self._spread_aware,
        }, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator model not found: {path}")
        state = joblib.load(path)
        obj = cls.__new__(cls)
        obj._model = state.get("model", state.get("iso"))
        obj._passthrough = state["passthrough"]
        obj._n_samples = state.get("n_samples", 0)
        obj._prev_brier = state.get("prev_brier", None)
        obj._regime_aware       = state.get("regime_aware",       False)
        obj._edge_aware         = state.get("edge_aware",         False)
        obj._disagreement_aware = state.get("disagreement_aware", False)
        obj._volatility_aware   = state.get("volatility_aware",   False)
        obj._spread_aware       = state.get("spread_aware",       False)
        return obj
