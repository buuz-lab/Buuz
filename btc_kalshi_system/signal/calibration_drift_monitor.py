"""
CalibrationDriftMonitor — rolling Brier-score drift detector.

Detects calibration drift when the system transitions from paper trading to
live trading. The calibrator is fit on paper-trade outcomes; real fills can
shift the distribution. This detects the shift before the circuit breaker
catches it.

Rolling 20-trade Brier score window. Alerts when score exceeds 1.3× the
baseline for 3 consecutive windows. State is persisted to Redis so it
survives process restarts.

Brier score per trade: (calibrated_prob - outcome)²
Baseline: mean Brier over first DRIFT_WINDOW recorded trades (paper-trade era).

Windows are non-overlapping: window 1 = trades 1-20, window 2 = trades 21-40, etc.
A total-trade counter is persisted alongside history so window boundaries survive
process restarts.
"""

import json
from collections import deque
from typing import Deque

import redis
from loguru import logger

from config import REDIS_URL

DRIFT_WINDOW = 20
DRIFT_ALERT_MULTIPLIER = 1.3
DRIFT_CONSECUTIVE_ALERT = 3

_KEY_HISTORY = "calibration_drift:history"
_KEY_BASELINE = "calibration_drift:baseline_brier"
_KEY_ALERT_COUNT = "calibration_drift:alert_count"
_KEY_TOTAL_COUNT = "calibration_drift:total_count"


class CalibrationDriftMonitor:
    """
    Rolling 20-trade Brier score window. Alerts when score exceeds
    1.3× the baseline for 3 consecutive windows. Persists to Redis.
    """

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._history: Deque[dict] = deque(maxlen=DRIFT_WINDOW)
        self._total_count: int = 0
        self._load_from_redis()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, calibrated_prob: float, outcome: int) -> None:
        """Append to rolling history, persist, recompute if window is full."""
        self._history.append(
            {"calibrated_prob": float(calibrated_prob), "outcome": int(outcome)}
        )
        self._total_count += 1
        self._persist_history()

        # Only recompute at non-overlapping window boundaries (every DRIFT_WINDOW trades)
        if self._total_count % DRIFT_WINDOW == 0:
            self._recompute_window()

    def current_brier(self) -> float:
        """Mean Brier over current window. 0.0 when < DRIFT_WINDOW trades recorded."""
        if len(self._history) < DRIFT_WINDOW:
            return 0.0
        return _mean_brier(self._history)

    def is_drifting(self) -> bool:
        """True when alert_count >= DRIFT_CONSECUTIVE_ALERT."""
        return self._get_alert_count() >= DRIFT_CONSECUTIVE_ALERT

    def reset_baseline(self) -> None:
        """Call after calibrator refit so baseline reflects new calibration era."""
        try:
            self._redis.delete(_KEY_BASELINE, _KEY_ALERT_COUNT, _KEY_TOTAL_COUNT, _KEY_HISTORY)
            self._history.clear()
            self._total_count = 0
            logger.info("CalibrationDriftMonitor: baseline reset after calibrator refit")
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: reset failed — {exc}")

    def baseline_brier(self) -> float | None:
        """None until first window fills."""
        try:
            raw = self._redis.get(_KEY_BASELINE)
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: Redis error reading baseline — {exc}")
            return None
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _recompute_window(self) -> None:
        """Called at each non-overlapping window boundary."""
        if not self._history:
            return
        brier = _mean_brier(self._history)
        baseline = self.baseline_brier()

        if baseline is None:
            # First window fills — this is the paper-trade baseline.
            self._set_baseline(brier)
            self._set_alert_count(0)
            logger.debug(
                f"CalibrationDriftMonitor: baseline set — brier={brier:.4f}"
            )
            return

        logger.debug(
            f"CalibrationDriftMonitor: window complete — "
            f"current_brier={brier:.4f} baseline={baseline:.4f}"
        )

        if brier > baseline * DRIFT_ALERT_MULTIPLIER:
            new_count = self._get_alert_count() + 1
            self._set_alert_count(new_count)
            logger.warning(
                f"CalibrationDriftMonitor: brier {brier:.4f} exceeds "
                f"{DRIFT_ALERT_MULTIPLIER}× baseline {baseline:.4f} "
                f"(alert_count={new_count})"
            )
        else:
            self._set_alert_count(0)

    def _get_alert_count(self) -> int:
        try:
            raw = self._redis.get(_KEY_ALERT_COUNT)
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: Redis error reading alert_count — {exc}")
            return 0
        if raw is None:
            return 0
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0

    def _set_alert_count(self, count: int) -> None:
        try:
            self._redis.set(_KEY_ALERT_COUNT, str(count))
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: failed to persist alert_count — {exc}")

    def _set_baseline(self, brier: float) -> None:
        try:
            self._redis.set(_KEY_BASELINE, str(brier))
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: failed to persist baseline — {exc}")

    def _persist_history(self) -> None:
        payload = list(self._history)
        try:
            self._redis.set(_KEY_HISTORY, json.dumps(payload))
            self._redis.set(_KEY_TOTAL_COUNT, str(self._total_count))
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: failed to persist history — {exc}")

    def _load_from_redis(self) -> None:
        """Best-effort load of prior history; silently start empty on any failure."""
        try:
            raw_history = self._redis.get(_KEY_HISTORY)
            raw_count = self._redis.get(_KEY_TOTAL_COUNT)
        except redis.RedisError as exc:
            logger.warning(f"CalibrationDriftMonitor: Redis unreachable on load — {exc}")
            return

        if raw_count is not None:
            try:
                self._total_count = int(raw_count)
            except (ValueError, TypeError):
                self._total_count = 0

        if raw_history is None:
            return
        try:
            entries = json.loads(raw_history)
            for entry in entries:
                self._history.append(
                    {
                        "calibrated_prob": float(entry["calibrated_prob"]),
                        "outcome": int(entry["outcome"]),
                    }
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                f"CalibrationDriftMonitor: corrupt history in Redis, starting empty — {exc}"
            )
            self._history.clear()
            self._total_count = 0


# ── Module-level helper ────────────────────────────────────────────────────────

def _mean_brier(history) -> float:
    """Mean Brier score over a sequence of {calibrated_prob, outcome} dicts."""
    scores = [(e["calibrated_prob"] - e["outcome"]) ** 2 for e in history]
    return sum(scores) / len(scores)
