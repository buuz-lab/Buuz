import threading
import time
from enum import Enum
from typing import Optional

from loguru import logger

import config
from btc_kalshi_system.execution.raw_http_client import KalshiRawClient


class ClientState(Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"
    BOTH_FAILED = "both_failed"


class KalshiClientRouter:
    _FAILURE_THRESHOLD: int = 3
    _RECOVERY_INTERVAL: float = 300.0  # 5 minutes

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
    ) -> None:
        self._api_key_id = api_key_id or config.KALSHI_API_KEY_ID
        self._private_key_path = private_key_path or config.KALSHI_PRIVATE_KEY_PATH

        try:
            self._raw = KalshiRawClient(
                api_key_id=self._api_key_id,
                private_key_path=self._private_key_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            # Key missing or malformed — log clearly and re-raise so the operator
            # sees an actionable message rather than an empty log file.
            logger.critical(
                f"KalshiRawClient failed to initialise: {exc}\n"
                "Check KALSHI_PRIVATE_KEY_PATH in .env and ensure the file exists."
            )
            raise

        self._primary = self._init_pykalshi()
        self._state: ClientState = (
            ClientState.PRIMARY if self._primary else ClientState.FALLBACK
        )
        self._consecutive_failures: int = 0
        self._last_recovery_attempt: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Init / recovery
    # ------------------------------------------------------------------

    def _init_pykalshi(self):
        try:
            from pykalshi import KalshiClient
            client = KalshiClient(
                api_key_id=self._api_key_id,
                private_key_path=self._private_key_path,
            )
            logger.info("pykalshi client initialized successfully")
            return client
        except (ImportError, Exception) as exc:
            logger.warning(f"pykalshi unavailable: {exc} — starting in fallback mode")
            return None

    def _maybe_attempt_recovery(self) -> None:
        with self._lock:
            if self._state is ClientState.PRIMARY:
                return
            if time.time() - self._last_recovery_attempt < self._RECOVERY_INTERVAL:
                return
            self._last_recovery_attempt = time.time()

        client = self._init_pykalshi()
        if client is not None:
            with self._lock:
                self._primary = client
                self._state = ClientState.PRIMARY
                self._consecutive_failures = 0
            logger.info("pykalshi recovered — switched back to PRIMARY")
        else:
            logger.warning("pykalshi recovery attempt failed — staying in current state")

    def _handle_primary_failure(self, exc: Exception) -> None:
        with self._lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            logger.warning(
                f"pykalshi failure #{failures}: {exc}"
            )
            if failures >= self._FAILURE_THRESHOLD:
                self._state = ClientState.FALLBACK
                logger.error(
                    f"pykalshi failed {failures} times consecutively — "
                    "switching to FALLBACK (raw HTTP)"
                )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route_through_primary(self, method: str, **kwargs) -> dict:
        from pykalshi import Action, Side

        if method == "place_order":
            ticker = kwargs["ticker"]
            side = Side.YES if kwargs["side"] == "yes" else Side.NO
            count_fp = str(kwargs["count"])
            price_dollars = f"{kwargs['price_cents'] / 100:.2f}"
            client_order_id = kwargs.get("client_order_id")

            order = self._primary.portfolio.place_order(
                ticker,
                Action.BUY,
                side,
                count_fp,
                yes_price_dollars=price_dollars if kwargs["side"] == "yes" else None,
                no_price_dollars=price_dollars if kwargs["side"] == "no" else None,
                client_order_id=client_order_id,
            )
            return order.data.model_dump()

        if method == "get_positions":
            positions = self._primary.portfolio.get_positions()
            return {"market_positions": [p.model_dump() for p in positions]}

        if method == "get_balance":
            return self._primary.portfolio.get_balance().model_dump()

        if method == "cancel_order":
            order = self._primary.portfolio.cancel_order(kwargs["order_id"])
            return order.data.model_dump()

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ClientState:
        return self._state

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        client_order_id: str | None = None,
    ) -> dict:
        self._maybe_attempt_recovery()

        with self._lock:
            state = self._state

        if state is ClientState.BOTH_FAILED:
            raise RuntimeError("Both Kalshi clients failed")

        if state is ClientState.PRIMARY:
            try:
                result = self._route_through_primary(
                    "place_order",
                    ticker=ticker,
                    side=side,
                    count=count,
                    price_cents=price_cents,
                    client_order_id=client_order_id,
                )
                with self._lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                self._handle_primary_failure(exc)

        # FALLBACK
        try:
            return self._raw.place_order(
                ticker=ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            with self._lock:
                self._state = ClientState.BOTH_FAILED
            logger.error(f"Raw HTTP client also failed: {exc} — state: BOTH_FAILED")
            raise RuntimeError("Both Kalshi clients failed") from exc

    def get_orderbook(self, ticker: str) -> dict:
        # Always use raw HTTP — no reason to route through pykalshi
        return self._raw.get_orderbook(ticker)

    def get_positions(self) -> dict:
        self._maybe_attempt_recovery()

        with self._lock:
            state = self._state

        if state is ClientState.BOTH_FAILED:
            raise RuntimeError("Both Kalshi clients failed")

        if state is ClientState.PRIMARY:
            try:
                result = self._route_through_primary("get_positions")
                with self._lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                self._handle_primary_failure(exc)

        try:
            return self._raw.get_positions()
        except Exception as exc:
            with self._lock:
                self._state = ClientState.BOTH_FAILED
            logger.error(f"Raw HTTP client also failed: {exc} — state: BOTH_FAILED")
            raise RuntimeError("Both Kalshi clients failed") from exc

    def get_balance(self) -> dict:
        self._maybe_attempt_recovery()

        with self._lock:
            state = self._state

        if state is ClientState.BOTH_FAILED:
            raise RuntimeError("Both Kalshi clients failed")

        if state is ClientState.PRIMARY:
            try:
                result = self._route_through_primary("get_balance")
                with self._lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                self._handle_primary_failure(exc)

        try:
            return self._raw.get_balance()
        except Exception as exc:
            with self._lock:
                self._state = ClientState.BOTH_FAILED
            logger.error(f"Raw HTTP client also failed: {exc} — state: BOTH_FAILED")
            raise RuntimeError("Both Kalshi clients failed") from exc

    def cancel_order(self, order_id: str) -> dict:
        self._maybe_attempt_recovery()

        with self._lock:
            state = self._state

        if state is ClientState.BOTH_FAILED:
            raise RuntimeError("Both Kalshi clients failed")

        if state is ClientState.PRIMARY:
            try:
                result = self._route_through_primary("cancel_order", order_id=order_id)
                with self._lock:
                    self._consecutive_failures = 0
                return result
            except Exception as exc:
                self._handle_primary_failure(exc)

        try:
            return self._raw.cancel_order(order_id)
        except Exception as exc:
            with self._lock:
                self._state = ClientState.BOTH_FAILED
            logger.error(f"Raw HTTP client also failed: {exc} — state: BOTH_FAILED")
            raise RuntimeError("Both Kalshi clients failed") from exc
