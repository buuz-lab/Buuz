"""
Standalone Kalshi REST client using RSA-PSS auth (SHA-256).
Works independently of pykalshi — intended as a failover when pykalshi breaks.
"""

import base64
import logging
import time
import uuid
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

import config

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiRawClient:
    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        self._api_key_id = api_key_id or config.KALSHI_API_KEY_ID
        key_path = Path(private_key_path or config.KALSHI_PRIVATE_KEY_PATH)

        if not key_path.exists():
            raise FileNotFoundError(f"Kalshi private key not found: {key_path}")

        try:
            pem = key_path.read_bytes()
            self._private_key = serialization.load_pem_private_key(pem, password=None)
        except Exception as exc:
            raise ValueError(f"Failed to load Kalshi private key: {exc}") from exc

        self._base_url = "https://api.elections.kalshi.com"
        self._session = requests.Session()

    def _sign(self, method: str, path: str) -> dict:
        ts_ms = str(int(time.time() * 1000))
        message = (ts_ms + method.upper() + path).encode()
        sig = self._private_key.sign(
            message,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = self._sign(method, path)
        url = self._base_url + path
        logger.debug("%s %s", method.upper(), path)
        try:
            resp = self._session.request(
                method.upper(),
                url,
                headers=headers,
                json=body,
                timeout=10,
            )
            logger.debug("%s %s → %d", method.upper(), path, resp.status_code)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Kalshi HTTP error: %s %s → %d — %s",
                method.upper(),
                path,
                exc.response.status_code,
                exc.response.text,
            )
            raise requests.HTTPError(
                f"Kalshi {method.upper()} {path} failed with "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_orderbook(self, ticker: str) -> dict:
        return self._request("GET", f"/trade-api/v2/markets/{ticker}/orderbook")

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        order_type: str = "limit",
        client_order_id: str | None = None,
    ) -> dict:
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())
        yes_price = price_cents if side == "yes" else 100 - price_cents
        no_price = 100 - price_cents if side == "yes" else price_cents
        body = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "type": order_type,
            "yes_price": yes_price,
            "no_price": no_price,
            "client_order_id": client_order_id,
        }
        return self._request("POST", "/trade-api/v2/portfolio/orders", body=body)

    def get_order(self, order_id: str) -> dict:
        return self._request("GET", f"/trade-api/v2/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")

    def get_positions(self) -> dict:
        return self._request("GET", "/trade-api/v2/portfolio/positions")

    def get_balance(self) -> dict:
        return self._request("GET", "/trade-api/v2/portfolio/balance")
