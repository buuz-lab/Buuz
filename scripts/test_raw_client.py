"""
Smoke test for KalshiRawClient — reads balance and positions only, no orders placed.
Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars (or .env file).
"""

import json
import sys

from btc_kalshi_system.execution import KalshiRawClient


def main() -> None:
    client = KalshiRawClient()

    print("--- Balance ---")
    balance = client.get_balance()
    print(json.dumps(balance, indent=2))

    print("\n--- Positions ---")
    positions = client.get_positions()
    print(json.dumps(positions, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
