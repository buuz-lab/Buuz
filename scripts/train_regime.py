"""
Train the RegimeModel from instrumented trades in trades.db.

Run this script once you have ≥500 resolved trades with captured regime features
(check `python3 scripts/regime_training_progress.py` if available, or run this
script — it will refuse to train when the qualifying-row count is too low).

Usage:
    python3 scripts/train_regime.py [--db trades.db] [--out models/regime.pkl]
                                    [--test-size 100] [--min-rows 500] [--dry-run]

Filtering rules
---------------
A trade row qualifies for training iff:
    features_stale  = 0          (Redis regime:features was fresh at trade time)
    funding_rate    IS NOT NULL  (excludes pre-instrumentation rows from before
                                  the schema migration that added these columns)
    outcome         IS NOT NULL  (trade has been resolved by Kalshi)

Label semantics
---------------
We're training to predict "did the 15-min BTC market close UP" — NOT "did this
particular trade win." Those are different on short trades:

    up_label = 1 if (direction == outcome) else 0

  • direction=1 (we bet up), outcome=1 (won)  → market went up  → label 1
  • direction=1 (we bet up), outcome=0 (lost) → market went down → label 0
  • direction=0 (we bet down), outcome=1 (won)  → market went down → label 0
  • direction=0 (we bet down), outcome=0 (lost) → market went up  → label 1

Train/test split
----------------
Time-ordered. The last `--test-size` rows are held out for evaluation; everything
older is used for training. Random splits would leak regime structure (crypto
regimes persist across consecutive trades).

Class balance
-------------
We pass `scale_pos_weight = neg / pos` to XGBoost only when the train label ratio
drifts outside the 35/65 band. Inside that band, the default weighting is fine
and adding a weight just adds noise.

Metrics reported
----------------
    Brier score       — calibration quality (0.25 = coin flip; lower is better)
    Accuracy          — fraction of test rows correctly classified
    Kronos agreement  — fraction of test rows where the regime model's direction
                        equals the Kronos direction. Informational: a very high
                        number means Gate 2 will rarely block anything; very low
                        means the model is fighting Kronos and you should NOT
                        flip REGIME_GATE2_ENFORCING to True until you understand
                        why.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Make the project root importable when run as `python3 scripts/train_regime.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.regime_model import RegimeModel

# Must match the feature key order RegimeModel.get_regime() expects so the
# trained model's column order matches the inference-time dict iteration order.
_FEATURE_COLS = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
]

_QUERY = f"""
SELECT {", ".join(_FEATURE_COLS)},
       direction, outcome, kronos_calibrated, timestamp
FROM trades
WHERE features_stale = 0
  AND funding_rate IS NOT NULL
  AND outcome IS NOT NULL
ORDER BY timestamp ASC
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db", help="Path to trades.db (default: trades.db)")
    p.add_argument("--out", default=config.REGIME_MODEL_PATH,
                   help=f"Output path for the trained model (default: {config.REGIME_MODEL_PATH})")
    p.add_argument("--test-size", type=int, default=100,
                   help="Number of most-recent trades to hold out for evaluation (default: 100)")
    p.add_argument("--min-rows", type=int, default=500,
                   help="Minimum qualifying rows required to train (default: 500)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report metrics but do NOT write the model file.")
    return p.parse_args()


def load_dataset(db_path: str) -> np.ndarray:
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(_QUERY).fetchall()
    finally:
        conn.close()
    return rows


def build_xy(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_up, kronos_cal) where y_up=1 iff the underlying market closed
    UP (regardless of which side we bet).
    """
    arr = np.array(rows, dtype=object)
    X = arr[:, : len(_FEATURE_COLS)].astype(np.float64)
    direction = arr[:, len(_FEATURE_COLS)].astype(int)
    outcome = arr[:, len(_FEATURE_COLS) + 1].astype(int)
    kronos_cal = arr[:, len(_FEATURE_COLS) + 2].astype(np.float64)
    y_up = (direction == outcome).astype(int)
    return X, y_up, kronos_cal


def maybe_scale_pos_weight(y: np.ndarray) -> dict:
    """
    Return a kwargs dict containing scale_pos_weight when the class ratio is
    outside [35/65, 65/35], otherwise empty so we don't override XGBoost defaults.
    """
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        sys.exit(f"Degenerate label distribution: pos={pos} neg={neg}. Refusing to train.")
    pos_frac = pos / (pos + neg)
    # Outside [0.35, 0.65] → impose the weight
    if pos_frac < 0.35 or pos_frac > 0.65:
        return {"scale_pos_weight": neg / pos}
    return {}


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true) ** 2))


def main() -> None:
    args = parse_args()

    rows = load_dataset(args.db)
    n_total = len(rows)
    print(f"Qualifying rows in {args.db}: {n_total}")

    if n_total < args.min_rows:
        sys.exit(
            f"Need ≥{args.min_rows} qualifying rows to train; have {n_total}. "
            f"Continue running paper trading and re-run this script later."
        )
    if n_total <= args.test_size + 50:
        sys.exit(
            f"Not enough rows to leave both a train set (>50) and a "
            f"test set ({args.test_size}). Increase --min-rows or wait for more data."
        )

    X, y_up, kronos_cal = build_xy(rows)

    # Time-based split — the data is already ORDER BY timestamp ASC.
    X_train, X_test = X[: -args.test_size], X[-args.test_size :]
    y_train, y_test = y_up[: -args.test_size], y_up[-args.test_size :]
    kronos_test = kronos_cal[-args.test_size :]

    pos_tr, neg_tr = int(y_train.sum()), int(len(y_train) - y_train.sum())
    pos_te, neg_te = int(y_test.sum()), int(len(y_test) - y_test.sum())
    print(f"Train: {len(y_train)} rows  (up={pos_tr}, down={neg_tr})")
    print(f"Test : {len(y_test)} rows  (up={pos_te}, down={neg_te})")

    extra_kwargs = maybe_scale_pos_weight(y_train)
    if extra_kwargs:
        print(f"Applying scale_pos_weight={extra_kwargs['scale_pos_weight']:.3f} "
              f"(train class balance is outside [35%, 65%])")
    else:
        print("Train class balance is within [35%, 65%] — no scale_pos_weight applied.")

    model = RegimeModel()
    model.train(X_train, y_train, **extra_kwargs)

    # ── Evaluation ───────────────────────────────────────────────────────────
    proba_test = model._clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    brier = brier_score(y_test, proba_test)
    accuracy = float((pred_test == y_test).mean())

    # Kronos calibrated probability is also a P(close > strike) estimate, so the
    # implied Kronos direction is comparable to the regime model's direction.
    kronos_dir_test = (kronos_test >= 0.5).astype(int)
    kronos_agreement = float((pred_test == kronos_dir_test).mean())

    print()
    print(f"Test Brier score    : {brier:.4f}   (0.25 = coin flip; lower is better)")
    print(f"Test accuracy       : {accuracy:.4f}")
    print(f"Kronos agreement %  : {kronos_agreement:.4f}   (informational)")
    print()
    if kronos_agreement < 0.55:
        print("WARNING: Kronos agreement < 55%. The regime model is contradicting Kronos")
        print("         on nearly half of trades. Investigate before enabling Gate 2")
        print("         enforcement — flipping REGIME_GATE2_ENFORCING=True now would")
        print("         block roughly that fraction of trades.")
    if brier > 0.25:
        print("WARNING: Brier > 0.25 (worse than a coin flip). Do NOT deploy this model.")

    if args.dry_run:
        print("\n--dry-run set — model NOT saved.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"\nSaved regime model to: {out_path}")
    print("Restart KronosV2 to pick it up. Gate 2 will run in SHADOW mode by default")
    print("(config.REGIME_GATE2_ENFORCING=False) — observe disagreement logs for ~50")
    print("trades, then flip to True to enable enforcement.")


if __name__ == "__main__":
    main()
