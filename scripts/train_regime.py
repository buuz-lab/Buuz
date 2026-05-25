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

# Must match _FEATURE_ORDER in regime_model.py and the keys returned by
# fusion._regime_features() — a mismatch silently corrupts the trained model.
_FEATURE_COLS = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
    "cvd_velocity",
    "cvd_acceleration",
    "brti_momentum_5min",
    "brti_momentum_15min",
    "candle_progress",
    "hour_sin",
    "hour_cos",
    "kalshi_implied_prob",
    "funding_window_proximity",
    "trend_slope_1h",
    "trend_r2_1h",
    "hourly_sr_proximity",
    "range_breakout_flag",
    "tape_speed_tpm",
    "large_print_direction",
    # Features 22-27: Deribit options + Kalshi spread (session 6)
    "atm_iv",
    "iv_rv_spread",
    "pcr_oi",
    "term_structure_slope",
    "skew_25d",
    "kalshi_spread_normalized",
]

_FEATURE_COLS_LEGACY = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
]

_QUERY_TEMPLATE = """
SELECT {cols},
       direction, outcome, kronos_calibrated, timestamp
FROM trades
WHERE features_stale = 0
  AND funding_rate IS NOT NULL
  {extra_filters}
  AND outcome IS NOT NULL
ORDER BY timestamp ASC
"""

_EXTRA_FILTERS_20 = """AND cvd_velocity IS NOT NULL
  AND brti_momentum_5min IS NOT NULL
  AND kalshi_implied_prob IS NOT NULL
  AND funding_window_proximity IS NOT NULL
  AND large_print_direction IS NOT NULL
  AND COALESCE(okx_stale, 0) = 0"""

_EXTRA_FILTERS_27 = _EXTRA_FILTERS_20 + "\n  AND deribit_stale = 0\n  AND atm_iv IS NOT NULL"

def _build_query(legacy: bool, use_27: bool = False) -> str:
    if legacy:
        cols = ", ".join(_FEATURE_COLS_LEGACY)
        return _QUERY_TEMPLATE.format(cols=cols, extra_filters="")
    cols = ", ".join(_FEATURE_COLS)
    extra = _EXTRA_FILTERS_27 if use_27 else _EXTRA_FILTERS_20
    return _QUERY_TEMPLATE.format(cols=cols, extra_filters=extra)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db", help="Path to trades.db (default: trades.db)")
    p.add_argument("--out", default=config.REGIME_MODEL_PATH,
                   help=f"Output path for the trained model (default: {config.REGIME_MODEL_PATH})")
    p.add_argument("--test-size", type=int, default=100,
                   help="Number of most-recent trades to hold out for evaluation (default: 100)")
    p.add_argument("--min-rows", type=int, default=500,
                   help="Minimum qualifying rows required to train (default: 500)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="If set, use only the most recent N qualifying rows for training.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report metrics but do NOT write the model file.")
    p.add_argument("--force", action="store_true",
                   help="Skip the low-variance feature gate and train anyway.")
    p.add_argument("--legacy", action="store_true",
                   help="Use only the original 6 features. Allows training on rows collected "
                        "before Phase 1 (no new IS NOT NULL filters). Pair with --force --min-rows 100.")
    return p.parse_args()


def load_dataset(db_path: str, max_rows: int | None = None, legacy: bool = False) -> list[tuple]:
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    # _FEATURE_COLS now has 27 entries (including Deribit options features 22–27).
    # Always use _EXTRA_FILTERS_27 so the query requires deribit_stale=0 and
    # atm_iv IS NOT NULL. Without this, rows where features 22–27 are NULL
    # (deribit_stale=1) would be included, and XGBoost would learn that those
    # features are "always missing" at training time — but inference provides real
    # values, causing a silent model corruption. The query will return 0 rows
    # until ~500 deribit_stale=0 rows have accumulated; that is the correct gate.
    query = _build_query(legacy, use_27=not legacy)
    conn = sqlite3.connect(db_path)
    try:
        if max_rows is not None:
            rows = conn.execute(
                query.replace("ORDER BY timestamp ASC",
                              f"ORDER BY timestamp DESC LIMIT {max_rows}")
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    return rows


def build_xy(rows: list[tuple], n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_up, kronos_cal) where y_up=1 iff the underlying market closed
    UP (regardless of which side we bet).
    """
    arr = np.array(rows, dtype=object)
    X = arr[:, :n_features].astype(np.float64)
    direction = arr[:, n_features].astype(int)
    outcome = arr[:, n_features + 1].astype(int)
    kronos_cal = arr[:, n_features + 2].astype(np.float64)
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

    active_cols = _FEATURE_COLS_LEGACY if args.legacy else _FEATURE_COLS
    if args.legacy:
        print("--legacy mode: using original 6 features (no new IS NOT NULL filters)")

    rows = load_dataset(args.db, max_rows=args.max_rows, legacy=args.legacy)
    n_total = len(rows)
    print(f"Qualifying rows in {args.db}: {n_total}")
    if args.max_rows is not None:
        print(f"--max-rows {args.max_rows}: using most recent {n_total} qualifying rows")

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

    X, y_up, kronos_cal = build_xy(rows, n_features=len(active_cols))

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

    # ── Feature variance gate ─────────────────────────────────────────────────
    low_variance_features: list[tuple[str, float]] = []
    for i, feat in enumerate(active_cols):
        std = float(X_train[:, i].std())
        if std < 1e-6:
            print(f"WARNING: feature '{feat}' has near-zero std in X_train: {std:.2e}")
            low_variance_features.append((feat, std))
    # tolerate up to 2 near-zero-std features; more than 2 suggests a pipeline failure
    if len(low_variance_features) > 2:
        print(
            f"\nWARNING: {len(low_variance_features)} features have near-zero variance. "
            "The model would be trained on effectively constant inputs — do NOT deploy it."
        )
        if not args.force:
            sys.exit(1)
        print("--force passed — proceeding despite low-variance features.")

    model = RegimeModel()
    model.train(X_train, y_train, **extra_kwargs)

    # ── Walk-forward cross-validation (evaluation only) ──────────────────────
    # 3-fold expanding walk-forward on the non-held-out rows only.  The final
    # model (trained on X_train above) is unaffected — this CV is for
    # confidence reporting only.  Using n_cv (= n_total - test_size) keeps the
    # held-out test set completely outside every CV fold.
    n_cv = n_total - args.test_size   # keep hold-out completely out of CV
    fold_cuts = [
        (0, int(0.4 * n_cv), int(0.4 * n_cv), int(0.6 * n_cv)),  # fold 1
        (0, int(0.6 * n_cv), int(0.6 * n_cv), int(0.8 * n_cv)),  # fold 2
        (0, int(0.8 * n_cv), int(0.8 * n_cv), n_cv),             # fold 3
    ]

    cv_briers: list[float] = []
    cv_accuracies: list[float] = []
    cv_kronos_agreements: list[float] = []

    print()
    print("── Walk-forward CV (3 folds) ─────────────────────────────────────────")
    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(fold_cuts, start=1):
        X_cv_train = X[tr_start:tr_end]
        y_cv_train = y_up[tr_start:tr_end]
        X_cv_test  = X[te_start:te_end]
        y_cv_test  = y_up[te_start:te_end]
        k_cv_test  = kronos_cal[te_start:te_end]

        try:
            cv_kwargs = maybe_scale_pos_weight(y_cv_train)
        except SystemExit:
            print(f"  Fold {fold_idx}: skipped — degenerate label distribution in train slice")
            continue
        cv_model = RegimeModel()
        cv_model.train(X_cv_train, y_cv_train, **cv_kwargs)

        proba_cv   = cv_model._clf.predict_proba(X_cv_test)[:, 1]
        pred_cv    = (proba_cv >= 0.5).astype(int)
        f_brier    = brier_score(y_cv_test, proba_cv)
        f_accuracy = float((pred_cv == y_cv_test).mean())
        k_dir_cv   = (k_cv_test >= 0.5).astype(int)
        f_kag      = float((pred_cv == k_dir_cv).mean())

        cv_briers.append(f_brier)
        cv_accuracies.append(f_accuracy)
        cv_kronos_agreements.append(f_kag)

        print(
            f"  Fold {fold_idx}  train=[{tr_start}:{tr_end}]  test=[{te_start}:{te_end}]  "
            f"Brier={f_brier:.4f}  Acc={f_accuracy:.4f}  KronosAgreement={f_kag:.4f}"
        )

    mean_brier = float(np.mean(cv_briers))
    std_brier  = float(np.std(cv_briers, ddof=1))
    mean_acc   = float(np.mean(cv_accuracies))
    std_acc    = float(np.std(cv_accuracies, ddof=1))
    mean_kag   = float(np.mean(cv_kronos_agreements))
    std_kag    = float(np.std(cv_kronos_agreements, ddof=1))

    print()
    print(f"  CV mean  Brier={mean_brier:.4f} ± {std_brier:.4f}   "
          f"Acc={mean_acc:.4f} ± {std_acc:.4f}   "
          f"KronosAgreement={mean_kag:.4f} ± {std_kag:.4f}")
    print("──────────────────────────────────────────────────────────────────────")

    if std_brier > 0.05:
        print(
            "\nWARNING: Brier std across folds is {:.4f} (> 0.05). Performance is highly "
            "variable across time windows — the model may be fitting regime-specific "
            "noise. Consider gathering more data or reviewing feature engineering before "
            "deploying.".format(std_brier)
        )
    if mean_kag < 0.55:
        print("WARNING: Kronos agreement < 55% (CV mean). The regime model is contradicting")
        print("         Kronos on nearly half of trades. Investigate before enabling Gate 2")
        print("         enforcement — flipping REGIME_GATE2_ENFORCING=True now would")
        print("         block roughly that fraction of trades.")
    if mean_brier > 0.25:
        print("WARNING: Brier > 0.25 (CV mean, worse than a coin flip). Do NOT deploy this model.")
    print()

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

    # ── Feature importances ───────────────────────────────────────────────────
    importances = model._clf.feature_importances_
    total_importance = float(importances.sum())
    ranked = sorted(zip(active_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances (descending):")
    for feat, imp in ranked:
        print(f"  {feat:<25s}  {imp:.4f}")
    top_feat, top_imp = ranked[0]
    if total_importance == 0:
        print("WARNING: all feature importances are zero — model may not have learned anything.")
    elif (top_imp / total_importance) > 0.60:
        print(
            f"\nWARNING: '{top_feat}' accounts for {top_imp / total_importance:.1%} of total "
            "importance. The model is essentially a single-feature classifier."
        )


if __name__ == "__main__":
    main()
