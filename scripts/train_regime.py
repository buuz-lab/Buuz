"""
Train the RegimeModel from candle_features in trades.db.

Training source
---------------
candle_features — logged at every 15-min BTC candle close, regardless of whether
a trade was placed.  This eliminates the selection bias of training on placed
trades (which cleared all Kalshi-based gates and skew the sample).

Run once candle_features has ≥672 rows (7 days × 96 candles/day).

Usage:
    python3 scripts/train_regime.py [--db trades.db] [--out models/regime.pkl]
                                    [--test-size 100] [--min-rows 672] [--dry-run]

Filtering rules
---------------
A candle_features row qualifies iff:
    features_stale  = 0            (Redis regime:features was fresh at candle close)
    btc_direction   IS NOT NULL    (candle resolved — close > open recorded)
    funding_rate    IS NOT NULL    (post-instrumentation rows only)
    cvd_velocity    IS NOT NULL    (21-feature era)
    large_print_direction IS NOT NULL

Label semantics
---------------
btc_direction = 1 if 15-min candle close > open, else 0.

This is clean ground truth — not "did Kronos win" (which conflates signal quality
with gate decisions) and not "direction == outcome" (which is circular because
the gates that produced those outcomes depend on Kalshi).

Train/test split
----------------
Time-ordered. The last --test-size rows are held out; everything older trains.
Random splits would leak regime structure (crypto regimes persist across candles).

Class balance
-------------
scale_pos_weight applied only when the up/down ratio drifts outside [35%, 65%].
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.regime_model import RegimeModel, _FEATURE_ORDER

# Must match _FEATURE_ORDER in regime_model.py and keys from fusion._regime_features().
_FEATURE_COLS = list(_FEATURE_ORDER)

_CANDLE_QUERY = """
SELECT {cols}, btc_direction, candle_ts
FROM candle_features
WHERE features_stale = 0
  AND btc_direction IS NOT NULL
  AND funding_rate IS NOT NULL
  AND cvd_velocity IS NOT NULL
  AND brti_momentum_5min IS NOT NULL
  AND funding_window_proximity IS NOT NULL
  AND large_print_direction IS NOT NULL
  AND atm_iv IS NOT NULL
ORDER BY candle_ts ASC
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="trades.db")
    p.add_argument("--out", default=config.REGIME_MODEL_PATH,
                   help=f"Output path (default: {config.REGIME_MODEL_PATH})")
    p.add_argument("--test-size", type=int, default=100,
                   help="Most-recent candles held out for evaluation (default: 100)")
    p.add_argument("--min-rows", type=int, default=672,
                   help="Minimum qualifying candles required to train (default: 672 = 7 days)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="If set, use only the most recent N qualifying rows.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report metrics but do NOT write the model file.")
    p.add_argument("--force", action="store_true",
                   help="Skip the low-variance feature gate and train anyway.")
    p.add_argument("--warm-start", action="store_true",
                   help="Continue training from existing regime.pkl (+25 trees). "
                        "Falls back to cold start if no model found.")
    return p.parse_args()


def load_dataset(db_path: str, max_rows: int | None = None) -> list[tuple]:
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    cols = ", ".join(_FEATURE_COLS)
    query = _CANDLE_QUERY.format(cols=cols)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    print(f"  candle_features qualifying rows: {len(rows)}")

    # Strip trailing candle_ts (sort key, not a feature).
    rows = [r[:-1] for r in rows]

    if max_rows is not None and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def build_xy(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) with 1-candle lag: features[N] → direction[N+1].

    This matches runtime behaviour where the model sees features at candle N's
    close and predicts candle N+1's direction.  Using same-candle labels inflates
    accuracy because features like brti_momentum_15min encode the candle's own
    return — the exact quantity being predicted.
    """
    arr = np.array(rows, dtype=object)
    n_features = len(_FEATURE_COLS)
    X = arr[:-1, :n_features].astype(np.float64)   # features from candle N
    y = arr[1:,  n_features].astype(int)            # direction of candle N+1
    return X, y


def maybe_scale_pos_weight(y: np.ndarray) -> dict:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        sys.exit(f"Degenerate label distribution: pos={pos} neg={neg}. Refusing to train.")
    pos_frac = pos / (pos + neg)
    if pos_frac < 0.35 or pos_frac > 0.65:
        return {"scale_pos_weight": neg / pos}
    return {}


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true) ** 2))


def main() -> None:
    args = parse_args()

    rows = load_dataset(args.db, max_rows=args.max_rows)
    n_total = len(rows)
    print(f"Qualifying candle rows in {args.db}: {n_total}")

    if args.max_rows is not None:
        print(f"--max-rows {args.max_rows}: using most recent {n_total} rows")

    if n_total < args.min_rows:
        sys.exit(
            f"Need ≥{args.min_rows} qualifying candle rows to train; have {n_total}.\n"
            f"At ~96 candles/day, that's ~{max(0, args.min_rows - n_total) // 96 + 1} more day(s)."
        )
    # After 1-candle lag we have n_total-1 (X, y) pairs.
    n_pairs = n_total - 1
    if n_pairs <= args.test_size + 50:
        sys.exit(
            f"Not enough lagged pairs for both train (>50) and test ({args.test_size}): "
            f"have {n_pairs}. Increase --min-rows or wait for more data."
        )

    X, y = build_xy(rows)

    X_train, X_test = X[: -args.test_size], X[-args.test_size :]
    y_train, y_test = y[: -args.test_size], y[-args.test_size :]

    pos_tr = int(y_train.sum())
    pos_te = int(y_test.sum())
    print(f"Train: {len(y_train)} rows  (up={pos_tr}, down={len(y_train)-pos_tr})")
    print(f"Test : {len(y_test)} rows  (up={pos_te}, down={len(y_test)-pos_te})")

    extra_kwargs = maybe_scale_pos_weight(y_train)
    if extra_kwargs:
        print(f"Applying scale_pos_weight={extra_kwargs['scale_pos_weight']:.3f} "
              f"(train class balance outside [35%, 65%])")
    else:
        print("Train class balance within [35%, 65%] — no scale_pos_weight applied.")

    # ── Feature variance gate ─────────────────────────────────────────────────
    low_variance: list[tuple[str, float]] = []
    for i, feat in enumerate(_FEATURE_COLS):
        std = float(X_train[:, i].std())
        if std < 1e-6:
            print(f"WARNING: feature '{feat}' has near-zero std: {std:.2e}")
            low_variance.append((feat, std))
    if len(low_variance) > 2:
        print(f"\nWARNING: {len(low_variance)} features have near-zero variance. "
              "Do NOT deploy this model.")
        if not args.force:
            sys.exit(1)
        print("--force passed — proceeding despite low-variance features.")

    warm_start_model = None
    if args.warm_start:
        try:
            warm_start_model = RegimeModel.load(args.out)
            n_existing = warm_start_model._clf.get_booster().num_boosted_rounds()
            print(f"Warm-start: loaded {args.out} ({n_existing} trees → +25)")
        except FileNotFoundError:
            print(f"Warm-start: no model at {args.out} — cold start (100 trees)")

    model = RegimeModel()
    model.train(X_train, y_train, warm_start_from=warm_start_model, **extra_kwargs)

    # ── Walk-forward CV (evaluation only) ────────────────────────────────────
    n_cv = n_total - args.test_size
    fold_cuts = [
        (0, int(0.4 * n_cv), int(0.4 * n_cv), int(0.6 * n_cv)),
        (0, int(0.6 * n_cv), int(0.6 * n_cv), int(0.8 * n_cv)),
        (0, int(0.8 * n_cv), int(0.8 * n_cv), n_cv),
    ]
    cv_briers: list[float] = []
    cv_accuracies: list[float] = []

    print()
    print("── Walk-forward CV (3 folds) ─────────────────────────────────────────")
    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(fold_cuts, start=1):
        X_cv_train = X[tr_start:tr_end]
        y_cv_train = y[tr_start:tr_end]
        X_cv_test  = X[te_start:te_end]
        y_cv_test  = y[te_start:te_end]
        try:
            cv_kwargs = maybe_scale_pos_weight(y_cv_train)
        except SystemExit:
            print(f"  Fold {fold_idx}: skipped — degenerate label distribution")
            continue
        cv_model = RegimeModel()
        cv_model.train(X_cv_train, y_cv_train, **cv_kwargs)
        proba_cv   = cv_model._clf.predict_proba(X_cv_test)[:, 1]
        pred_cv    = (proba_cv >= 0.5).astype(int)
        f_brier    = brier_score(y_cv_test, proba_cv)
        f_accuracy = float((pred_cv == y_cv_test).mean())
        cv_briers.append(f_brier)
        cv_accuracies.append(f_accuracy)
        print(f"  Fold {fold_idx}  train=[{tr_start}:{tr_end}]  test=[{te_start}:{te_end}]  "
              f"Brier={f_brier:.4f}  Acc={f_accuracy:.4f}")

    mean_brier = float(np.mean(cv_briers))
    std_brier  = float(np.std(cv_briers, ddof=1))
    mean_acc   = float(np.mean(cv_accuracies))
    std_acc    = float(np.std(cv_accuracies, ddof=1))
    print()
    print(f"  CV mean  Brier={mean_brier:.4f} ± {std_brier:.4f}   "
          f"Acc={mean_acc:.4f} ± {std_acc:.4f}")
    print("──────────────────────────────────────────────────────────────────────")

    if std_brier > 0.05:
        print(f"\nWARNING: Brier std {std_brier:.4f} > 0.05 — high variance across folds. "
              "Consider waiting for more data before deploying.")
    if mean_brier > 0.25:
        print("WARNING: Brier > 0.25 (CV mean, worse than coin flip). Do NOT deploy.")

    # ── Final test-set evaluation ─────────────────────────────────────────────
    proba_test = model._clf.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)
    test_brier = brier_score(y_test, proba_test)
    test_acc   = float((pred_test == y_test).mean())
    print(f"\nFinal held-out test  Brier={test_brier:.4f}  Acc={test_acc:.4f}")

    # ── Feature importances ───────────────────────────────────────────────────
    importances = model._clf.feature_importances_
    total_imp = float(importances.sum())
    ranked = sorted(zip(_FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances (descending):")
    for feat, imp in ranked:
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        print(f"  {feat:<25s}  {imp:.4f}  ({pct:.1f}%)")
    if total_imp > 0:
        top_feat, top_imp = ranked[0]
        if (top_imp / total_imp) > 0.60:
            print(f"\nWARNING: '{top_feat}' accounts for {top_imp/total_imp:.1%} of total "
                  "importance — essentially a single-feature classifier.")

    # ── Calibration sanity check ──────────────────────────────────────────────
    # Does the model's confidence correlate with actual accuracy on the training
    # set? Replaces the rigid k15=0.85 synthetic probe with a real data check.
    print("\n── Calibration sanity (training predictions) ─────────────────────────")
    try:
        train_proba = model._clf.predict_proba(X_train)[:, 1]
        tiers = [
            ("Low  (|p-0.5|<0.10)", lambda p: abs(p - 0.5) < 0.10),
            ("Med  (0.10–0.20)",     lambda p: 0.10 <= abs(p - 0.5) < 0.20),
            ("High (|p-0.5|>0.20)", lambda p: abs(p - 0.5) >= 0.20),
        ]
        tier_accs: dict[str, float] = {}
        for tier_name, tier_fn in tiers:
            mask = np.array([tier_fn(p) for p in train_proba])
            n = int(mask.sum())
            if n == 0:
                print(f"  {tier_name:<22s}  n=0   (no predictions in this range)")
                continue
            tier_y = y_train[mask]
            tier_p = train_proba[mask]
            brier  = float(np.mean((tier_p - tier_y) ** 2))
            acc    = float(((tier_p >= 0.5).astype(int) == tier_y).mean())
            tier_accs[tier_name] = acc
            if n >= 10:
                win_rate = float(tier_y.mean()) * 100
                print(f"  {tier_name:<22s}  n={n:<4d}  win_rate={win_rate:.0f}%  Brier={brier:.3f}  acc={acc:.0%}")
            else:
                print(f"  {tier_name:<22s}  n={n:<4d}  (accumulating — need 10+ for stats)")

        low_key  = "Low  (|p-0.5|<0.10)"
        high_key = "High (|p-0.5|>0.20)"
        if low_key in tier_accs and high_key in tier_accs:
            if tier_accs[high_key] > tier_accs[low_key]:
                print(f"  ✓ Calibration gradient present — high acc={tier_accs[high_key]:.0%} > low acc={tier_accs[low_key]:.0%}")
            else:
                print(f"  ⚠ No calibration gradient — high acc={tier_accs[high_key]:.0%} ≤ low acc={tier_accs[low_key]:.0%}")

        k15_imp = importances[_FEATURE_COLS.index("kronos_raw_15min")] / total_imp * 100 if total_imp > 0 else 0
        k5_imp  = importances[_FEATURE_COLS.index("kronos_raw_5min")]  / total_imp * 100 if total_imp > 0 else 0
        print(f"  Kronos importance: k15={k15_imp:.1f}%  k5={k5_imp:.1f}%  combined={k15_imp+k5_imp:.1f}%")

    except Exception as exc:
        print(f"  Could not run calibration sanity check: {exc}")
    print("──────────────────────────────────────────────────────────────────────")

    if args.dry_run:
        print("\n--dry-run set — model NOT saved.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    # Save SHAP baseline snapshot for monitor display and diagnostics.
    try:
        import json as _json
        from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
        _snapshot = compute_baseline_snapshot(model._clf, X_train, _FEATURE_COLS)
        _shap_path = out_path.parent / "regime_shap_baseline.json"
        _shap_path.write_text(_json.dumps(_snapshot, indent=2))
        print(f"  SHAP baseline saved → {_shap_path} (n={_snapshot['n_rows']} rows)")
    except Exception as _exc:
        print(f"  SHAP baseline save failed (non-fatal): {_exc}")
    print(f"\nSaved regime model → {out_path}")
    print("Restart KronosV2 to pick it up.")
    print("Gate 2 runs in SHADOW mode (config.REGIME_GATE2_ENFORCING=False) by default.")
    print("Observe disagreement logs for ~50 trades before flipping to True.")


if __name__ == "__main__":
    main()
