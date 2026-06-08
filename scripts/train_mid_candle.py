"""
Train the MidCandleModel from candle_features in trades.db.

Training source
---------------
candle_features rows where a mid-candle snapshot was captured at 40-60% progress.
Unlike the regime model (which predicts candle N+1 from candle N's close features),
this model predicts the SAME candle's direction from in-flight microstructure —
no 1-candle lag. At 40-60% the close hasn't happened yet, so features are not leaky.

Run once candle_features has ≥200 qualifying rows with cvd_since_open NOT NULL
(~2-3 days of live data at ~77 qualifying snaps/day).

Usage:
    python3 scripts/train_mid_candle.py [--db trades.db] [--out models/mid_candle.pkl]
                                        [--test-size 50] [--min-rows 200] [--dry-run]

Filtering rules
---------------
A candle_features row qualifies iff:
    btc_direction        IS NOT NULL   (candle resolved)
    cvd_since_open       IS NOT NULL   (mid-candle snapshot was captured)
    kalshi_mid_candle_mid IS NOT NULL  (Kalshi orderbook was live at snapshot)

No features_stale filter — mid-candle features (CVD accumulator, BRTI, Kalshi OB,
k5/k15 cache) are independent of regime:features Redis staleness.

Label semantics
---------------
btc_direction = 1 if 15-min candle close > open, else 0. Same candle as the
snapshot. Not "did we win" — that conflates signal quality with gate decisions.

k5/k15 notes
------------
k5_at_midcandle: most-recent cached k5 at snapshot time. k5 refreshes at every
5-min close (~33% and ~66% of the 15-min candle), so this is at most 5 min stale.

k15_at_midcandle: cached k15 from the prior 15-min close (regime anchor). It does
not update mid-candle — k15 only sees closed candles. This is intentional: k15
represents the prior-trend prior; k5_k15_delta captures the in-candle divergence.
The model learns to use k15 as a baseline, not a real-time signal.

Train/test split
----------------
Time-ordered. The last --test-size rows are held out. Random splits would leak
regime structure (crypto regimes persist across candles).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.mid_candle_model import MidCandleModel, _MID_CANDLE_FEATURES

_CANDLE_QUERY = """
SELECT {cols}, btc_direction, candle_ts
FROM candle_features
WHERE btc_direction IS NOT NULL
  AND cvd_since_open IS NOT NULL
  AND kalshi_mid_candle_mid IS NOT NULL
  AND tick_count_since_open > 200
ORDER BY candle_ts ASC
"""

_LAST_TRAINED_PATH = "models/mid_candle_last_trained.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db",        default="trades.db")
    p.add_argument("--out",       default=config.MID_CANDLE_MODEL_PATH,
                   help=f"Output path (default: {config.MID_CANDLE_MODEL_PATH})")
    p.add_argument("--test-size", type=int, default=50,
                   help="Most-recent rows held out for evaluation (default: 50)")
    p.add_argument("--min-rows",  type=int, default=200,
                   help="Minimum qualifying rows to train (default: 200 ≈ 2.5 days)")
    p.add_argument("--max-rows",  type=int, default=None,
                   help="If set, use only the most recent N qualifying rows.")
    p.add_argument("--dry-run",   action="store_true",
                   help="Report metrics but do NOT write the model file.")
    p.add_argument("--force",     action="store_true",
                   help="Skip low-variance feature gate and train anyway.")
    return p.parse_args()


def load_dataset(db_path: str, max_rows: int | None = None) -> list[tuple]:
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    cols = ", ".join(_MID_CANDLE_FEATURES)
    query = _CANDLE_QUERY.format(cols=cols)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError as exc:
        # Columns may not exist yet if the schema migration hasn't run.
        sys.exit(
            f"Schema error — have you restarted KronosV2 after the mid-candle "
            f"feature commit to apply migrations?\n  {exc}"
        )
    finally:
        conn.close()

    print(f"  Qualifying rows (btc_direction + cvd_since_open not null): {len(rows)}")

    # Strip trailing candle_ts (sort key only).
    rows = [r[:-1] for r in rows]

    if max_rows is not None and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def build_xy(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Same-candle pairing — no lag. features[i] → direction[i]."""
    arr = np.array(rows, dtype=object)
    n_features = len(_MID_CANDLE_FEATURES)
    X = arr[:, :n_features].astype(np.float64)
    y = arr[:, n_features].astype(int)
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
    print(f"Qualifying mid-candle rows in {args.db}: {n_total}")

    if n_total < args.min_rows:
        days_remaining = max(0, args.min_rows - n_total) / 77
        sys.exit(
            f"Need ≥{args.min_rows} qualifying rows to train; have {n_total}.\n"
            f"At ~77 qualifying snaps/day, that's ~{days_remaining:.1f} more day(s).\n"
            f"Tip: run with --dry-run to preview the dataset without training."
        )

    if n_total <= args.test_size + 30:
        sys.exit(
            f"Not enough rows for both train (>30) and test ({args.test_size}): "
            f"have {n_total}. Increase --min-rows or wait for more data."
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
    # tick_count_since_open is integer but stored as float64 — variance check still valid.
    low_variance: list[tuple[str, float]] = []
    for i, feat in enumerate(_MID_CANDLE_FEATURES):
        col = X_train[:, i]
        non_nan = col[~np.isnan(col)]
        if len(non_nan) == 0:
            print(f"WARNING: feature '{feat}' is entirely NaN — will be ignored by XGBoost.")
            low_variance.append((feat, 0.0))
            continue
        std = float(non_nan.std())
        nan_frac = (np.isnan(col)).mean()
        if nan_frac > 0.5:
            print(f"WARNING: feature '{feat}' is {nan_frac:.0%} NaN — insufficient data yet.")
        if std < 1e-6:
            print(f"WARNING: feature '{feat}' has near-zero std: {std:.2e}")
            low_variance.append((feat, std))

    if len(low_variance) > 2:
        print(f"\nWARNING: {len(low_variance)} features have near-zero variance.")
        if not args.force:
            sys.exit("Run with --force to train anyway, but do NOT deploy this model.")
        print("--force passed — proceeding despite low-variance features.")

    model = MidCandleModel()
    model.train(X_train, y_train, **extra_kwargs)

    # ── Walk-forward CV ───────────────────────────────────────────────────────
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
        X_cv_tr = X[tr_start:tr_end]
        y_cv_tr = y[tr_start:tr_end]
        X_cv_te = X[te_start:te_end]
        y_cv_te = y[te_start:te_end]
        try:
            cv_kw = maybe_scale_pos_weight(y_cv_tr)
        except SystemExit:
            print(f"  Fold {fold_idx}: skipped — degenerate label distribution")
            continue
        cv_model = MidCandleModel()
        cv_model.train(X_cv_tr, y_cv_tr, **cv_kw)
        proba_cv = cv_model._clf.predict_proba(X_cv_te)[:, 1]
        pred_cv  = (proba_cv >= 0.5).astype(int)
        f_brier  = brier_score(y_cv_te, proba_cv)
        f_acc    = float((pred_cv == y_cv_te).mean())
        cv_briers.append(f_brier)
        cv_accuracies.append(f_acc)
        print(f"  Fold {fold_idx}  train=[{tr_start}:{tr_end}]  test=[{te_start}:{te_end}]  "
              f"Brier={f_brier:.4f}  Acc={f_acc:.4f}")

    if cv_briers:
        mean_brier = float(np.mean(cv_briers))
        std_brier  = float(np.std(cv_briers, ddof=1)) if len(cv_briers) > 1 else 0.0
        mean_acc   = float(np.mean(cv_accuracies))
        print()
        print(f"  CV mean  Brier={mean_brier:.4f} ± {std_brier:.4f}   Acc={mean_acc:.4f}")
        if std_brier > 0.05:
            print(f"  WARNING: Brier std {std_brier:.4f} > 0.05 — high variance across folds.")
            print("  Consider waiting for more data before deploying.")
        if mean_brier > 0.25:
            print("  WARNING: Brier > 0.25 (CV mean, worse than coin flip). Do NOT deploy.")
    print("──────────────────────────────────────────────────────────────────────")

    # ── Final test-set evaluation ─────────────────────────────────────────────
    proba_test = model._clf.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)
    test_brier = brier_score(y_test, proba_test)
    test_acc   = float((pred_test == y_test).mean())
    print(f"\nFinal held-out test  Brier={test_brier:.4f}  Acc={test_acc:.4f}")

    # ── Feature importances ───────────────────────────────────────────────────
    importances = model._clf.feature_importances_
    total_imp = float(importances.sum())
    ranked = sorted(zip(_MID_CANDLE_FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances (descending):")
    for feat, imp in ranked:
        pct = imp / total_imp * 100 if total_imp > 0 else 0.0
        nan_frac = np.isnan(X_train[:, _MID_CANDLE_FEATURES.index(feat)]).mean()
        nan_note = f"  [{nan_frac:.0%} NaN]" if nan_frac > 0.1 else ""
        print(f"  {feat:<30s}  {imp:.4f}  ({pct:.1f}%){nan_note}")

    if total_imp > 0:
        top_feat, top_imp = ranked[0]
        if (top_imp / total_imp) > 0.60:
            print(f"\nWARNING: '{top_feat}' accounts for {top_imp/total_imp:.1%} of total "
                  "importance — essentially a single-feature classifier. Wait for more data.")

    # ── High-confidence profitability check ───────────────────────────────────
    # The model only generates value when its high-confidence calls are accurate.
    # Check that the top/bottom 20% of prob_up on the test set have better accuracy
    # than the middle 60% — if not, the model hasn't learned to be selective.
    print("\n── High-confidence profitability check ───────────────────────────────")
    if len(proba_test) >= 20:
        lo_thresh = np.percentile(proba_test, 20)
        hi_thresh = np.percentile(proba_test, 80)
        mask_hi = proba_test >= hi_thresh   # strong UP predictions
        mask_lo = proba_test <= lo_thresh   # strong DOWN predictions
        mask_mid = ~mask_hi & ~mask_lo

        def _acc(mask):
            if mask.sum() == 0:
                return float("nan"), 0
            preds = (proba_test[mask] >= 0.5).astype(int)
            return float((preds == y_test[mask]).mean()), int(mask.sum())

        hi_acc, hi_n   = _acc(mask_hi)
        lo_acc, lo_n   = _acc(mask_lo)
        mid_acc, mid_n = _acc(mask_mid)

        print(f"  Strong UP   (prob≥{hi_thresh:.2f}): acc={hi_acc:.3f}  n={hi_n}")
        print(f"  Strong DOWN (prob≤{lo_thresh:.2f}): acc={lo_acc:.3f}  n={lo_n}")
        print(f"  Middle 60%              : acc={mid_acc:.3f}  n={mid_n}")

        avg_tail_acc = np.nanmean([hi_acc, lo_acc])
        if avg_tail_acc > mid_acc + 0.05:
            print(f"  ✓ Tails ({avg_tail_acc:.3f}) outperform middle ({mid_acc:.3f}) "
                  "— model is selective. Conditional deployment viable.")
        else:
            print(f"  ✗ Tails ({avg_tail_acc:.3f}) do NOT outperform middle ({mid_acc:.3f}).")
            print("  High-confidence calls are not more accurate than random.")
            print("  Do NOT deploy for live entries yet — continue collecting data.")
    else:
        print("  Too few test rows for profitability check (need ≥20).")
    print("──────────────────────────────────────────────────────────────────────")

    if args.dry_run:
        print("\n--dry-run set — model NOT saved.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"\nSaved mid-candle model → {out_path}")

    # Write last-trained record for auto-retrain tracking.
    last_trained = {
        "trained_at_rows": n_total,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows_at_train": n_total,
        "test_brier": round(test_brier, 4),
        "test_acc": round(test_acc, 4),
    }
    Path(_LAST_TRAINED_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(_LAST_TRAINED_PATH).write_text(json.dumps(last_trained, indent=4))
    print(f"Wrote training record → {_LAST_TRAINED_PATH}")
    print("\nNext step: restart KronosV2 to pick up the model for live snapshot scoring.")
    print("Deploy Gate 15 only after verifying tails outperform the middle on 2+ retrains.")


if __name__ == "__main__":
    main()
