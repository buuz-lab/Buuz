"""
Phase 3c: Train the Calibrator on regime_prob + signal_edge from trades.db.

Usage:
    python3 scripts/train_calibrator.py [--db trades.db] [--out models/calibrator.pkl]
                                        [--window 300] [--min-rows 200] [--dry-run]

Data sources
------------
Training rows are drawn from both tables to eliminate selection bias:
  - trades: placed trades with resolved outcomes
  - gate_rejections: blocked signals with resolved counterfactual outcomes
    (shadow=1 Gate 7 rows excluded via WHERE shadow=0)

Only rows where regime_prob IS NOT NULL are used — rows from before regime v2 deployed
have no regime_prob and are excluded.

Label semantics
---------------
The calibrator maps regime_prob (P(market UP) from XGBoost) to a calibrated probability.
It must be trained with y_yes = P(YES happened):

    direction=1, outcome=1 (YES win = market UP)    → y_yes=1
    direction=1, outcome=0 (YES loss = market DOWN)  → y_yes=0
    direction=0, outcome=1 (NO win = market DOWN)   → y_yes=0
    direction=0, outcome=0 (NO loss = market UP)    → y_yes=1

signal_edge = abs(regime_prob - kalshi_mid_cents/100) at trade time — stored signed in DB,
taken absolute during training.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.calibrator import Calibrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--out", default=config.CALIBRATOR_MODEL_PATH)
    p.add_argument("--window", type=int, default=300,
                   help="Number of most-recent training-ready rows to use (default: 300)")
    p.add_argument("--min-rows", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.db).exists():
        sys.exit(f"Database not found: {args.db}")

    # ticker + timestamp included for per-market deduplication (dropped before training).
    # gate_rejections fires 2-4x per market within one candle — same outcome, different
    # regime_prob as model conviction builds mid-candle. Keeping all entries would give
    # correlated label samples that bias the calibrator against high-confidence calls.
    # Fix: keep EARLIEST entry per ticker (first signal = cleanest, before intra-candle
    # drift inflates model conviction on the developing candle).
    # trades does not have a signal_edge column (gate_rejections does).
    # Use NULL AS signal_edge for trades rows — the calibrator handles NULL edges gracefully.
    _UNION_QUERY = """
        SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome,
               kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
               shap_coherence, ticker, timestamp FROM (
            SELECT regime_prob, NULL AS signal_edge, deepseek_regime, direction, outcome,
                   CAST(strftime('%s', timestamp) AS REAL) AS timestamp,
                   kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
                   NULL AS shap_coherence, ticker
            FROM trades
            WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL
            UNION ALL
            SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome, timestamp,
                   kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
                   shap_coherence, ticker
            FROM gate_rejections
            WHERE outcome IS NOT NULL AND shadow = 0
              AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
        )
        ORDER BY timestamp DESC LIMIT ?
    """
    # Count unique markets (tickers) — the real denominator after deduplication.
    _COUNT_QUERY = """
        SELECT COUNT(DISTINCT ticker) FROM (
            SELECT ticker FROM trades
            WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL
            UNION ALL
            SELECT ticker FROM gate_rejections
            WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
              AND shadow = 0
        )
    """

    conn = sqlite3.connect(args.db)
    try:
        unique_markets = conn.execute(_COUNT_QUERY).fetchone()[0]
        rows = conn.execute(_UNION_QUERY, (args.window,)).fetchall()
    finally:
        conn.close()

    # Deduplicate: keep earliest entry per ticker.
    # Sort ascending by timestamp (index 10), keep first seen per ticker (index 9).
    seen: set[str] = set()
    deduped: list[tuple] = []
    for r in sorted(rows, key=lambda x: x[10]):
        if r[9] not in seen:
            seen.add(r[9])
            deduped.append(r)
    n_raw = len(rows)
    rows = deduped
    n = len(rows)

    print(f"Phase 3c training rows in {args.db}: {unique_markets} unique markets  "
          f"({n_raw} raw entries, {n_raw - n} same-candle duplicates removed, using {n})")
    if unique_markets < args.min_rows:
        sys.exit(
            f"Need ≥{args.min_rows} unique markets; have {unique_markets}. "
            f"Regime v2 must be deployed and generating predictions before Phase 3c can train."
        )

    regime_probs  = np.array([r[0] for r in rows], dtype=float)
    abs_edges     = np.abs(np.array([r[1] for r in rows], dtype=float))
    regimes       = np.array([r[2] for r in rows], dtype=object)
    directions    = np.array([r[3] for r in rows], dtype=float)
    outcomes      = np.array([r[4] for r in rows], dtype=float)
    y_yes = np.where(directions == 1, outcomes, 1.0 - outcomes)

    # New calibrator context features (None→NaN→replaced with 0 via np.nan_to_num)
    kronos_k15    = np.array([r[5] if r[5] is not None else np.nan for r in rows], dtype=float)
    volatilities  = np.array([r[6] if r[6] is not None else np.nan for r in rows], dtype=float)
    spreads       = np.array([r[7] if r[7] is not None else np.nan for r in rows], dtype=float)
    # disagreement: abs(regime_prob - k15). When k15 is missing, use 0 (neutral).
    disagreements = np.abs(regime_probs - np.where(np.isnan(kronos_k15), regime_probs, kronos_k15))
    volatilities  = np.nan_to_num(volatilities,  nan=0.0)
    spreads       = np.nan_to_num(spreads,        nan=0.0)
    shap_coherences_raw = np.array([r[8] if r[8] is not None else np.nan for r in rows], dtype=float)
    shap_coherences = None if np.all(np.isnan(shap_coherences_raw)) else np.nan_to_num(shap_coherences_raw, nan=0.5)

    # Load existing calibrator for pre-retrain Brier comparison
    pre_brier: float | None = None
    if Path(args.out).exists():
        try:
            existing = Calibrator.load(args.out)
            pre_brier = existing.brier_score(regime_probs, y_yes)
            print(f"Existing calibrator: n_samples={existing.n_samples} passthrough={existing._passthrough} edge_aware={existing._edge_aware}")
            print(f"Pre-retrain Brier:  {pre_brier:.4f}")
        except Exception as exc:
            print(f"Could not load existing calibrator: {exc}")
    else:
        print(f"No existing calibrator at {args.out} — fitting fresh")

    cal = Calibrator()
    cal.fit(
        regime_probs, y_yes,
        regimes=regimes,
        edges=abs_edges,
        disagreements=disagreements,
        volatilities=volatilities,
        spreads=spreads,
        shap_coherences=shap_coherences,
    )
    post_brier = cal.brier_score(regime_probs, y_yes)

    print(f"Post-retrain Brier: {post_brier:.4f}")
    print(f"Passthrough:  {cal._passthrough}")
    print(f"Edge-aware:   {cal._edge_aware}")
    print(f"n_samples:    {cal.n_samples}")

    if pre_brier is not None and post_brier > pre_brier:
        print(f"WARNING: new Brier {post_brier:.4f} > old Brier {pre_brier:.4f} — calibration degraded")

    # Compression map: regime_prob → calibrated_prob at different edge levels.
    # Low edge = regime agrees with market; high edge = big gap between regime and market.
    checkpoints = [0.60, 0.70, 0.80, 0.90, 1.00]
    for edge_level, label in [(0.05, "tight edge (5¢)"), (0.15, "normal edge (15¢)"), (0.30, "wide edge (30¢)")]:
        print(f"\nCompression map (regime_prob → cal_prob, trending_up, {label}):")
        for raw in checkpoints:
            cal_val = cal.transform(raw, regime="trending_up", edge=edge_level)
            bar = "█" * int((cal_val - 0.50) * 200)
            print(f"  {raw:.2f} → {cal_val:.4f}  {bar}")

    if args.dry_run:
        print("\n--dry-run set — calibrator NOT saved.")
        return

    os.makedirs("models", exist_ok=True)
    cal.save(args.out)
    print(f"\nSaved calibrator to: {args.out}")

    import datetime, json
    meta = {
        "trained_at_rows": n,
        "trained_at_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_rows_at_train": total_available,
    }
    meta_path = Path(args.out).parent / "calibrator_last_trained.json"
    meta_path.write_text(json.dumps(meta, indent=4))
    print(f"Metadata written to: {meta_path}")


if __name__ == "__main__":
    main()
