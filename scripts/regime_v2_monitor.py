"""
Regime v2 monitor — catches structural failures that auto-retrain cannot.

Checks (each rated PASS / WARN / FAIL):
  1. API health       — null rates + zero-variance for key features in last 24h
  2. Distribution drift — recent 24h feature means vs 14-day baseline (>2σ = WARN)
  3. Kalshi edge trend  — is our Brier advantage over Kalshi open holding?
  4. Training health    — stale rate, rows since last train, pause flag status

Usage:
    python3 scripts/regime_v2_monitor.py [--db trades.db] [--hours 24]

Crontab (every 12h):
    0 */12 * * * cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/regime_v2_monitor.py >> logs/regime_v2_monitor.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Key features to monitor for structural failures ──────────────────────────
# These are the most likely to degrade silently if an upstream API breaks.
_WATCH_FEATURES = [
    "cvd_normalized",      # zeroes out when OKX/Kraken trade APIs fail
    "funding_rate",        # zeroes out when all funding sources fail
    "oi_delta_pct",        # zeroes out with funding
    "atm_iv",              # goes NULL when Deribit fails
    "kronos_raw_15min",    # goes NULL if Kronos background loop stalls
    "volume_ratio_1h",     # goes 1.0 constant when OKX candle API fails
    "large_print_direction", # zeroes out with CVD
]

_MARKER_PATH = "models/regime_last_trained.json"
_PAUSE_FLAG   = Path("models/regime_paused.flag")
_MODEL_PATH   = "models/regime.pkl"
_SHAP_BASELINE_PATH = "models/regime_shap_baseline.json"

_STATUS = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}


def _conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ts_cutoff(hours: int) -> str:
    return (_now_utc() - timedelta(hours=hours)).isoformat()


# ── Section 1: API health ────────────────────────────────────────────────────

def section_api_health(conn: sqlite3.Connection, hours: int) -> list[str]:
    cutoff = _ts_cutoff(hours)
    issues: list[str] = []

    total = conn.execute(
        "SELECT COUNT(*) FROM candle_features WHERE candle_ts >= ?", (cutoff,)
    ).fetchone()[0]

    if total == 0:
        return [f"FAIL  No candle_features rows in last {hours}h — candle logger may be down"]

    print(f"\n── API Health (last {hours}h, n={total} candles) {'─'*40}")
    overall_ok = True

    for feat in _WATCH_FEATURES:
        null_count = conn.execute(
            f"SELECT COUNT(*) FROM candle_features WHERE candle_ts >= ? AND {feat} IS NULL",
            (cutoff,)
        ).fetchone()[0]
        null_pct = null_count / total * 100

        # Zero-variance: feature is non-null but constant
        rows = conn.execute(
            f"SELECT {feat} FROM candle_features WHERE candle_ts >= ? AND {feat} IS NOT NULL",
            (cutoff,)
        ).fetchall()
        vals = [r[0] for r in rows]
        zero_var = len(vals) > 5 and len(set(round(v, 6) for v in vals)) == 1

        if null_pct > 30 or zero_var:
            status = "FAIL"
            overall_ok = False
        elif null_pct > 10:
            status = "WARN"
        else:
            status = "PASS"

        suffix = f"  ← {'NULL' if null_pct > 10 else ''}{'CONSTANT' if zero_var else ''}" if status != "PASS" else ""
        print(f"  {_STATUS[status]} {feat:<28s}  null={null_pct:4.1f}%  zero_var={zero_var}{suffix}")

        if status in ("WARN", "FAIL"):
            issues.append(f"{status}  {feat}: null={null_pct:.1f}%, zero_var={zero_var}")

    if overall_ok:
        print(f"  {_STATUS['PASS']} All API sources healthy")

    return issues


# ── Section 2: Distribution drift ───────────────────────────────────────────

def section_distribution_drift(conn: sqlite3.Connection, hours: int) -> list[str]:
    """Compare recent 24h feature means vs 14-day baseline. Flag >2σ shifts."""
    cutoff_recent = _ts_cutoff(hours)
    cutoff_baseline = _ts_cutoff(14 * 24)
    issues: list[str] = []

    print(f"\n── Distribution Drift (last {hours}h vs 14-day baseline) {'─'*30}")

    drifted = False
    for feat in _WATCH_FEATURES:
        baseline = conn.execute(
            f"SELECT AVG({feat}), SUM(({feat} - (SELECT AVG({feat}) FROM candle_features "
            f"WHERE candle_ts >= ? AND {feat} IS NOT NULL)) * "
            f"({feat} - (SELECT AVG({feat}) FROM candle_features "
            f"WHERE candle_ts >= ? AND {feat} IS NOT NULL))) / COUNT(*) "
            f"FROM candle_features WHERE candle_ts >= ? AND {feat} IS NOT NULL",
            (cutoff_baseline, cutoff_baseline, cutoff_baseline)
        ).fetchone()

        # Simpler: just get mean + std separately
        stats = conn.execute(
            f"SELECT AVG({feat}), COUNT(*) FROM candle_features "
            f"WHERE candle_ts >= ? AND {feat} IS NOT NULL",
            (cutoff_baseline,)
        ).fetchone()
        hist_mean, hist_n = stats
        if hist_mean is None or hist_n < 20:
            continue

        # Historical std
        var_row = conn.execute(
            f"SELECT AVG(({feat} - ?) * ({feat} - ?)) FROM candle_features "
            f"WHERE candle_ts >= ? AND {feat} IS NOT NULL",
            (hist_mean, hist_mean, cutoff_baseline)
        ).fetchone()
        hist_std = var_row[0] ** 0.5 if var_row[0] and var_row[0] > 0 else None

        recent = conn.execute(
            f"SELECT AVG({feat}), COUNT(*) FROM candle_features "
            f"WHERE candle_ts >= ? AND {feat} IS NOT NULL",
            (cutoff_recent,)
        ).fetchone()
        recent_mean, recent_n = recent

        if recent_mean is None or recent_n < 3 or hist_std is None or hist_std < 1e-9:
            continue

        sigma_shift = abs(recent_mean - hist_mean) / hist_std

        if sigma_shift > 3.0:
            status = "FAIL"
        elif sigma_shift > 2.0:
            status = "WARN"
        else:
            status = "PASS"

        if status != "PASS":
            drifted = True
            print(f"  {_STATUS[status]} {feat:<28s}  hist={hist_mean:+.4f}  recent={recent_mean:+.4f}  shift={sigma_shift:.1f}σ")
            issues.append(f"{status}  {feat}: {sigma_shift:.1f}σ drift (hist={hist_mean:.4f}, recent={recent_mean:.4f})")

    if not drifted:
        print(f"  {_STATUS['PASS']} No significant distribution drift detected")

    return issues


# ── Section 3: Kalshi edge trend ─────────────────────────────────────────────

def section_kalshi_edge(conn: sqlite3.Connection) -> list[str]:
    issues: list[str] = []
    print(f"\n── Kalshi Edge Trend {'─'*50}")

    def _brier_acc(period_cutoff: str | None = None) -> tuple:
        where = "WHERE features_stale=0 AND atm_iv IS NOT NULL AND kalshi_open_mid IS NOT NULL AND kronos_raw_15min IS NOT NULL"
        if period_cutoff:
            where += f" AND candle_ts >= '{period_cutoff}'"
        row = conn.execute(f"""
            SELECT COUNT(*),
              AVG((kalshi_open_mid - btc_direction)*(kalshi_open_mid - btc_direction)),
              AVG((kronos_raw_15min - btc_direction)*(kronos_raw_15min - btc_direction)),
              AVG(CASE WHEN kalshi_open_mid > 0.5 AND btc_direction=1 OR kalshi_open_mid < 0.5 AND btc_direction=0 THEN 1.0 ELSE 0.0 END),
              AVG(CASE WHEN kronos_raw_15min > 0.5 AND btc_direction=1 OR kronos_raw_15min < 0.5 AND btc_direction=0 THEN 1.0 ELSE 0.0 END)
            FROM candle_features {where}
        """).fetchone()
        return row

    all_time = _brier_acc()
    recent_7d = _brier_acc(_ts_cutoff(7 * 24))

    def _print_row(label: str, row: tuple) -> float | None:
        n, kb, sb, ka, sa = row
        if n < 10 or kb is None:
            print(f"  {label:<12s}  n={n:<4d}  (insufficient data)")
            return None
        adv = (kb - sb) / kb * 100 if kb > 0 else 0
        print(f"  {label:<12s}  n={n:<4d}  kalshi={kb:.4f} ({ka*100:.1f}%)  k15={sb:.4f} ({sa*100:.1f}%)  advantage={adv:+.1f}%")
        return adv

    all_adv = _print_row("all-time", all_time)
    rec_adv  = _print_row("last 7d", recent_7d)

    # Live regime v2 Brier — only available after model deploys (regime_prob non-NULL)
    regime_rows = conn.execute(
        "SELECT COUNT(*) FROM candle_features "
        "WHERE features_stale=0 AND atm_iv IS NOT NULL AND regime_prob IS NOT NULL"
    ).fetchone()[0]
    if regime_rows >= 20:
        rv2 = conn.execute("""
            SELECT COUNT(*),
              AVG((regime_prob - btc_direction)*(regime_prob - btc_direction)),
              AVG((kalshi_open_mid - btc_direction)*(kalshi_open_mid - btc_direction)),
              AVG(CASE WHEN regime_prob > 0.5 AND btc_direction=1 OR regime_prob < 0.5 AND btc_direction=0 THEN 1.0 ELSE 0.0 END)
            FROM candle_features
            WHERE features_stale=0 AND atm_iv IS NOT NULL
              AND regime_prob IS NOT NULL AND kalshi_open_mid IS NOT NULL
        """).fetchone()
        n, rb, kb, ra = rv2
        if rb is not None and n >= 20:
            adv = (kb - rb) / kb * 100 if kb and kb > 0 else 0
            print(f"  regime_v2     n={n:<4d}  regime={rb:.4f} ({ra*100:.1f}%)  kalshi={kb:.4f}  advantage={adv:+.1f}%")
            if adv < 0:
                issues.append(f"WARN  Regime v2 Brier WORSE than Kalshi open ({rb:.4f} vs {kb:.4f})")
    elif regime_rows > 0:
        print(f"  regime_v2     n={regime_rows:<4d}  (accumulating — need 20 for comparison)")

    if all_adv is not None and rec_adv is not None:
        delta = rec_adv - all_adv
        if delta < -5:
            status = "WARN"
            msg = f"Edge narrowing: recent advantage {rec_adv:.1f}% vs all-time {all_adv:.1f}% (Δ={delta:+.1f}%)"
            print(f"  {_STATUS[status]} {msg}")
            issues.append(f"WARN  {msg}")
        elif rec_adv < 0:
            status = "FAIL"
            msg = f"Edge LOST: recent k15 Brier WORSE than Kalshi ({rec_adv:.1f}%)"
            print(f"  {_STATUS[status]} {msg}")
            issues.append(f"FAIL  {msg}")
        else:
            print(f"  {_STATUS['PASS']} Edge holding  (Δ={delta:+.1f}% vs all-time baseline)")

    return issues


# ── Section 4: Training health ───────────────────────────────────────────────

def section_training_health(conn: sqlite3.Connection) -> list[str]:
    issues: list[str] = []
    print(f"\n── Training Health {'─'*52}")

    # Row count and stale rate last 24h
    cutoff_24h = _ts_cutoff(24)
    recent = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN features_stale=0 AND atm_iv IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM candle_features WHERE candle_ts >= ?", (cutoff_24h,)
    ).fetchone()
    recent_total, recent_qual = recent
    stale_rate = (1 - recent_qual / recent_total) * 100 if recent_total > 0 else 100

    total_qual = conn.execute(
        "SELECT SUM(CASE WHEN features_stale=0 AND atm_iv IS NOT NULL THEN 1 ELSE 0 END) FROM candle_features"
    ).fetchone()[0] or 0

    print(f"  Qualifying rows (all-time)  : {total_qual}")
    print(f"  Last 24h: {recent_total} logged, {recent_qual} qualifying ({stale_rate:.0f}% stale)")

    if stale_rate > 30:
        issues.append(f"WARN  Stale rate last 24h: {stale_rate:.0f}% (>30% threshold)")
        print(f"  {_STATUS['WARN']} High stale rate: {stale_rate:.0f}%")
    else:
        print(f"  {_STATUS['PASS']} Stale rate OK: {stale_rate:.0f}%")

    # Marker
    marker_path = Path(_MARKER_PATH)
    if marker_path.exists():
        try:
            marker = json.loads(marker_path.read_text())
            last_rows = marker.get("trained_at_rows", 0)
            last_ts   = marker.get("trained_at_timestamp", "unknown")
            last_brier = marker.get("holdout_brier")
            dir_mean  = marker.get("direction_mean_at_train")
            rows_since = total_qual - last_rows
            next_trigger = max(0, 200 - rows_since)
            print(f"  Last train: {last_rows} rows  ({last_ts[:19]})")
            if last_brier:
                print(f"  Deploy holdout Brier: {last_brier:.4f}")
            if dir_mean is not None:
                print(f"  Direction mean at train: {dir_mean:.3f}  (current bear=<0.45, bull=>0.55)")
            print(f"  Rows since last train: {rows_since}  (next row trigger in {next_trigger} more rows)")
        except Exception:
            print("  Marker file unreadable")
    else:
        print(f"  {_STATUS['WARN']} No marker — model not yet trained")
        issues.append("WARN  No regime model trained yet")

    # Model and pause flag
    print()
    model_exists = Path(_MODEL_PATH).exists()
    paused = _PAUSE_FLAG.exists()
    print(f"  Model deployed : {'YES' if model_exists else 'NO  ← bootstrap mode'}")
    if paused:
        print(f"  {_STATUS['WARN']} PAUSED — models/regime_paused.flag exists (drawdown protection active)")
        issues.append("WARN  Regime model is paused (pause flag present)")
    else:
        print(f"  {_STATUS['PASS']} Not paused")

    return issues


# ── Section 5: Calibration curve + SHAP ─────────────────────────────────────

def section_calibration_and_shap(conn: sqlite3.Connection) -> list[str]:
    """Section 5: calibration curve by confidence tier + SHAP feature contributions."""
    issues: list[str] = []
    print(f"\n── Calibration Curve + SHAP {'─'*43}")

    # ── 5a: Calibration curve ─────────────────────────────────────────────────
    # shap_coherence column is added later; fall back to NULL if absent
    _cf_cols = {r[1] for r in conn.execute("PRAGMA table_info(candle_features)").fetchall()}
    _coh_expr = "shap_coherence" if "shap_coherence" in _cf_cols else "NULL"
    rows = conn.execute(f"""
        SELECT regime_prob, btc_direction, kalshi_open_mid, {_coh_expr}
        FROM candle_features
        WHERE features_stale=0 AND regime_prob IS NOT NULL
          AND btc_direction IS NOT NULL AND kalshi_open_mid IS NOT NULL
        ORDER BY candle_ts ASC
    """).fetchall()

    n_total = len(rows)
    if n_total == 0:
        print("  (no regime_prob rows yet — check back after regime v2 deploys)")
    else:
        tiers = [
            ("Low  (|p-0.5|<0.10)", lambda p: abs(p - 0.5) < 0.10,  False),
            ("Med  (0.10–0.20)",     lambda p: 0.10 <= abs(p - 0.5) < 0.20, False),
            ("High (|p-0.5|>0.20)", lambda p: abs(p - 0.5) >= 0.20, True),
        ]
        high_regime_brier = high_kalshi_brier = high_n = None

        for tier_name, tier_fn, is_high in tiers:
            tier_rows = [(p, y, k, c) for p, y, k, c in rows if tier_fn(p)]
            n = len(tier_rows)
            if n == 0:
                print(f"  {tier_name:<24s}  n=0   (accumulating)")
                continue
            regime_ps  = [r[0] for r in tier_rows]
            ys         = [r[1] for r in tier_rows]
            kalshi_ks  = [r[2] for r in tier_rows]
            coherences = [r[3] for r in tier_rows if r[3] is not None]

            r_brier = sum((p - y) ** 2 for p, y in zip(regime_ps, ys)) / n
            k_brier = sum((k - y) ** 2 for k, y in zip(kalshi_ks, ys)) / n
            adv = (k_brier - r_brier) / k_brier * 100 if k_brier > 0 else 0

            if n >= 10:
                win_rate = sum(ys) / n * 100
                coh_str = f"  coh={sum(coherences)/len(coherences):.2f}" if coherences else ""
                print(f"  {tier_name:<24s}  n={n:<4d}  win={win_rate:.0f}%  "
                      f"regime={r_brier:.3f}  kalshi={k_brier:.3f}  adv={adv:+.1f}%{coh_str}")
            else:
                print(f"  {tier_name:<24s}  n={n:<4d}  (accumulating — need 10+)")

            if is_high:
                high_regime_brier, high_kalshi_brier, high_n = r_brier, k_brier, n

        # Key signal: does model beat Kalshi in high-confidence tier?
        if high_n is not None and high_n >= 10:
            if high_regime_brier < high_kalshi_brier:
                print(f"  {_STATUS['PASS']} HIGH tier beats Kalshi — go-live signal present")
            else:
                msg = f"High-confidence regime ({high_regime_brier:.3f}) ≥ Kalshi ({high_kalshi_brier:.3f})"
                print(f"  {_STATUS['WARN']} {msg}")
                issues.append(f"WARN  {msg}")

        print(f"  Total regime_prob rows: {n_total}")

    # ── 5b: SHAP feature contributions ───────────────────────────────────────
    shap_path = Path(_SHAP_BASELINE_PATH)
    if shap_path.exists():
        try:
            snapshot = json.loads(shap_path.read_text())
            n_snap   = snapshot.get("n_rows", 0)
            computed = snapshot.get("computed_at", "unknown")[:19]
            print(f"\n  SHAP baseline  (n_train={n_snap}, updated {computed})")
            print(f"  {'Feature':<28s}  {'Mean|SHAP|':>10s}  {'Importance':>10s}")
            for feat in snapshot["features"][:10]:
                print(f"  {feat['name']:<28s}  {feat['mean_abs_shap']:>10.4f}  {feat['importance']:>10.3%}")
        except Exception as exc:
            print(f"  {_STATUS['WARN']} Could not read SHAP baseline: {exc}")
    else:
        print("\n  (SHAP baseline not yet available — appears after first train_regime.py run)")

    return issues


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--hours", type=int, default=24, help="Window for API health + drift checks (default: 24)")
    args = p.parse_args()

    now_str = _now_utc().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*70}")
    print(f"  Regime v2 Monitor  —  {now_str}")
    print(f"{'='*70}")

    conn = _conn(args.db)
    all_issues: list[str] = []
    try:
        all_issues += section_api_health(conn, args.hours)
        all_issues += section_distribution_drift(conn, args.hours)
        all_issues += section_kalshi_edge(conn)
        all_issues += section_training_health(conn)
        all_issues += section_calibration_and_shap(conn)
    finally:
        conn.close()

    print(f"\n── Summary {'─'*60}")
    if not all_issues:
        print(f"  {_STATUS['PASS']} All checks passed — system healthy")
    else:
        fails  = [i for i in all_issues if i.startswith("FAIL")]
        warns  = [i for i in all_issues if i.startswith("WARN")]
        if fails:
            print(f"  {_STATUS['FAIL']} {len(fails)} FAIL(s):")
            for f in fails:
                print(f"      {f}")
        if warns:
            print(f"  {_STATUS['WARN']} {len(warns)} WARN(s):")
            for w in warns:
                print(f"      {w}")
    print()


if __name__ == "__main__":
    main()
