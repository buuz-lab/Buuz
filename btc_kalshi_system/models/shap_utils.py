from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xgboost as xgb


def compute_coherence(clf, X: np.ndarray) -> float:
    """
    Fraction of the model's feature SHAP contributions pointing in the same
    direction as the final prediction. Score near 1.0 = coherent feature
    agreement; near 0.5 = features split, one dominant feature driving the call.

    Uses XGBoost's built-in pred_contribs — no external shap library needed.
    X must be shape (1, n_features).
    """
    assert X.shape[0] == 1, f"compute_coherence expects shape (1, n_features), got {X.shape}"
    booster = clf.get_booster()
    dmatrix = xgb.DMatrix(X)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]   # shape: (n_features + 1,)
    feature_contribs = contribs[:-1]                    # drop bias column
    total_prediction = contribs.sum()                   # full margin including bias
    if total_prediction == 0.0 or not np.isfinite(total_prediction):
        return 0.5
    pred_sign = 1 if total_prediction > 0 else -1
    n_agree = int(np.sum(feature_contribs * pred_sign > 0))
    return round(float(n_agree / len(feature_contribs)), 4)


def compute_baseline_snapshot(
    clf,
    X_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Mean absolute SHAP per feature over the training set, sorted descending.
    Returns a JSON-serializable dict for saving to models/regime_shap_baseline.json.
    Updated on every train_regime.py run.
    """
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != X_train n_features {X_train.shape[1]}"
        )
    booster = clf.get_booster()
    dmatrix = xgb.DMatrix(X_train)
    contribs = booster.predict(dmatrix, pred_contribs=True)[:, :-1]  # drop bias
    mean_abs_shap = np.nanmean(np.abs(contribs), axis=0)
    importances = clf.feature_importances_
    total_imp = float(importances.sum()) or 1.0

    ranked = sorted(
        zip(feature_names, mean_abs_shap.tolist(), importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "features": [
            {
                "name": name,
                "mean_abs_shap": round(shap, 6),
                "importance": round(imp / total_imp, 6),
            }
            for name, shap, imp in ranked
        ],
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X_train)),
    }
