from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def screen_features(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    *,
    label_col: str = "label",
    min_auc: float = 0.55,
    min_pr_auc: float | None = None,
) -> pd.DataFrame:
    """Rank candidate features using labels from a validation-only frame."""
    labels = frame[label_col].to_numpy(dtype=np.int8)
    prevalence = float(labels.mean()) if labels.size else 0.0
    pr_floor = prevalence if min_pr_auc is None else min_pr_auc
    rows: list[dict[str, float | str]] = []
    for column in candidate_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0 or np.unique(labels[finite]).size < 2:
            continue
        auc = float(roc_auc_score(labels[finite], values[finite]))
        pr_auc = float(average_precision_score(labels[finite], values[finite]))
        rows.append({
            "feature": column,
            "roc_auc": max(auc, 1.0 - auc),
            "pr_auc": pr_auc,
            "finite_fraction": float(finite.mean()),
            "selected": bool(max(auc, 1.0 - auc) >= min_auc and pr_auc >= pr_floor),
        })
    return pd.DataFrame(rows, columns=["feature", "roc_auc", "pr_auc", "finite_fraction", "selected"]).sort_values(
        ["selected", "roc_auc", "pr_auc"], ascending=False
    ).reset_index(drop=True)
