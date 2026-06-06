from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict[str, float]:
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except Exception:
            metrics["ROC-AUC"] = float("nan")
    else:
        metrics["ROC-AUC"] = float("nan")
    return metrics


def score_estimator(
    estimator: Any, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    return evaluate_predictions(y_test, y_pred, y_score), y_pred, y_score
