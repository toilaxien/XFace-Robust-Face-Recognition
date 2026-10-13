from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    return np.sum(a * b, axis=1)


def verification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    threshold = thresholds[eer_index]
    predictions = scores >= threshold
    accuracy = float((predictions == labels).mean() * 100.0)
    return {
        "accuracy": accuracy,
        "auc": float(auc),
        "eer": float(((fpr[eer_index] + fnr[eer_index]) / 2.0) * 100.0),
        "threshold": float(threshold),
    }
