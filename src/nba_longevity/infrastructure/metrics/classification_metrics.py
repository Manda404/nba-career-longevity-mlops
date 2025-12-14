from typing import Sequence
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    y_pred = [1 if p >= threshold else 0 for p in y_proba]

    return {
        "auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
