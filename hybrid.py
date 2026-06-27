"""Hybrid scoring: combine KGE anomaly scores with the logical signal (LO2, LO12).

This is the concrete realisation of the one-pager's central idea — *"the output
of the logical rules feeds into the embedding model ... to refine the detection
accuracy."*

The embedding model (``pipeline.score``) assigns every candidate triple a
plausibility ``score`` (higher = more plausible).  We turn that into an
**anomaly** score (lower plausibility => more anomalous) and then *boost* it for
triples that the symbolic reasoner has implicated in a lateral-movement / attack
chain:

    hybrid_anomaly = zscore(-kge_score) + lambda * logic_flag

where ``logic_flag`` is 1 when the triple's head or tail is a flagged entity.
The function :func:`evaluate_models` reports ROC-AUC / average precision for the
pure-KGE baseline and for the hybrid score so the report can show whether
symbolic + sub-symbolic together beat either alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def logic_flags(df: pd.DataFrame, flagged: set[str]) -> np.ndarray:
    """1.0 where a triple's head or tail is a flagged entity, else 0.0."""
    if not flagged:
        return np.zeros(len(df), dtype=float)
    head = df["head"].astype(str)
    tail = df["tail"].astype(str)
    return ((head.isin(flagged)) | (tail.isin(flagged))).to_numpy(dtype=float)


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardise to zero mean / unit variance (constant arrays -> zeros)."""
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def kge_anomaly(df: pd.DataFrame) -> np.ndarray:
    """Anomaly score from the embedding model: lower plausibility = higher."""
    return -df["score"].to_numpy(dtype=float)


def hybrid_anomaly(df: pd.DataFrame, flagged: set[str], lam: float = 1.0) -> np.ndarray:
    """Combine the standardised KGE anomaly score with the logical flag."""
    return zscore(kge_anomaly(df)) + lam * logic_flags(df, flagged)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_frame(
    df: pd.DataFrame, flagged: set[str], lam: float = 1.0
) -> dict:
    """Compute KGE-only and hybrid metrics for one labelled score frame.

    ``df`` must have columns ``head``, ``tail``, ``score`` and a binary
    ``label`` (1 = malicious / red-team, 0 = normal).
    """
    labels = df["label"].to_numpy(dtype=int)
    kge = kge_anomaly(df)
    hybrid = hybrid_anomaly(df, flagged, lam)
    flag = logic_flags(df, flagged)
    out = {
        "kge": _metrics(labels, kge),
        "hybrid": _metrics(labels, hybrid),
        "flag_coverage": float(flag.mean()),
        "flagged_positive_rate": float(flag[labels == 1].mean()) if (labels == 1).any() else 0.0,
        "flagged_negative_rate": float(flag[labels == 0].mean()) if (labels == 0).any() else 0.0,
    }
    out["roc_auc_delta"] = out["hybrid"]["roc_auc"] - out["kge"]["roc_auc"]
    return out


def labelled_frame(normal: pd.DataFrame, red: pd.DataFrame) -> pd.DataFrame:
    """Concatenate normal (label 0) and red-team (label 1) score frames."""
    normal = normal.copy()
    red = red.copy()
    normal["label"] = 0
    red["label"] = 1
    return pd.concat([normal, red], ignore_index=True)
