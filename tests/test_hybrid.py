"""Tests for hybrid (logic + KGE) scoring (LO2, LO12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hybrid


def _frame():
    # 4 normal triples (label 0) and 2 red-team triples (label 1).
    # KGE score is *worse* (less negative) at separating them than the logic
    # flag, so the hybrid score should help.
    normal = pd.DataFrame(
        {
            "head": ["user:a", "user:b", "user:c", "user:d"],
            "relation": ["logs_on_to"] * 4,
            "tail": ["computer:1", "computer:2", "computer:3", "computer:4"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    red = pd.DataFrame(
        {
            "head": ["user:evil", "user:evil"],
            "relation": ["logs_on_to", "logs_on_to"],
            "tail": ["computer:dc", "computer:fs"],
            "score": [0.55, 0.5],
        }
    )
    return hybrid.labelled_frame(normal, red)


def test_logic_flags_match_entities():
    df = _frame()
    flagged = {"user:evil", "computer:dc"}
    flags = hybrid.logic_flags(df, flagged)
    assert flags.sum() == 2  # both red-team rows touch a flagged entity
    assert flags[df["label"].to_numpy() == 0].sum() == 0


def test_zscore_constant_is_zero():
    assert np.allclose(hybrid.zscore(np.array([3.0, 3.0, 3.0])), 0.0)


def test_hybrid_improves_or_matches_auc():
    df = _frame()
    flagged = {"user:evil"}
    res = hybrid.evaluate_frame(df, flagged, lam=2.0)
    assert res["hybrid"]["roc_auc"] >= res["kge"]["roc_auc"]
    assert res["roc_auc_delta"] >= 0
    assert res["flagged_positive_rate"] == 1.0
    assert res["flagged_negative_rate"] == 0.0


def test_no_flags_leaves_scores_unchanged():
    df = _frame()
    res = hybrid.evaluate_frame(df, set(), lam=1.0)
    assert res["roc_auc_delta"] == 0.0
