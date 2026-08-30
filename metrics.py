"""Evaluation metrics for uplift models.

pehe/bias require ground-truth tau (only available in simulation);
qini_curve/auuc/uplift_at_k only need actually-observed (treatment, outcome)
pairs, so they also apply to real data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def pehe(tau_hat, tau_true) -> float:
    """Precision in Estimation of Heterogeneous Effects:
    sqrt(mean((tau_hat - tau_true)^2)).
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau_true = np.asarray(tau_true, dtype=float)
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def bias(tau_hat, tau_true) -> float:
    """Mean signed error: mean(tau_hat - tau_true)."""
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau_true = np.asarray(tau_true, dtype=float)
    return float(np.mean(tau_hat - tau_true))


def uplift_at_k(tau_hat, tau_true, k: float = 0.2) -> float:
    """Mean TRUE effect among the top-k fraction of customers ranked by
    tau_hat -- "how good are the customers this model would actually pick."
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau_true = np.asarray(tau_true, dtype=float)
    n = len(tau_hat)
    n_top = max(1, int(np.ceil(n * k)))
    order = np.argsort(-tau_hat)
    top_idx = order[:n_top]
    return float(np.mean(tau_true[top_idx]))


def qini_curve(tau_hat, treatment, y) -> Tuple[np.ndarray, np.ndarray]:
    """Radcliffe's Qini-curve construction, computed from actually-observed
    (treatment, outcome) pairs after ranking customers by tau_hat descending.

    At the point where the top i customers (by predicted uplift) have been
    "targeted":
        qini(i) = Y1_cum(i) - Y0_cum(i) * (N1_cum(i) / N0_cum(i))
    where Y1_cum/Y0_cum are cumulative outcome sums among treated/control
    units in that top-i slice and N1_cum/N0_cum are their cumulative counts.
    This rescales the control response by the treated/control ratio so
    curves stay comparable even when the two arms are imbalanced.

    Returns (fractions, qini_values), both starting at (0, 0).
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    treatment = np.asarray(treatment)
    y = np.asarray(y, dtype=float)
    n = len(tau_hat)

    order = np.argsort(-tau_hat)
    treatment_sorted = treatment[order]
    y_sorted = y[order]

    is_treated = (treatment_sorted == 1).astype(float)
    is_control = 1.0 - is_treated

    n1_cum = np.cumsum(is_treated)
    n0_cum = np.cumsum(is_control)
    y1_cum = np.cumsum(y_sorted * is_treated)
    y0_cum = np.cumsum(y_sorted * is_control)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n0_cum > 0, n1_cum / n0_cum, 0.0)
    qini_values = y1_cum - y0_cum * ratio

    fractions = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    qini_values = np.concatenate([[0.0], qini_values])
    return fractions, qini_values


def auuc(fractions: np.ndarray, qini_values: np.ndarray) -> float:
    """Area Under the Uplift Curve: trapezoid-rule integral of the Qini
    curve over the fraction-of-population axis."""
    fractions = np.asarray(fractions, dtype=float)
    qini_values = np.asarray(qini_values, dtype=float)
    widths = np.diff(fractions)
    heights = (qini_values[1:] + qini_values[:-1]) / 2.0
    return float(np.sum(widths * heights))
