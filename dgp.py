"""Data sources for the uplift-modeling experiments.

Primary source: a synthetic "offer targeting" data-generating process (DGP)
with a KNOWN, nonlinear ground-truth treatment effect tau(x). This is what
Part 1 / Part 2 of run_experiment.py evaluate on, because accuracy metrics
like PEHE and bias require the individual counterfactual outcome, which no
real dataset ever reveals -- only a simulation with a known tau(x) can be
scored that way. This is also how Kunzel et al. (2019) validate their own
claims about the X-learner.

Secondary source (best-effort): the real Hillstrom MineThatData email RCT,
fetched via the optional `scikit-uplift` package. It has no ground-truth
tau(x), so it is only usable for ranking-based metrics (Qini/AUUC/uplift@k),
never for PEHE/bias. If the package isn't installed or the download fails
(e.g. no network access), `try_load_hillstrom` returns a clear failure
reason instead of raising, and callers fall back to the synthetic DGP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

FEATURE_NAMES: List[str] = [
    "recency",
    "frequency",
    "monetary",
    "tenure",
    "engagement",
    "noise1",
    "noise2",
    "noise3",
]

# Shape parameters of the ground-truth uplift surface tau(x). Uplift peaks
# for mid-engagement, high-monetary customers (the offer converts customers
# who are interested but not yet big spenders) and goes negative for
# high-tenure/already-loyal customers (the offer just discounts a purchase
# they were going to make anyway -- cannibalization).
PEAK_UPLIFT = 6.0
ENGAGEMENT_CENTER = 0.62
ENGAGEMENT_WIDTH = 0.15
TENURE_THRESHOLD = 0.65
CANNIBALIZATION_STRENGTH = 5.0


@dataclass
class Dataset:
    """A treatment-effect dataset: covariates, treatment, observed outcome,
    and (when available) the ground-truth CATE tau_true used to score PEHE."""

    X: np.ndarray
    treatment: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    name: str
    tau_true: Optional[np.ndarray] = None


def true_tau(X: np.ndarray) -> np.ndarray:
    """Ground-truth nonlinear CATE for the synthetic DGP.

    Depends only on monetary, tenure, and engagement (columns 2, 3, 4 of the
    synthetic feature matrix); recency/frequency and the 3 noise columns do
    not affect it.
    """
    monetary = X[:, 2]
    tenure = X[:, 3]
    engagement = X[:, 4]
    peak = PEAK_UPLIFT * np.exp(
        -((engagement - ENGAGEMENT_CENTER) ** 2) / (2 * ENGAGEMENT_WIDTH ** 2)
    )
    monetary_boost = 0.4 + 0.6 * monetary
    cannibalization = CANNIBALIZATION_STRENGTH * np.clip(tenure - TENURE_THRESHOLD, 0.0, None)
    return peak * monetary_boost - cannibalization


def _baseline_mu0(X: np.ndarray) -> np.ndarray:
    """Untreated potential outcome mu0(x) -- a smooth function of the 5
    "real" covariates; the 3 noise columns do not enter it either."""
    recency, frequency, monetary, tenure, engagement = (X[:, i] for i in range(5))
    return 20.0 - 8.0 * recency + 10.0 * frequency + 25.0 * monetary + 6.0 * tenure + 5.0 * engagement


def _linear_effects(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """A fully linear mu0(x)/tau(x) pair, used only by the sanity tests to
    check that linear-regression base learners recover a linear tau(x)
    almost exactly (no gaussian bump / clipping that a linear model can't
    represent)."""
    recency, frequency, monetary, tenure, engagement = (X[:, i] for i in range(5))
    mu0 = 5.0 + 2.0 * recency + 3.0 * frequency + 4.0 * monetary + 1.5 * tenure + 2.5 * engagement
    tau = 1.0 + 3.0 * engagement + 2.0 * monetary - 2.5 * tenure
    return mu0, tau


def generate_synthetic(
    n: int = 20_000,
    treat_fraction: float = 0.5,
    seed: int = 0,
    sigma: float = 3.0,
    linear: bool = False,
) -> Dataset:
    """Simulate n customers as a valid RCT: treatment is assigned by a coin
    flip independent of X, so treat_fraction controls arm imbalance without
    introducing confounding.

    Set linear=True to swap in a fully linear mu0(x)/tau(x) pair (used by
    test_sanity.py to check exact recovery with linear base learners).
    """
    rng = np.random.default_rng(seed)

    recency = rng.uniform(0, 1, n)
    frequency = rng.uniform(0, 1, n)
    monetary = rng.uniform(0, 1, n)
    tenure = rng.uniform(0, 1, n)
    engagement = rng.uniform(0, 1, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.uniform(-1, 1, n)
    noise3 = rng.integers(0, 2, n).astype(float)
    X = np.column_stack(
        [recency, frequency, monetary, tenure, engagement, noise1, noise2, noise3]
    )

    treatment = (rng.uniform(0, 1, n) < treat_fraction).astype(int)

    if linear:
        mu0, tau = _linear_effects(X)
    else:
        mu0 = _baseline_mu0(X)
        tau = true_tau(X)

    eps = rng.normal(0, sigma, n)
    y = mu0 + treatment * tau + eps

    return Dataset(
        X=X,
        treatment=treatment,
        y=y,
        feature_names=list(FEATURE_NAMES),
        name="synthetic-linear" if linear else "synthetic",
        tau_true=tau,
    )


def try_load_hillstrom(target_col: str = "visit") -> Tuple[Optional[Dataset], Optional[str]]:
    """Best-effort fetch of the real Hillstrom MineThatData email RCT via the
    optional `scikit-uplift` package. Never raises: returns (Dataset, None)
    on success, or (None, reason) on any failure (package missing, no
    network, API mismatch, ...) so callers can fall back to the synthetic DGP.

    Two of the three original arms ("Mens E-Mail", "Womens E-Mail") are
    pooled into treatment=1 ("received an email offer"); "No E-Mail" is
    treatment=0. There is no ground-truth tau_true for real customers, so
    the returned Dataset has tau_true=None.
    """
    try:
        from sklift.datasets import fetch_hillstrom
    except ImportError as exc:
        return None, f"scikit-uplift is not installed ({exc})"

    try:
        import pandas as pd

        bunch = fetch_hillstrom(target_col=target_col)
        raw = bunch.data
        treatment_raw = bunch.treatment
        y = bunch.target.to_numpy(dtype=float)

        cat_cols = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
        X_df = pd.get_dummies(raw, columns=cat_cols, drop_first=True)
        feature_names = list(X_df.columns)
        X = X_df.to_numpy(dtype=float)
        treatment = (treatment_raw != "No E-Mail").to_numpy().astype(int)
    except Exception as exc:  # noqa: BLE001 - any failure -> caller falls back
        return None, f"download/processing failed ({type(exc).__name__}: {exc})"

    return (
        Dataset(
            X=X,
            treatment=treatment,
            y=y,
            feature_names=feature_names,
            name="Hillstrom MineThatData email RCT",
            tau_true=None,
        ),
        None,
    )
