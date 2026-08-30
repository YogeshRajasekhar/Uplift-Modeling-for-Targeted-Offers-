"""Meta-learners for CATE estimation, implemented directly from the formulas
in Kunzel, Sekhon, Bickel & Yu (2019), "Metalearners for estimating
heterogeneous treatment effects using machine learning," PNAS 116(10):
4156-4165 (arXiv:1706.03461) -- not a wrapper around causalml/econml.

All three learners share one base-model factory (a zero-argument callable
returning an unfitted scikit-learn regressor), so swapping
GradientBoostingRegressor for LinearRegression (as the sanity tests do)
changes only the factory, not the learner code.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

BaseModelFactory = Callable[[], BaseEstimator]


def make_gbr_factory(random_state: Optional[int] = None, **kwargs) -> BaseModelFactory:
    """Default base-learner factory: gradient-boosted regression trees."""
    params = dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state)
    params.update(kwargs)
    return lambda: GradientBoostingRegressor(**params)


def make_linear_factory() -> BaseModelFactory:
    """Linear-regression base-learner factory, used by the sanity tests to
    check exact recovery of a linear tau(x)."""
    from sklearn.linear_model import LinearRegression

    return lambda: LinearRegression()


def _as_array(X) -> np.ndarray:
    return np.asarray(X)


class SLearner:
    """Single model with treatment as a feature (the naive baseline):
    tau_hat(x) = mu_hat(x, 1) - mu_hat(x, 0).
    """

    def __init__(self, base_model_factory: Optional[BaseModelFactory] = None):
        self.base_model_factory = base_model_factory or make_gbr_factory()

    def fit(self, X, treatment, y) -> "SLearner":
        X, treatment, y = _as_array(X), _as_array(treatment), _as_array(y)
        X_aug = np.column_stack([X, treatment])
        self.model_ = self.base_model_factory().fit(X_aug, y)
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_array(X)
        n = X.shape[0]
        X0 = np.column_stack([X, np.zeros(n)])
        X1 = np.column_stack([X, np.ones(n)])
        return self.model_.predict(X1) - self.model_.predict(X0)


class TLearner:
    """Two separate response models, one per arm:
    mu0_hat fit on control units, mu1_hat fit on treated units;
    tau_hat(x) = mu1_hat(x) - mu0_hat(x).
    """

    def __init__(self, base_model_factory: Optional[BaseModelFactory] = None):
        self.base_model_factory = base_model_factory or make_gbr_factory()

    def fit(self, X, treatment, y) -> "TLearner":
        X, treatment, y = _as_array(X), _as_array(treatment), _as_array(y)
        mask1 = treatment == 1
        mask0 = ~mask1
        self.mu0_ = self.base_model_factory().fit(X[mask0], y[mask0])
        self.mu1_ = self.base_model_factory().fit(X[mask1], y[mask1])
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_array(X)
        return self.mu1_.predict(X) - self.mu0_.predict(X)


class XLearner:
    """Three-stage X-learner (Kunzel et al. 2019, Section 2 / Algorithm 2):

    Stage 1 -- same as the T-learner: mu0_hat fit on control units,
    mu1_hat fit on treated units.

    Stage 2 -- impute each unit's individual treatment effect using the
    OTHER arm's outcome model, then regress the imputed effects on X within
    each arm:
        D1_i = Y_i - mu0_hat(X_i)   for treated units  -> regress D1 on X
               (treated units only) to get tau1_hat
        D0_i = mu1_hat(X_i) - Y_i   for control units  -> regress D0 on X
               (control units only) to get tau0_hat

    Stage 3 -- combine the two CATE estimates with a propensity model
    e_hat(x) = P(T=1|x):
        tau_hat(x) = e_hat(x) * tau0_hat(x) + (1 - e_hat(x)) * tau1_hat(x)
    """

    def __init__(
        self,
        base_model_factory: Optional[BaseModelFactory] = None,
        propensity_model: Optional[BaseEstimator] = None,
    ):
        self.base_model_factory = base_model_factory or make_gbr_factory()
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)

    def fit(self, X, treatment, y) -> "XLearner":
        X, treatment, y = _as_array(X), _as_array(treatment), _as_array(y)
        mask1 = treatment == 1
        mask0 = ~mask1
        X0, y0 = X[mask0], y[mask0]
        X1, y1 = X[mask1], y[mask1]

        # Stage 1
        self.mu0_ = self.base_model_factory().fit(X0, y0)
        self.mu1_ = self.base_model_factory().fit(X1, y1)

        # Stage 2
        D1 = y1 - self.mu0_.predict(X1)
        D0 = self.mu1_.predict(X0) - y0
        self.tau1_model_ = self.base_model_factory().fit(X1, D1)
        self.tau0_model_ = self.base_model_factory().fit(X0, D0)

        # Stage 3
        from sklearn.base import clone

        self.propensity_model_ = clone(self.propensity_model).fit(X, treatment)
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_array(X)
        e_hat = self.propensity_model_.predict_proba(X)[:, 1]
        tau1_hat = self.tau1_model_.predict(X)
        tau0_hat = self.tau0_model_.predict(X)
        return e_hat * tau0_hat + (1.0 - e_hat) * tau1_hat
