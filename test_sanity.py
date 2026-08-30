"""Sanity checks for the meta-learner implementations.

Runnable directly (`python test_sanity.py`) or via pytest.

(a) On a fully linear DGP with linear-regression base learners, both the
    T-learner and X-learner should recover the known linear tau(x) almost
    exactly. This catches sign errors or swapped mu0/mu1.
(b) Averaged over multiple seeds at a meaningfully imbalanced
    treat_fraction, the X-learner's mean PEHE should be lower than the
    T-learner's -- confirms the core result from Kunzel et al. (2019) isn't
    a single lucky seed.
"""

from __future__ import annotations

import sys

import numpy as np

from dgp import generate_synthetic
from learners import SLearner, TLearner, XLearner, make_gbr_factory, make_linear_factory
from metrics import pehe


LINEAR_PEHE_THRESHOLD = 0.3
IMBALANCED_TREAT_FRACTION = 0.1
IMBALANCED_N_SEEDS = 8


def test_linear_recovery():
    """(a) Linear DGP + linear base learners -> near-exact tau recovery."""
    ds = generate_synthetic(n=6000, treat_fraction=0.5, seed=0, sigma=0.1, linear=True)

    t_learner = TLearner(base_model_factory=make_linear_factory()).fit(ds.X, ds.treatment, ds.y)
    x_learner = XLearner(base_model_factory=make_linear_factory()).fit(ds.X, ds.treatment, ds.y)

    t_pehe = pehe(t_learner.predict(ds.X), ds.tau_true)
    x_pehe = pehe(x_learner.predict(ds.X), ds.tau_true)

    print(f"  T-learner PEHE on linear DGP: {t_pehe:.4f}")
    print(f"  X-learner PEHE on linear DGP: {x_pehe:.4f}")

    assert t_pehe < LINEAR_PEHE_THRESHOLD, (
        f"T-learner PEHE {t_pehe:.4f} exceeds threshold {LINEAR_PEHE_THRESHOLD} "
        "on a linear DGP with linear base learners -- check for a sign error "
        "or swapped mu0_hat/mu1_hat."
    )
    assert x_pehe < LINEAR_PEHE_THRESHOLD, (
        f"X-learner PEHE {x_pehe:.4f} exceeds threshold {LINEAR_PEHE_THRESHOLD} "
        "on a linear DGP with linear base learners -- check the D1/D0 "
        "imputation or the propensity-weighted combination step."
    )


def test_xlearner_beats_tlearner_under_imbalance():
    """(b) Under meaningful treatment-arm imbalance, X-learner's mean PEHE
    (averaged over several seeds) should beat T-learner's."""
    t_pehes = []
    x_pehes = []

    for seed in range(IMBALANCED_N_SEEDS):
        ds = generate_synthetic(n=8000, treat_fraction=IMBALANCED_TREAT_FRACTION, seed=seed)
        base_factory = make_gbr_factory(random_state=seed)

        t_learner = TLearner(base_model_factory=base_factory).fit(ds.X, ds.treatment, ds.y)
        x_learner = XLearner(base_model_factory=base_factory).fit(ds.X, ds.treatment, ds.y)

        t_pehes.append(pehe(t_learner.predict(ds.X), ds.tau_true))
        x_pehes.append(pehe(x_learner.predict(ds.X), ds.tau_true))

    t_mean, x_mean = float(np.mean(t_pehes)), float(np.mean(x_pehes))
    print(f"  T-learner mean PEHE over {IMBALANCED_N_SEEDS} seeds "
          f"(treat_fraction={IMBALANCED_TREAT_FRACTION}): {t_mean:.4f}")
    print(f"  X-learner mean PEHE over {IMBALANCED_N_SEEDS} seeds "
          f"(treat_fraction={IMBALANCED_TREAT_FRACTION}): {x_mean:.4f}")

    assert x_mean < t_mean, (
        f"Expected X-learner mean PEHE ({x_mean:.4f}) to beat T-learner "
        f"({t_mean:.4f}) under treat_fraction={IMBALANCED_TREAT_FRACTION}, "
        "the paper's central claim about the X-learner's advantage under "
        "arm imbalance."
    )


def test_all_learners_produce_finite_predictions():
    """Basic smoke test: S/T/X-learners all fit and predict finite values."""
    ds = generate_synthetic(n=1000, treat_fraction=0.3, seed=1)
    for cls in (SLearner, TLearner, XLearner):
        learner = cls().fit(ds.X, ds.treatment, ds.y)
        tau_hat = learner.predict(ds.X)
        assert np.all(np.isfinite(tau_hat)), f"{cls.__name__} produced non-finite predictions"


TESTS = [
    test_linear_recovery,
    test_xlearner_beats_tlearner_under_imbalance,
    test_all_learners_produce_finite_predictions,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        print(f"Running {test.__name__} ...")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAILED: {exc}")
        else:
            print("  PASSED")
    print()
    if failures:
        print(f"{failures}/{len(TESTS)} sanity check(s) FAILED")
        return 1
    print(f"All {len(TESTS)} sanity checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
