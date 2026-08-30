# Uplift Modeling for Targeted Offers

Causal meta-learners (S-, T-, X-learner) for deciding **who should receive a
promotional offer**, implemented from the formulas in Kunzel, Sekhon, Bickel
& Yu (2019) and evaluated empirically on both a synthetic ground-truth DGP
and a real email-marketing RCT.

## Business framing

A retailer can send a promotional offer (a discount code, a targeted email)
to some subset of customers. Sending it to everyone is wasteful: some
customers would have purchased anyway (the offer just cuts margin on a sale
that was going to happen — "cannibalization"), and some customers are
unmoved by any offer. The quantity that actually matters for targeting is
not "how likely is this customer to respond" (a pure propensity/response
model) but **the incremental effect of the offer** — the conditional average
treatment effect (CATE), `tau(x) = E[Y(1) - Y(0) | X=x]` — the extra revenue
(or extra conversion probability) *caused* by sending the offer to a
customer with features `x`, relative to not sending it. Ranking customers by
`tau_hat(x)` and targeting the top fraction is the offer-targeting decision
this project evaluates.

## Paper

Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). "Metalearners
for estimating heterogeneous treatment effects using machine learning."
*Proceedings of the National Academy of Sciences*, 116(10), 4156-4165.
arXiv:1706.03461.

**A note on how the formulas were verified.** This sandbox's network egress
policy blocks arxiv.org, pnas.org, ncbi.nlm.nih.gov, ar5iv, and
semanticscholar.org directly (403/407 from the organization's egress proxy —
confirmed via the proxy's own status endpoint, not a transient failure), so
the paper's PDF/abstract could not be fetched in-session for a line-by-line
read. Instead, the exact S/T/X-learner formulas implemented in `learners.py`
were cross-checked against `causalml`'s `XLearner`/`TLearner` source
(`raw.githubusercontent.com`, which *is* reachable here), whose docstrings
and code cite this same paper and reproduce the identical D1/D0 imputation
and propensity-weighted combination structure the task description
specifies. The formulas below match both that reference implementation and
the well-established public description of the method.

- **S-learner** (naive baseline): one model `mu_hat(x, w)` with treatment as
  a feature. `tau_hat(x) = mu_hat(x,1) - mu_hat(x,0)`.
- **T-learner**: two models, `mu0_hat` fit only on control units, `mu1_hat`
  fit only on treated units. `tau_hat(x) = mu1_hat(x) - mu0_hat(x)`.
- **X-learner** (3 stages):
  1. Same `mu0_hat`/`mu1_hat` as the T-learner.
  2. Impute each unit's effect using the *other* arm's model, then regress
     within-arm: `D1_i = Y_i - mu0_hat(X_i)` for treated units -> regress
     `D1` on `X` (treated only) to get `tau1_hat`; `D0_i = mu1_hat(X_i) -
     Y_i` for control units -> regress `D0` on `X` (control only) to get
     `tau0_hat`.
  3. Fit a propensity model `e_hat(x) = P(T=1|x)` (logistic regression) and
     combine: `tau_hat(x) = e_hat(x)*tau0_hat(x) + (1-e_hat(x))*tau1_hat(x)`.

All three share one base-model factory (`learners.make_gbr_factory`, default
`GradientBoostingRegressor`); `test_sanity.py` swaps in a linear-regression
factory to check exact recovery on a linear DGP.

## Dataset

**Both** a real dataset and a synthetic DGP are used, for different purposes,
and `run_experiment.py` runs both every time:

- **Real data (Part 0, demo only): Hillstrom MineThatData email RCT**,
  fetched via `scikit-uplift`'s `fetch_hillstrom`. This *did* download
  successfully in this environment (`raw.githubusercontent.com`, where
  scikit-uplift hosts it, is reachable even though arxiv/PNAS are not).
  64,000 customers, 66.7% received an email offer (the "Womens E-Mail" and
  "Mens E-Mail" arms pooled into treatment=1; "No E-Mail" is control), 15
  features after one-hot-encoding recency/history/segment/zip/channel.
  This is a genuine randomized experiment, so S/T/X-learner can be fit and
  ranked with real Qini/AUUC/uplift@k. **It is demo-only and excluded from
  the PEHE-based experiments below**, for the reason the task explicitly
  calls out: no real dataset ever reveals a customer's counterfactual
  outcome (what they would have done under the *other* treatment), so
  accuracy metrics like PEHE and bias are fundamentally not computable
  against real data — only rank-based metrics are.
- **Synthetic DGP (Parts 1 and 2, the primary evaluation): `dgp.py`**, 8
  covariates (recency, frequency, monetary, tenure, engagement, plus 3 pure
  noise features), treatment assigned by an independent coin flip
  (`treat_fraction`-controlled, so it's a valid RCT by construction, not
  just a modeling assumption), and a **known** nonlinear ground-truth
  `tau(x)` that peaks for mid-engagement/high-monetary customers and goes
  negative for high-tenure/already-loyal customers (offer cannibalization).
  Because `tau(x)` is known exactly, PEHE and bias are computable — this is
  also how the paper itself validates the X-learner's claimed advantage,
  via simulation with known ground truth, not real data.

If `scikit-uplift` were unavailable or the download failed here (as it well
might in a more locked-down environment), `dgp.try_load_hillstrom()` returns
`(None, reason)` instead of raising, `run_experiment.py` prints the reason
and proceeds with the synthetic DGP for everything, and this README would
say so plainly here. In this run, the real fetch **succeeded**.

## Results

All numbers below are exactly what `run_experiment.py` produced (see
`results/*.json` for full precision); nothing here is hand-tuned after the
fact.

### Part 0 — real data demo (Hillstrom, n=64,000, 66.7% treated)

No ground truth, so only ranking metrics:

| Method    | AUUC    | Uplift@20% |
|-----------|--------:|-----------:|
| S-learner | 424.08  | 0.1945     |
| T-learner | 408.65  | 0.1964     |
| X-learner | 415.64  | **0.1995** |

X-learner picks the best top-20% cohort by true (observed) response rate;
S-learner edges out both on AUUC. With only moderate imbalance (66.7/33.3)
and no ground truth to average out sampling noise, this one real-data run
is not strong evidence either way about learner ranking — see the honest
note below.

### Part 1 — balanced run (n=20,000, 50/50 treat/control, synthetic, seed=42)

| Method    | PEHE   | Bias    | Uplift@20% | AUUC    |
|-----------|-------:|--------:|-----------:|--------:|
| S-learner | 1.1747 | -0.1919 | 3.2580     | 3290.36 |
| T-learner | 1.0481 | -0.0708 | 3.5208     | 3116.95 |
| X-learner | **0.4776** | **-0.0561** | **3.8603** | **3570.87** |
| Oracle (ranks by true tau) | 0.0000 | 0.0000 | 3.9260 | 3685.67 |
| Random targeting | n/a | n/a | 1.3108 | 2066.31 |

X-learner roughly halves PEHE relative to T-learner (0.478 vs. 1.048) and
gets closest to the oracle's Qini curve (see
`results/part1_qini_curves.png`). Note the PEHE ranking (S worse than T)
does **not** match the AUUC ranking (S better than T on AUUC) — PEHE scores
pointwise magnitude accuracy, AUUC scores rank quality, and they need not
agree; S-learner apparently gets the *relative ordering* of customers a bit
more right than T-learner here even though its `tau_hat` magnitudes are
less accurate.

### Part 2 — imbalance sweep (synthetic, n=15,000, 6 seeds per treat_fraction)

Mean PEHE ± SD:

| treat_fraction | S-learner | T-learner | X-learner |
|---------------:|----------:|----------:|----------:|
| 0.50 | 1.039 ± 0.038 | 1.105 ± 0.044 | **0.508 ± 0.037** |
| 0.35 | 1.118 ± 0.052 | 1.139 ± 0.015 | **0.555 ± 0.043** |
| 0.20 | 1.133 ± 0.068 | 1.247 ± 0.045 | **0.685 ± 0.036** |
| 0.10 | 1.165 ± 0.087 | 1.459 ± 0.030 | **0.898 ± 0.049** |
| 0.05 | 1.351 ± 0.101 | 1.757 ± 0.070 | **1.159 ± 0.100** |

See `results/part2_imbalance_sweep.png`.

**Honest note — this doesn't match the paper's headline claim as cleanly as
hoped.** The paper's central claim is that the X-learner's advantage grows
as one arm shrinks. X-learner is indeed the best of the three at *every*
imbalance level tested here (never "wildly worse," as the sanity check
requires), and T-learner does degrade fastest in absolute terms as
`treat_fraction` shrinks. But looking at the **X-learner's advantage over
T-learner specifically**:

- Absolute gap `PEHE(T) - PEHE(X)`: 0.597 (tf=0.50) -> 0.584 -> 0.562 ->
  0.561 -> 0.598 (tf=0.05) — essentially **flat**, not widening.
- Relative improvement `(PEHE(T)-PEHE(X))/PEHE(T)`: 54.0% -> 51.3% -> 45.1%
  -> 38.4% -> **34.0%** (tf=0.05) — actually **narrows** as imbalance
  increases, the opposite direction from the paper's framing.

A plausible reason: this simulation uses a mild base learner
(`GradientBoostingRegressor`, 100 trees, depth 3) and even the most
imbalanced setting tested (`treat_fraction=0.05`, n=15,000) still leaves
~500 treated units in the training split — enough for a regularized GBM to
fit a reasonably stable `mu1_hat`, so the T-learner's minority-arm model
never gets catastrophically noisy the way it would with a more flexible
base learner, higher-dimensional `X`, or a smaller `n`. The paper's own
simulations use different scales and base learners tuned to expose the
effect more sharply. The takeaway that *does* replicate robustly here (and
is what `test_sanity.py` checks) is the more modest, still economically
important one: **X-learner is never worse than T-learner under imbalance in
this DGP, and is often roughly 2x more accurate in absolute PEHE** — just
not with a cleanly widening relative gap.

## Project layout

- `dgp.py` — synthetic offer-targeting DGP + best-effort real-data loader.
- `learners.py` — S-/T-/X-learner, implemented from the paper's formulas.
- `metrics.py` — PEHE, bias, Qini curve (Radcliffe), AUUC, uplift@k.
- `run_experiment.py` — Part 0 (real-data demo), Part 1 (balanced run +
  Qini plot), Part 2 (imbalance sweep + PEHE-vs-imbalance plot). Writes
  `results/*.json` and `results/*.png`.
- `test_sanity.py` — (a) linear-DGP exact-recovery check, (b) X-learner
  beats T-learner under imbalance, averaged over seeds.

## How to run it

```bash
pip install -r requirements.txt          # core: numpy, scikit-learn, matplotlib
# optional, for the real-data demo:
pip install scikit-uplift pandas

python test_sanity.py                    # ~40s; both checks should PASS
python run_experiment.py                 # ~5 min; writes results/*.json, results/*.png
```

`run_experiment.py` requires no network access to run — if the optional
Hillstrom fetch fails or `scikit-uplift`/`pandas` aren't installed, it
prints the reason and proceeds straight to the synthetic Parts 1 and 2.
