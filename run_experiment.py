"""Two experiments comparing S/T/X meta-learners on the synthetic
offer-targeting DGP, plus a best-effort demo on the real Hillstrom dataset.

Part 1 -- a single balanced (50/50) run, n~20,000: PEHE, bias, uplift@20%
          for S/T/X-learner plus an oracle (ranks by true tau) and a
          random-targeting baseline, and Qini curves with AUUC.
Part 2 -- an imbalance sweep over treat_fraction in
          [0.5, 0.35, 0.2, 0.1, 0.05], several seeds each: PEHE (mean +/- SD)
          vs. treat_fraction for S/T/X-learner, testing the paper's central
          claim that the X-learner is especially advantageous when one
          treatment arm is much smaller than the other.

Saves numeric results to results/*.json and figures to results/*.png.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgp import generate_synthetic, try_load_hillstrom
from learners import SLearner, TLearner, XLearner, make_gbr_factory
from metrics import auuc, bias, pehe, qini_curve, uplift_at_k

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LEARNER_CLASSES = {"S-learner": SLearner, "T-learner": TLearner, "X-learner": XLearner}
COLORS = {"S-learner": "#d62728", "T-learner": "#1f77b4", "X-learner": "#2ca02c",
          "Oracle": "#7f7f7f", "Random": "#7f7f7f"}


def fit_learners(X_train, t_train, y_train, seed):
    fitted = {}
    for name, cls in LEARNER_CLASSES.items():
        base_factory = make_gbr_factory(random_state=seed)
        fitted[name] = cls(base_model_factory=base_factory).fit(X_train, t_train, y_train)
    return fitted


# --------------------------------------------------------------------------
# Part 0: best-effort real-data demo (no ground truth -> no PEHE/bias)
# --------------------------------------------------------------------------

def run_real_data_demo() -> dict:
    print("=" * 70)
    print("Part 0: attempting to load real public uplift dataset (Hillstrom)")
    print("=" * 70)

    dataset, reason = try_load_hillstrom()
    if dataset is None:
        print(f"Real dataset NOT used: {reason}")
        print("Falling back to the synthetic DGP for all experiments below.")
        return {"used": False, "reason": reason}

    print(f"Loaded real dataset: {dataset.name} "
          f"(n={len(dataset.y)}, {dataset.treatment.mean():.1%} treated, "
          f"{dataset.X.shape[1]} features after one-hot encoding).")
    print("Ground truth tau(x) is unavailable for real customers, so only "
          "ranking-based metrics (Qini/AUUC/uplift@k) are computed here -- "
          "no PEHE/bias.")

    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        dataset.X, dataset.treatment, dataset.y, test_size=0.3, random_state=0,
        stratify=dataset.treatment,
    )
    fitted = fit_learners(X_train, t_train, y_train, seed=0)

    result = {"used": True, "name": dataset.name, "n": int(len(dataset.y)),
              "treat_fraction": float(dataset.treatment.mean()), "learners": {}}
    for name, learner in fitted.items():
        tau_hat = learner.predict(X_test)
        fractions, qini_values = qini_curve(tau_hat, t_test, y_test)
        result["learners"][name] = {
            "auuc": auuc(fractions, qini_values),
            "uplift_at_20pct": uplift_at_k(tau_hat, y_test, k=0.2),
        }
        print(f"  {name:10s} AUUC={result['learners'][name]['auuc']:.3f}  "
              f"uplift@20%(observed y proxy)={result['learners'][name]['uplift_at_20pct']:.4f}")

    with open(RESULTS_DIR / "real_data_demo.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# --------------------------------------------------------------------------
# Part 1: single balanced run
# --------------------------------------------------------------------------

def run_part1(n=20_000, treat_fraction=0.5, seed=42) -> dict:
    print()
    print("=" * 70)
    print(f"Part 1: balanced run (n={n}, treat_fraction={treat_fraction}, seed={seed})")
    print("=" * 70)

    ds = generate_synthetic(n=n, treat_fraction=treat_fraction, seed=seed)
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, random_state=seed, stratify=ds.treatment
    )

    X_train, t_train, y_train = ds.X[idx_train], ds.treatment[idx_train], ds.y[idx_train]
    X_test, t_test, y_test = ds.X[idx_test], ds.treatment[idx_test], ds.y[idx_test]
    tau_test = ds.tau_true[idx_test]

    fitted = fit_learners(X_train, t_train, y_train, seed=seed)

    rng = np.random.default_rng(seed)
    random_scores = rng.uniform(0, 1, len(idx_test))

    table = {}
    curves = {}

    for name, learner in fitted.items():
        tau_hat = learner.predict(X_test)
        fractions, qini_values = qini_curve(tau_hat, t_test, y_test)
        table[name] = {
            "pehe": pehe(tau_hat, tau_test),
            "bias": bias(tau_hat, tau_test),
            "uplift_at_20pct": uplift_at_k(tau_hat, tau_test, k=0.2),
            "auuc": auuc(fractions, qini_values),
        }
        curves[name] = {"fractions": fractions.tolist(), "qini_values": qini_values.tolist()}

    # Oracle: ranks by the true tau -- trivially perfect PEHE/bias.
    fractions, qini_values = qini_curve(tau_test, t_test, y_test)
    table["Oracle"] = {
        "pehe": pehe(tau_test, tau_test),
        "bias": bias(tau_test, tau_test),
        "uplift_at_20pct": uplift_at_k(tau_test, tau_test, k=0.2),
        "auuc": auuc(fractions, qini_values),
    }
    curves["Oracle"] = {"fractions": fractions.tolist(), "qini_values": qini_values.tolist()}

    # Random targeting: ranks by an uninformative random score.
    fractions, qini_values = qini_curve(random_scores, t_test, y_test)
    table["Random"] = {
        "pehe": None,
        "bias": None,
        "uplift_at_20pct": uplift_at_k(random_scores, tau_test, k=0.2),
        "auuc": auuc(fractions, qini_values),
    }
    curves["Random"] = {"fractions": fractions.tolist(), "qini_values": qini_values.tolist()}

    print(f"{'Method':<12}{'PEHE':>10}{'Bias':>10}{'Uplift@20%':>14}{'AUUC':>10}")
    for name, m in table.items():
        pehe_s = f"{m['pehe']:.4f}" if m["pehe"] is not None else "n/a"
        bias_s = f"{m['bias']:.4f}" if m["bias"] is not None else "n/a"
        print(f"{name:<12}{pehe_s:>10}{bias_s:>10}{m['uplift_at_20pct']:>14.4f}{m['auuc']:>10.3f}")

    plot_qini_curves(curves, path=RESULTS_DIR / "part1_qini_curves.png")

    result = {"n": n, "treat_fraction": treat_fraction, "seed": seed, "table": table}
    with open(RESULTS_DIR / "part1_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_qini_curves(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in ("S-learner", "T-learner", "X-learner", "Oracle"):
        c = curves[name]
        fractions, qini_values = np.array(c["fractions"]), np.array(c["qini_values"])
        aucc = auuc(fractions, qini_values)
        style = "--" if name == "Oracle" else "-"
        ax.plot(fractions, qini_values, style, color=COLORS[name], lw=2,
                 label=f"{name} (AUUC={aucc:.2f})")

    random_c = curves["Random"]
    fr = np.array(random_c["fractions"])
    end_val = random_c["qini_values"][-1]
    ax.plot(fr, fr * end_val, ":", color="black", lw=1.5, label="Random targeting")

    ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    ax.set_ylabel("Qini: cumulative incremental outcome")
    ax.set_title("Qini curves -- S/T/X-learner vs. oracle vs. random")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------
# Part 2: imbalance sweep
# --------------------------------------------------------------------------

def run_part2(treat_fractions=(0.5, 0.35, 0.2, 0.1, 0.05), n_seeds=6, n=15_000) -> dict:
    print()
    print("=" * 70)
    print(f"Part 2: imbalance sweep, treat_fraction in {treat_fractions}, "
          f"{n_seeds} seeds each (n={n})")
    print("=" * 70)

    per_learner = {name: {tf: [] for tf in treat_fractions} for name in LEARNER_CLASSES}
    t0 = time.time()

    for tf in treat_fractions:
        for seed in range(n_seeds):
            ds = generate_synthetic(n=n, treat_fraction=tf, seed=1000 + seed)
            idx = np.arange(n)
            idx_train, idx_test = train_test_split(
                idx, test_size=0.3, random_state=seed, stratify=ds.treatment
            )
            X_train, t_train, y_train = ds.X[idx_train], ds.treatment[idx_train], ds.y[idx_train]
            X_test = ds.X[idx_test]
            tau_test = ds.tau_true[idx_test]

            fitted = fit_learners(X_train, t_train, y_train, seed=seed)
            for name, learner in fitted.items():
                tau_hat = learner.predict(X_test)
                per_learner[name][tf].append(pehe(tau_hat, tau_test))

        elapsed = time.time() - t0
        print(f"  treat_fraction={tf:<5} done ({elapsed:.0f}s elapsed)")

    summary = {
        name: {
            str(tf): {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "values": vals}
            for tf, vals in tf_map.items()
        }
        for name, tf_map in per_learner.items()
    }

    print(f"\n{'treat_fraction':<16}" + "".join(f"{name:>16}" for name in LEARNER_CLASSES))
    for tf in treat_fractions:
        row = f"{tf:<16}"
        for name in LEARNER_CLASSES:
            s = summary[name][str(tf)]
            row += f"{s['mean']:>10.3f}+/-{s['sd']:<4.2f}"
        print(row)

    plot_imbalance_sweep(summary, treat_fractions, path=RESULTS_DIR / "part2_imbalance_sweep.png")

    result = {"treat_fractions": list(treat_fractions), "n_seeds": n_seeds, "n": n,
              "summary": summary}
    with open(RESULTS_DIR / "part2_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def plot_imbalance_sweep(summary: dict, treat_fractions, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tf_arr = np.array(treat_fractions)
    for name in LEARNER_CLASSES:
        means = np.array([summary[name][str(tf)]["mean"] for tf in treat_fractions])
        sds = np.array([summary[name][str(tf)]["sd"] for tf in treat_fractions])
        ax.errorbar(tf_arr, means, yerr=sds, marker="o", capsize=4, lw=2,
                     color=COLORS[name], label=name)

    ax.set_xlabel("treat_fraction (fraction of customers treated)")
    ax.set_ylabel("PEHE (mean +/- SD across seeds)")
    ax.set_title("PEHE vs. treatment/control imbalance")
    ax.invert_xaxis()
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    real_data_result = run_real_data_demo()
    part1_result = run_part1()
    part2_result = run_part2()

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(
            {"real_data_demo": real_data_result, "part1": part1_result, "part2": part2_result},
            f,
            indent=2,
        )
    print("\nAll results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
