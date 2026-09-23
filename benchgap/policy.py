"""The threshold policy: checks that stop a cutoff from being chosen somewhere it can't transfer.

Every frozen-threshold failure across the four studies had the same shape. A
cutoff was chosen on a validation window unlike the data it was then applied
to: ozone's off-season window, solar's solar-maximum year, a phishing window
with no benign rows at all. `select_threshold` cannot see any of that; it just
returns the best cutoff on whatever it is given. These functions look at the
validation data before a cutoff is chosen, and put the cutoff on a calibrated
probability scale so that it means the same thing when the model is refit.

The policy itself, in words, is in THRESHOLDS.md.
"""

from __future__ import annotations

import numpy as np

from .evaluate import brier, peak_tss, score_at, select_threshold


class ValidationError(ValueError):
    """The validation window cannot support choosing a cutoff."""


def check_validation(y_val, *, min_positives: int = 30, min_negatives: int = 30,
                     reference_base_rate: float | None = None,
                     max_base_rate_ratio: float = 2.0) -> dict:
    """Refuse a validation window a cutoff can't honestly be chosen on.

    Hard failures raise `ValidationError`: a class that is missing or too rare
    to locate a cutoff. With one class every cutoff scores the same and
    `select_threshold` silently returns its 0.5 default.

    `reference_base_rate` is the base rate of the period the cutoff will be used
    in (the in-season rate, the current cycle phase). If validation's base rate
    differs from it by more than `max_base_rate_ratio` either way, the window is
    from a different regime; that is reported as `regime_mismatch`, not raised,
    because sometimes nothing better exists. It must then be disclosed.
    """
    y_val = np.asarray(y_val).astype(int)
    pos, neg = int(y_val.sum()), int(len(y_val) - y_val.sum())
    if pos < min_positives or neg < min_negatives:
        raise ValidationError(
            f"validation has {pos} positives and {neg} negatives; need at least "
            f"{min_positives} and {min_negatives} to choose a cutoff")
    rate = pos / len(y_val)
    out = {"positives": pos, "negatives": neg, "base_rate": round(rate, 6),
           "reference_base_rate": reference_base_rate, "regime_mismatch": False}
    if reference_base_rate:
        ratio = max(rate / reference_base_rate, reference_base_rate / rate)
        out["base_rate_ratio"] = round(ratio, 3)
        out["regime_mismatch"] = bool(ratio > max_base_rate_ratio)
    return out


def isotonic_calibrator(prob_val, y_val):
    """Fit a monotone map from raw scores to observed frequencies (pool adjacent violators).

    Returns a function. After calibration a cutoff is a probability, not a
    position on one particular model's score scale, so a refit model calibrated
    the same way can keep it. numpy only, so the package stays dependency-free.
    """
    order = np.argsort(np.asarray(prob_val, dtype=float), kind="mergesort")
    x = np.asarray(prob_val, dtype=float)[order]
    y = np.asarray(y_val, dtype=float)[order]
    # blocks of (mean, weight, right edge); merge while the means decrease
    means, weights, edges = [], [], []
    for xi, yi in zip(x, y):
        means.append(yi); weights.append(1.0); edges.append(xi)
        while len(means) > 1 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            weights[-2] = w
            edges[-2] = edges[-1]
            means.pop(); weights.pop(); edges.pop()
    edges_a, means_a = np.array(edges), np.array(means)

    def calibrate(prob):
        idx = np.searchsorted(edges_a, np.asarray(prob, dtype=float), side="left")
        return means_a[np.clip(idx, 0, len(means_a) - 1)]
    return calibrate


def transfer_report(y, prob, threshold: float, *, flag_at: float = 0.05) -> dict:
    """Frozen-cutoff TSS next to threshold-free peak TSS and Brier, on the same rows.

    A frozen score well below the peak means the cutoff didn't transfer; the
    model may rank just as well. Report this beside every frozen-threshold
    result, so "the model got worse" and "the cutoff moved" can't be confused.
    """
    frozen = score_at(y, prob, threshold).tss
    peak, best = peak_tss(y, prob)
    return {"frozen_tss": round(frozen, 6), "peak_tss": round(peak, 6),
            "peak_threshold": round(best, 6), "shortfall": round(peak - frozen, 6),
            "threshold_did_not_transfer": bool(peak - frozen > flag_at),
            "brier": round(brier(y, prob), 6)}


def choose_threshold(y_val, prob_val, *, reference_base_rate: float | None = None,
                     objective: str = "tss", **check_kwargs) -> dict:
    """The whole policy in one call: check the window, calibrate, then choose.

    Returns the calibrator (apply it to every later score before comparing with
    the cutoff), the cutoff on the calibrated scale, and the validation check.
    """
    check = check_validation(y_val, reference_base_rate=reference_base_rate, **check_kwargs)
    calibrate = isotonic_calibrator(prob_val, y_val)
    thr = select_threshold(np.asarray(y_val), calibrate(prob_val), objective)
    return {"calibrate": calibrate, "threshold": thr, "validation": check}
