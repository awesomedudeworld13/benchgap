"""Invariants this package exists to guarantee.

Run: `python tests/test_benchgap.py` (or `python -m pytest tests/ -q`).

These are not exhaustive unit tests. Each guards a way a study using this
package could produce a confidently wrong number with no visible error.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchgap as bg                                             # noqa: E402


def test_tss_is_zero_for_any_constant_forecast():
    """The property that makes TSS the honest headline at a low base rate.

    Accuracy cannot do this job: the same useless forecast scores 99.5% on it.
    """
    y = np.array([1] * 5 + [0] * 995)
    for constant in (0.0, 0.25, 0.5, 1.0):
        prob = np.full(len(y), constant)
        for thr in (0.0, 0.3, 0.7, 1.0):
            s = bg.score_at(y, prob, thr)
            assert abs(s.tss) < 1e-12, f"constant {constant} at thr {thr} -> TSS {s.tss}"
    assert bg.score_at(y, np.zeros(len(y)), 0.5).accuracy >= 0.99


def test_threshold_selection_optimises_the_named_objective():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 4000)
    prob = np.clip(y * 0.45 + rng.normal(0.3, 0.2, 4000), 0, 1)

    thr_f1 = bg.select_threshold(y, prob, "f1")
    thr_tss = bg.select_threshold(y, prob, "tss")

    assert bg.score_at(y, prob, thr_f1).f1 >= bg.score_at(y, prob, thr_tss).f1
    assert bg.score_at(y, prob, thr_tss).tss >= bg.score_at(y, prob, thr_f1).tss


def test_grouped_bootstrap_is_wider_than_iid():
    """The headline reason this package exists.

    Twenty correlated observations of one event must not be reported as twenty
    independent ones. If this ever passes trivially, every published interval
    downstream is too narrow.
    """
    rng = np.random.default_rng(3)
    y, prob, groups = [], [], []
    for event in range(40):
        label = event % 2
        correct = (event % 4) != 3      # a quarter of EVENTS are wrong together
        centre = (0.65 if label else 0.35) if correct else (0.35 if label else 0.65)
        for _ in range(20):             # 20 correlated rows per event
            y.append(label)
            prob.append(np.clip(centre + rng.normal(0, 0.02), 0, 1))
            groups.append(f"event{event}")
    y, prob, groups = np.array(y), np.array(prob), np.array(groups)

    lo_g, hi_g = bg.cluster_bootstrap_ci(y, prob, 0.5, groups=groups, n_resamples=300)
    lo_i, hi_i = bg.cluster_bootstrap_ci(y, prob, 0.5, groups=None, n_resamples=300)
    assert (hi_g - lo_g) > (hi_i - lo_i), (
        "grouped interval was not wider than i.i.d.; correlated rows would be "
        "reported as independent observations"
    )


def test_evaluate_cell_rejects_mismatched_inputs():
    """A length mismatch is a silent disaster; fail loudly instead."""
    y = np.array([0, 1, 0, 1])
    for bad in (np.array([0.1, 0.2]), ):
        try:
            bg.evaluate_cell("c", "d", y, bad, {"tss": 0.5}, bootstrap=False)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched prob length was accepted")
    try:
        bg.evaluate_cell("c", "d", y, np.array([0.1, 0.2, 0.3, 0.4]),
                         {"tss": 0.5}, groups=np.array(["a", "b"]), bootstrap=False)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched groups length was accepted")


def test_attribute_skips_missing_cells_but_keeps_total():
    """A study missing one cell still reports the steps it can support."""
    def cell(name, tss_value):
        c = bg.Cell(name=name, description="", n=10, base_rate=0.5)
        c.scores = {"tss": {"tss": tss_value}, "f1": {"tss": tss_value}}
        return c

    steps = [
        ("leakage", "a", "b", "why"),
        ("drift", "b", "c", "why"),
    ]
    full = bg.attribute({"a": cell("a", 0.9), "b": cell("b", 0.7), "c": cell("c", 0.4)}, steps)
    assert full["tss"]["leakage"] == 0.2
    assert abs(full["tss"]["drift"] - 0.3) < 1e-9
    assert abs(full["tss"]["total"] - 0.5) < 1e-9

    # Middle cell absent: both steps are unsupported, but the total spans a->c
    # and is computed from its endpoints rather than by summing steps.
    partial = bg.attribute({"a": cell("a", 0.9), "c": cell("c", 0.4)}, steps)
    assert "leakage" not in partial["tss"] and "drift" not in partial["tss"]
    assert abs(partial["tss"]["total"] - 0.5) < 1e-9


def test_recovery_reports_both_directions():
    """A harmful result is a finding, not a null.

    Testing only for improvement would report a measurably WORSE outcome as
    'not significant' and discard it. That mistake was made once already.
    """
    assert bg.recovery(0.5, 0.7, (0.05, 0.30))["direction"] == "improves"
    assert bg.recovery(0.5, 0.7, (0.05, 0.30))["significant"] is True

    harmed = bg.recovery(0.64, 0.50, (-0.255, -0.009))
    assert harmed["direction"] == "harms"
    assert harmed["significant"] is True
    assert harmed["delta"] < 0

    assert bg.recovery(0.5, 0.52, (-0.10, 0.14))["direction"] == "inconclusive"
    assert bg.recovery(0.5, 0.52, (-0.10, 0.14))["significant"] is False


def test_peak_tss_is_an_upper_bound_on_any_frozen_threshold():
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.1, 2000)
    prob = np.clip(y * 0.4 + rng.normal(0.3, 0.25, 2000), 0, 1)
    peak, _ = bg.peak_tss(y, prob)
    for thr in np.linspace(0, 1, 25):
        assert bg.score_at(y, prob, thr).tss <= peak + 1e-9


def test_brier_skill_is_zero_for_climatology():
    """A constant base-rate forecast must score exactly zero skill."""
    y = np.array([1] * 30 + [0] * 270)
    prob = np.full(len(y), y.mean())
    assert abs(bg.brier_skill(y, prob)) < 1e-12


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1; print(f"  FAIL  {name}: {exc}")
    print(f"\n{'all checks passed' if not failures else f'{failures} FAILED'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
