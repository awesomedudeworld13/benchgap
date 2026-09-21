"""Evaluation cells and gap attribution.

A "cell" is one model scored on one dataset at frozen operating points. A study
lines several up so that each step from one to the next changes **exactly one
thing**, which turns "the score fell by X" into "the score fell by X, and here
is which decision cost what".

Deliberately array-in, dict-out. This module never sees a DataFrame, a model
object, or a file path, because the three studies that drove its design
disagree about all of those: one scores URL strings, one scores rows of
weather readings, one scores solar magnetogram windows. What they agree on is
the shape of the question -- labels, predicted probabilities, and a grouping
key for uncertainty -- so that is the entire interface. Anything wider would
have forced a data model onto domains that do not want one.

The grouping key is the part people get wrong. Observations in these problems
are rarely independent:

    phishing   many URLs sit on one hacked website  -> group by website
    ozone      many stations record one smog event  -> group by day
    solar      many snapshots of one sunspot group  -> group by region

Resampling rows independently in any of those treats one event as many, and
reports a confidence interval several times narrower than the truth. Passing
``groups`` is not optional politeness; it is the difference between an interval
that means something and one that does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import evaluate

BOOTSTRAP_RESAMPLES = 1000


@dataclass
class Cell:
    """One model scored on one dataset at frozen operating points."""

    name: str
    description: str
    n: int
    base_rate: float
    scores: dict = field(default_factory=dict)
    peak_tss: float = float("nan")
    peak_threshold: float = float("nan")
    extra: dict = field(default_factory=dict)

    def tss(self, operating_point: str = "tss") -> float:
        return self.scores[operating_point]["tss"]

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "description": self.description,
            "n": self.n,
            "base_rate": round(self.base_rate, 6),
            "peak_tss": round(self.peak_tss, 6),
            "peak_threshold": round(self.peak_threshold, 6),
            "operating_points": self.scores,
        }
        out.update(self.extra)
        return out


def evaluate_cell(name: str, description: str, y, prob, thresholds: dict,
                  groups=None, n_resamples: int = BOOTSTRAP_RESAMPLES,
                  bootstrap: bool = True, **extra) -> Cell:
    """Score ``prob`` against ``y`` at each frozen threshold.

    ``thresholds`` maps an operating-point name to its decision threshold.
    Those thresholds must already be frozen -- chosen on validation data, not on
    the ``y`` being passed here. Choosing a threshold on the data you are about
    to report is the most common way an evaluation quietly becomes optimistic,
    and this function cannot detect it for you.

    ``peak_tss`` is also reported: the best TSS reachable in hindsight by
    picking the threshold after seeing the answers. It is never a deployable
    number, only an upper bound, which is why every table that shows it shows a
    frozen-threshold score beside it.

    Any additional keyword arguments are passed through into the result dict, so
    a caller can attach domain-specific context (a day count, a region count)
    without this module needing to know what those are.
    """
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    if len(y) != len(prob):
        raise ValueError(f"y has {len(y)} rows but prob has {len(prob)}")
    if groups is not None and len(groups) != len(y):
        raise ValueError(f"groups has {len(groups)} rows but y has {len(y)}")

    peak, peak_thr = evaluate.peak_tss(y, prob)
    cell = Cell(name=name, description=description, n=int(len(y)),
                base_rate=float(y.mean()) if len(y) else float("nan"),
                peak_tss=peak, peak_threshold=peak_thr, extra=extra)

    for point, thr in thresholds.items():
        s = evaluate.score_at(y, prob, thr).to_dict()
        if bootstrap:
            lo, hi = evaluate.cluster_bootstrap_ci(
                y, prob, thr,
                groups=None if groups is None else np.asarray(groups),
                n_resamples=n_resamples)
            s["tss_ci95"] = [round(lo, 6), round(hi, 6)]
        cell.scores[point] = s
    return cell


def attribute(cells: dict, steps, operating_points=("f1", "tss")) -> dict:
    """Decompose the total drop across an ordered chain of cells.

    ``steps`` is a sequence of ``(name, from_cell, to_cell, meaning)``. A step
    whose endpoints are not both present is skipped rather than guessed at, so a
    study missing a cell still reports the steps it can support.

    ``total`` spans the first step's origin to the last step's destination, and
    is computed directly from those two endpoints rather than by summing the
    steps -- so it stays correct even when an intermediate step was skipped.
    """
    steps = list(steps)
    out: dict = {}
    for point in operating_points:
        rows = {}
        for name, a, b, _meaning in steps:
            if a in cells and b in cells:
                rows[name] = round(cells[a].tss(point) - cells[b].tss(point), 6)
        if steps:
            first, last = steps[0][1], steps[-1][2]
            if first in cells and last in cells:
                rows["total"] = round(cells[first].tss(point) - cells[last].tss(point), 6)
        out[point] = rows
    return out


def step_descriptions(steps) -> list[dict]:
    """The prose half of ``steps``, for embedding in a results document."""
    return [{"step": n, "from": a, "to": b, "meaning": m} for n, a, b, m in steps]


def recovery(gap_cell_tss: float, recovered_cell_tss: float,
             ci: tuple[float, float]) -> dict:
    """Interpret a retraining comparison, in whichever direction it went.

    Answers "does training on operational data close the gap?". The interval can
    exclude zero on *either* side, and the negative case is a genuine finding
    rather than a null: it means retraining measurably hurt. Testing only for
    improvement would report that as "not significant" and discard it -- which
    is a mistake this function exists to prevent, having been made once.
    """
    lo, hi = ci
    delta = recovered_cell_tss - gap_cell_tss
    if lo > 0:
        direction, significant = "improves", True
    elif hi < 0:
        direction, significant = "harms", True
    else:
        direction, significant = "inconclusive", False
    return {
        "delta": round(delta, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "direction": direction,
        "significant": significant,
    }
