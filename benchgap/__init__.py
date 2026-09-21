"""benchgap — measure how much of a benchmark score survives deployment.

    from benchgap import score_at, select_threshold, cluster_bootstrap_ci
    from benchgap import Cell, evaluate_cell, attribute

See README.md for what the pieces are for and why the grouping key matters.
"""

from .cells import (
    BOOTSTRAP_RESAMPLES,
    Cell,
    attribute,
    evaluate_cell,
    recovery,
    step_descriptions,
)
from .evaluate import (
    Scores,
    brier,
    brier_skill,
    cluster_bootstrap_ci,
    confusion,
    paired_difference_ci,
    peak_tss,
    score_at,
    select_threshold,
    tss_from_confusion,
)

__version__ = "0.1.0"

__all__ = [
    "Scores", "brier", "brier_skill", "cluster_bootstrap_ci", "confusion",
    "paired_difference_ci", "peak_tss", "score_at", "select_threshold",
    "tss_from_confusion", "Cell", "attribute", "evaluate_cell", "recovery",
    "step_descriptions", "BOOTSTRAP_RESAMPLES", "__version__",
]
