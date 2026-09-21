# benchgap

**Measure how much of a machine-learning benchmark score survives deployment.**

A model scores 0.99 on a held-out test set. What does it score on real data six
months later? The gap between those two numbers is usually reported — if at all —
as a single figure, which names an effect without explaining it. `benchgap`
helps you break it into pieces you can act on, with confidence intervals that
aren't quietly wrong.

```bash
pip install benchgap
```

Depends on `numpy` and nothing else.

---

## The part most people get wrong

Confidence intervals on model evaluations are routinely several times too
narrow, because the rows being resampled aren't independent:

| Domain | The trap | Group by |
|---|---|---|
| Phishing detection | 200 URLs on one hacked website look like 200 observations; they're closer to **one** | website |
| Air-quality forecasting | 17 stations recording the same smog event look like 17; they're **one** weather event | day |
| Solar flare forecasting | 300 snapshots of one sunspot group look like 300; the group is **one** thing | region |

Resample rows independently in any of those and you'll report an interval that
looks convincing and isn't. `benchgap` takes a `groups` argument everywhere it
computes uncertainty, and resamples whole groups.

```python
import benchgap as bg

lo, hi = bg.cluster_bootstrap_ci(y, prob, threshold, groups=website, n_resamples=1000)
```

This isn't optional politeness. It's the difference between an interval that
means something and one that doesn't.

---

## Quick start

### Score a model honestly

```python
import benchgap as bg

# Pick the decision threshold on VALIDATION data, then freeze it.
# Choosing it on the data you're about to report is the most common way an
# evaluation quietly becomes optimistic.
threshold = bg.select_threshold(y_val, prob_val, objective="tss")

scores = bg.score_at(y_test, prob_test, threshold)
print(scores.tss, scores.recall, scores.false_alarm_rate)
```

**TSS** (True Skill Statistic) = recall − false-alarm rate. It is **0 for any
constant forecast**, which makes the zero line a real no-skill baseline.
Accuracy can't do that job: when one outcome is rare, a model that always
predicts the common case scores 98% and has no skill at all.

### Build a chain of comparisons

A "cell" is one model scored on one dataset. Line several up so each step
changes **exactly one thing**, and the total drop becomes attributable instead
of merely observed.

```python
cells = {
    "shuffled":    bg.evaluate_cell("shuffled", "random split — the common protocol",
                                    y1, p1, thresholds, groups=g1),
    "honest":      bg.evaluate_cell("honest", "grouped split — no leakage",
                                    y2, p2, thresholds, groups=g2),
    "operational": bg.evaluate_cell("operational", "live data, model frozen",
                                    y3, p3, thresholds, groups=g3),
}

steps = [
    ("leakage", "shuffled", "honest", "Same data; only the split changes."),
    ("drift",   "honest",   "operational", "Same protocol; later data."),
]

print(bg.attribute(cells, steps)["tss"])
# {'leakage': 0.215, 'drift': -0.077, 'total': 0.138}
```

A step whose endpoints are missing is skipped rather than guessed at, and
`total` is computed from the first and last cell directly — so it stays correct
even when an intermediate cell doesn't exist.

### Ask whether retraining helps

```python
result = bg.recovery(gap_tss, retrained_tss, ci)
# {'delta': -0.136, 'ci95': [-0.255, -0.009],
#  'direction': 'harms', 'significant': True}
```

The interval can exclude zero in **either** direction. A retrained model that's
measurably *worse* is a finding, not a null — testing only for improvement would
report it as "not significant" and throw it away.

---

## What's in it

| | |
|---|---|
| `score_at`, `tss_from_confusion`, `brier`, `brier_skill`, `confusion` | metrics, with TSS as the headline |
| `select_threshold`, `peak_tss` | pick an operating point on validation; `peak_tss` is a hindsight upper bound, never deployable |
| `cluster_bootstrap_ci`, `paired_difference_ci` | uncertainty that respects grouping |
| `Cell`, `evaluate_cell`, `attribute`, `step_descriptions` | chains of comparisons and their decomposition |
| `recovery` | interpret a retraining comparison in either direction |

The interface is **arrays in, dicts out**. It never sees a DataFrame, a model
object, or a file path — the three studies that drove its design disagree about
all of those, and anything wider would have forced a data model onto domains
that don't want one.

## Where it came from

Extracted unchanged from three independent studies that had each arrived at the
same code:

| Study | What it measures |
|---|---|
| [phish-drift](https://github.com/awesomedudeworld13/phish-drift) | phishing-URL datasets, two of which a regex solves better than published models |
| [ozone-drift](https://github.com/awesomedudeworld13/ozone-drift) | Houston smog forecasting — the control case, where honest testing holds up |
| [SolarFlarePredictor](https://github.com/solarflarepredictor-cmd/SolarFlarePredictor) | solar flare forecasting from satellite magnetic-field data |

Those studies make a claim that only works if every number is comparable:
*how far a benchmark score falls in deployment depends on what kind of change
the model faces.* Three copies of the scoring code kept in sync by a hash check
would drift eventually — and when they did, the comparison would silently stop
being a comparison. Hence one package.

## License

MIT.
