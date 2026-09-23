# Threshold policy

Adopted 2026-09-23 for every study that scores with benchgap (phish-drift,
ozone-drift, SolarFlarePredictor). It governs every pre-registration written
after that date. Designs registered before it keep their frozen choices and are
reported as registered.

## Why this exists

The same failure happened four times in one project:

| Where | Validation window | What went wrong |
|---|---|---|
| Solar, cycle 24 (July) | F1 cutoff tuned on the benchmark | F1 point collapsed at the operational base rate |
| Ozone, testing-new | Oct–May, mostly outside ozone season | TSS-best cutoff came out very low; the refit model F warns on 85% of site-days against a 10% exceedance rate |
| Solar, testing-new | 2024, the solar-maximum year | Cutoff 0.69 says "yes" too rarely in quieter years (R2 failed) |
| Phishing, testing-new P2 | Last 14 days of a month | Benign rows are dated inside mid-month crawl windows, so validation held zero benign rows and every URL was called phishing |

It also produced a published result that turned out to be a cutoff artifact:
ozone's "retraining hurt" (−0.136 at frozen cutoffs, but 0.678 against 0.677
at the best cutoff on the same rows).

## The rules

1. **Validation must hold both classes in usable numbers.** At least 30
   positives and 30 negatives, checked in code (`check_validation`) before any
   cutoff is chosen. A window that fails is a design error; fix the window, do
   not proceed.

2. **Validation must come from the regime the cutoff will be used in.** State
   the deployment regime's base rate (in-season, current cycle phase, current
   month) and check validation against it. A ratio above 2 either way is a
   mismatch. If nothing better exists, it must be disclosed next to every
   frozen-cutoff result.

3. **Calibrate, then choose.** Scores are mapped to probabilities on validation
   (`isotonic_calibrator`) and the cutoff is chosen on the calibrated scale
   (`choose_threshold`). A cutoff is then a probability rather than a point on
   one model's private score scale.

4. **Never move a cutoff between models.** A refit model (ozone F, solar F)
   gets its own calibration on validation data it did not train on. Carrying a
   raw cutoff from a dev model to a refit model is not allowed.

5. **Report three numbers, not one.** Every frozen-cutoff TSS is shown next to
   the threshold-free peak TSS and the Brier score on the same rows
   (`transfer_report`). If frozen is more than 0.05 below peak, the result is
   labelled a cutoff-transfer result, and no claim about the model's skill may
   rest on it.

6. **Baselines get the same treatment as models.** A persistence or regex
   baseline gets a cutoff chosen on the same validation window by the same
   rule. Ozone's model looked far ahead of persistence only because persistence
   was scored at a fixed cutoff while the model's was tuned.

## What it does not fix

Calibration assumes validation looks like deployment. When the base rate
shifts sharply (seasons, solar cycle), calibrated probabilities are still off
and rule 2 is the real protection. Brier scores do not depend on any cutoff,
which is why rule 5 makes them mandatory.
