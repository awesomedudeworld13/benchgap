"""Every candidate headline quantity, for every domain, in one table.

AI DISCLOSURE: this script was written by Claude (Anthropic's Claude Code),
2026-09-23, at the team's request. It only collects numbers the four studies
already published and does arithmetic on them. It makes no claims; choosing
which quantity to lead with, and what it means, is the researchers' job.

    python studies/cross_domain.py            # -> studies/cross_domain.md, .json

Reads each study's results from local clones with `git show <ref>:<path>`,
so it works for private repos and never depends on uncommitted files. Clones
are expected next to this repo (override with --root). Branch results are
labelled "branch (not merged)".

Quantities, all at each study's frozen TSS operating point unless marked peak:

  RQ1, per domain (question 1: benchmark score vs live score)
    shuffled      benchmark TSS with a random split (the common protocol)
    honest        benchmark TSS with a grouped or chronological split
    live          TSS on live / operational data, model frozen
    *_peak        the same with the best possible cutoff for those rows
                  (a hindsight upper bound; removes cutoff transfer)
    gap           honest - live             (frozen and peak)
    inflation     shuffled - honest
    retention     live / honest             (share of honest skill that survives)

  RQ2, where a live-trained model exists (question 2: does live training help)
    live_bench_model   benchmark-trained model on the live test
    live_live_model    live-trained model on the same live test
    model_moved        live_live_model - live_bench_model  (same rows, so this
                       is the part of any gap change that is real skill)
    Where a study also reports each model's own benchmark score, the other part
    of a gap change, benchmark_moved, is reported too: gap change =
    model_moved + benchmark_moved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def show(root: Path, repo: str, ref: str, path: str) -> dict | None:
    try:
        out = subprocess.run(["git", "-C", str(root / repo), "show", f"{ref}:{path}"],
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return json.loads(out)


def op(cell: dict) -> tuple[float, list | None]:
    s = cell["operating_points"]["tss"]
    return s["tss"], s.get("tss_ci95")


def rq1_row(domain: str, source: str, cells: dict, shuffled: str, honest: str, live: str) -> dict:
    sh, _ = op(cells[shuffled])
    ho, ho_ci = op(cells[honest])
    li, li_ci = op(cells[live])
    hp, lp = cells[honest]["peak_tss"], cells[live]["peak_tss"]
    return {"domain": domain, "source": source,
            "shuffled": sh, "honest": ho, "honest_ci": ho_ci, "live": li, "live_ci": li_ci,
            "honest_peak": hp, "live_peak": lp,
            "gap": ho - li, "gap_peak": hp - lp, "inflation": sh - ho,
            "retention": li / ho if ho else None,
            "base_rate_honest": cells[honest]["base_rate"], "base_rate_live": cells[live]["base_rate"]}


def collect(root: Path) -> dict:
    rq1, rq2, missing = [], [], []

    for tier in ("core", "extended"):
        r = show(root, "ozone-drift", "origin/main", f"results/{tier}.json")
        if r is None:
            missing.append(f"ozone {tier}"); continue
        rq1.append(rq1_row(f"ozone ({tier})", "main", r["cells"], "cell1_benchmark_random",
                           "cell2_benchmark_chronological", "cell3_operational"))
        lt = r["live_training"]["cells"]
        b, d = lt["B_benchmark_trained_on_operational"], lt["D_operational_trained_on_operational"]
        a, c = lt["A_benchmark_trained_on_benchmark"], lt["C_operational_trained_on_benchmark"]
        rq2.append({"domain": f"ozone ({tier})", "source": "main",
                    "live_bench_model": op(b)[0], "live_live_model": op(d)[0],
                    "model_moved": op(d)[0] - op(b)[0],
                    "model_moved_ci": r["live_training"]["recovery_ci95"],
                    "model_moved_peak": d["peak_tss"] - b["peak_tss"],
                    "note": "live-trained model's own held-out test is the live test itself (cell D), "
                            "so benchmark_moved is not separately defined; cells A and C: "
                            f"{op(a)[0]:.3f}, {op(c)[0]:.3f}"})

    for name, fname in (("solar flares", "protocol_cells.json"), ("geomagnetic storms", "storm_protocol_cells.json")):
        r = show(root, "SolarFlareProject", "origin/main", f"solarflare/experiments/results/{fname}")
        if r is None:
            missing.append(name); continue
        ks = list(r["cells"])
        rq1.append(rq1_row(name, "main", r["cells"], ks[0], ks[1], ks[2]))

    for corpus in ("phiusiil", "kaitholikkal", "faizann", "hannousse"):
        r = show(root, "phish-drift", "origin/main", f"results/{corpus}.json")
        if r is None:
            missing.append(f"phishing {corpus}"); continue
        rq1.append(rq1_row(f"phishing ({corpus})", "main", r["cells"], "cell1_benchmark_random",
                           "cell2_benchmark_disjoint", "cell4_live_realistic_benign"))

    s = show(root, "SolarFlareProject", "origin/testing-new", "testing_new/rq2_gap_cycle25.json")
    if s:
        dep, liv = s["models"]["deployed"], s["models"]["live_D"]
        rq2.append({"domain": "solar flares, cycle 25", "source": "branch (not merged), exploratory",
                    "live_bench_model": dep["frozen"]["operational_tss"],
                    "live_live_model": liv["frozen"]["operational_tss"],
                    "model_moved": s["decomposition_frozen"]["operational_skill_gain"]["value"],
                    "model_moved_ci": s["decomposition_frozen"]["operational_skill_gain"]["ci"],
                    "model_moved_peak": s["decomposition_peak"]["operational_skill_gain"]["value"],
                    "model_moved_peak_ci": s["decomposition_peak"]["operational_skill_gain"]["ci"],
                    "benchmark_moved": s["decomposition_frozen"]["benchmark_drop"],
                    "gap_change": s["gap_closed_by_live_training_frozen"]["value"],
                    "note": "each model scored on its own benchmark; the two benchmarks are different test sets"})
    else:
        missing.append("solar cycle-25 RQ2 (branch)")

    for fname, label in (("testing_new/retro_results.json", "Tranco top-50k benign (pre-registered)"),
                         ("testing_new/retro_random_results.json", "random-crawl benign (exploratory)")):
        r = show(root, "phish-drift", "origin/testing-new", fname)
        if r is None:
            missing.append(f"phishing RQ2 {label}"); continue
        p = r.get("P1_realistic_rf") or r["P1_crawl_random_rf"]
        dlive = p["D_live_on_live"]["tss"]
        for corpus, v in p["benchmarks"].items():
            rq2.append({"domain": f"phishing ({corpus}), {label}", "source": "branch (not merged)",
                        "live_bench_model": v["B_bench_on_live"]["tss"], "live_live_model": dlive,
                        "model_moved": v["recovery_D_minus_B"]["value"],
                        "model_moved_ci": v["recovery_D_minus_B"]["ci95"],
                        "note": "live-trained model's own benchmark is not defined separately"})
    return {"rq1": rq1, "rq2": rq2, "missing": missing}


def f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) and x < 0 else f"{x:.{d}f}"


def ci(c):
    return "" if not c else f" ({c[0]:.3f} to {c[1]:.3f})"


def markdown(doc: dict) -> str:
    out = ["# Cross-domain numbers",
           "",
           "Generated by `studies/cross_domain.py` (AI-written; see its header). Numbers only:",
           "which quantity to lead with, and what the pattern means, is left to the researchers.",
           "",
           "## Question 1: benchmark score vs live score (TSS)",
           "",
           "| Domain | Shuffled | Honest (95% CI) | Live (95% CI) | Gap | Gap, peak | Inflation | Retention | Base rate honest / live |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in doc["rq1"]:
        out.append(f"| {r['domain']} | {f(r['shuffled'])} | {f(r['honest'])}{ci(r['honest_ci'])} | "
                   f"{f(r['live'])}{ci(r['live_ci'])} | {f(r['gap'])} | {f(r['gap_peak'])} | "
                   f"{f(r['inflation'])} | {f(r['retention'], 2)} | "
                   f"{r['base_rate_honest']:.3f} / {r['base_rate_live']:.3f} |")
    out += ["",
            "- **Honest** = grouped or chronological split; **Live** = operational data, model frozen.",
            "- **Gap** = honest − live. **Gap, peak** uses each cell's best possible cutoff, so it excludes cutoff transfer.",
            "- **Inflation** = shuffled − honest. **Retention** = live ÷ honest.",
            "",
            "## Question 2: does a live-trained model do better on the same live test?",
            "",
            "| Domain | Source | Benchmark-trained on live | Live-trained on live | Model moved (95% CI) | Model moved, peak | Benchmark moved |",
            "|---|---|---|---|---|---|---|"]
    for r in doc["rq2"]:
        out.append(f"| {r['domain']} | {r['source']} | {f(r['live_bench_model'])} | {f(r['live_live_model'])} | "
                   f"{f(r['model_moved'])}{ci(r.get('model_moved_ci'))} | "
                   f"{f(r.get('model_moved_peak'))}{ci(r.get('model_moved_peak_ci'))} | "
                   f"{f(r.get('benchmark_moved'))} |")
    out += ["",
            "- **Model moved** compares two models on the same live rows, so any change in a gap that it does not account for came from the benchmark side (**Benchmark moved**).",
            "- Geomagnetic storms have no live-trained model yet.",
            ""]
    notes: dict[str, list[str]] = {}
    for r in doc["rq2"]:
        if r.get("note"):
            notes.setdefault(r["note"], []).append(r["domain"])
    if notes:
        out += ["### Notes", ""] + [f"- {', '.join(ds) if len(ds) < 3 else ds[0].split(' (')[0] + f' (all {len(ds)} rows)'}: {n}" for n, ds in notes.items()] + [""]
    if doc["missing"]:
        out += ["### Not found", "", *[f"- {m}" for m in doc["missing"]], ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent.parent,
                    help="folder holding the phish-drift, ozone-drift and SolarFlareProject clones")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        for repo in ("phish-drift", "ozone-drift", "SolarFlareProject"):
            subprocess.run(["git", "-C", str(a.root / repo), "fetch", "-q", "origin"], check=False)
    doc = collect(a.root)
    (HERE / "cross_domain.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (HERE / "cross_domain.md").write_text(markdown(doc), encoding="utf-8")
    print(f"wrote {HERE / 'cross_domain.md'} ({len(doc['rq1'])} RQ1 rows, {len(doc['rq2'])} RQ2 rows)")


if __name__ == "__main__":
    main()
