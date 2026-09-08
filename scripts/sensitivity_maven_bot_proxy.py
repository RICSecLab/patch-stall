#!/usr/bin/env python3
"""Maven bot-contamination proxy via one-day propagation.

Maven Central does not reliably expose SCM URLs, so Maven events cannot be
classified directly by collect/classify_bot_updates.py. This script bounds the
fraction of Maven "manual" propagation events that could be bot-generated and
states the direction of the resulting bias on the cross-ecosystem
manual-propagation comparison.

It:
  1. Reports, per ecosystem, a same-day / within-one-day proxy for automated
     ("bot") updates among manual events, with BOTH denominators made explicit
     (total events and manual events), alongside the measured bot rates for
     npm and PyPI (both among-classified and total denominators).
  2. Computes an extreme upper-bound recomputation of the manual-propagation
     rate: assume EVERY manual event resolving within one day is a bot, remove
     it, and check whether the cross-ecosystem ordering (Maven >= PyPI >> npm)
     survives the worst case.
  3. Re-fits the per-ecosystem propagation Cox PH + TOST after EXCLUDING
     manual events with propagation_days <= 1, to show that the upstream
     health decoupling (and the borderline Maven contributor result) is
     robust to removing suspected bot updates.

Output: data/sensitivity_maven_bot_proxy.json
Methodology mirrors scripts/analyze_cross_ecosystem.py (RQ3) and
scripts/sensitivity_tost_margin.py; helper functions are reused directly.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cross_ecosystem import (  # noqa: E402
    load_base, load_health, load_propagation,
    load_severity_index, map_constraint_abstract,
)
from sensitivity_tost_margin import (  # noqa: E402
    build_propagation_df, fit_and_tost, HEALTH_COVS, MARGINS,
)

ECOS = [("npm", "npm_propagation.csv"),
        ("PyPI", "pypi_propagation.csv"),
        ("Maven", "maven_propagation.csv")]
ONE_DAY = 1.0


def pout(msg):
    print(msg, file=sys.stderr)


def load_measured_bot_rates():
    """Measured Dependabot/Renovate rates for npm and PyPI, both denominators."""
    p = DATA_DIR / "sensitivity_bot_classification.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    out = {}
    for eco_key, eco_name in [("npm", "npm"), ("pypi", "PyPI")]:
        e = d["ecosystems"][eco_key]
        c = e["counts"]
        bots = c["dependabot"] + c["renovate"] + c["other_bot"]
        classified = bots + c["human"]
        out[eco_name] = {
            "sampled_total": e["total"],
            "bots": bots,
            "classified": classified,
            "unknown": c["unknown"],
            "bot_rate_total_denominator": round(bots / e["total"], 4),
            "bot_rate_among_classified": round(bots / classified, 4) if classified else None,
        }
    # Maven: unclassifiable (no SCM URLs)
    m = d["ecosystems"]["maven"]
    out["Maven"] = {
        "sampled_total": m["total"],
        "classified": m["counts"]["human"] + m["counts"]["dependabot"]
        + m["counts"]["renovate"] + m["counts"]["other_bot"],
        "unknown": m["counts"]["unknown"],
        "note": "unclassifiable: Maven Central does not reliably expose SCM URLs",
    }
    return out


def descriptive_bounds(combined):
    """Per-ecosystem manual counts and same-day / <=1-day proxy fractions."""
    rows = {}
    for eco, _ in ECOS:
        sub = combined[combined["ecosystem"] == eco]
        total = len(sub)
        manual = sub[sub["propagation_type"] == "manual"]
        n_manual = len(manual)
        days = pd.to_numeric(manual["propagation_days"], errors="coerce")
        same_day = int((days == 0).sum())
        le1 = int((days <= ONE_DAY).sum())
        rows[eco] = {
            "total_events": total,
            "manual": n_manual,
            "manual_rate_of_total": round(n_manual / total, 4) if total else None,
            "manual_same_day": same_day,
            "manual_le1day": le1,
            "same_day_frac_of_manual": round(same_day / n_manual, 4) if n_manual else None,
            "le1day_frac_of_manual": round(le1 / n_manual, 4) if n_manual else None,
        }
    return rows


def extreme_bound(desc):
    """Worst case: treat every <=1-day manual event as a bot; drop and recompute."""
    out = {}
    order_pairs = []
    for eco in ["npm", "PyPI", "Maven"]:
        d = desc[eco]
        human_manual = d["manual"] - d["manual_le1day"]
        # Denominator: keep total events (removed events are reclassified out of manual)
        rate = human_manual / d["total_events"] if d["total_events"] else None
        out[eco] = {
            "manual_after_removing_le1day": human_manual,
            "manual_rate_after_removing_le1day": round(rate, 4) if rate is not None else None,
        }
        order_pairs.append((eco, rate))
    # Ordering check: Maven >= PyPI >> npm should survive
    r = {e: v for e, v in order_pairs}
    out["ordering"] = {
        "worst_case_rates": {e: round(r[e], 4) for e in r},
        "maven_ge_pypi": r["Maven"] >= r["PyPI"],
        "pypi_gt_npm": r["PyPI"] > r["npm"],
        "ordering_preserved": (r["Maven"] >= r["PyPI"] > r["npm"]),
    }
    return out


def build_combined():
    base = load_base()
    health = load_health()
    sev_index = load_severity_index()
    frames = [load_propagation(name, fname) for name, fname in ECOS]
    combined = pd.concat(frames, ignore_index=True)
    combined["constraint_abstract"] = combined.apply(map_constraint_abstract, axis=1)
    if not sev_index.empty:
        sev_lu = sev_index[["osv_id", "cvss_score"]].drop_duplicates("osv_id")
        combined = combined.merge(sev_lu, on="osv_id", how="left", suffixes=("_raw", ""))
        if "cvss_score_raw" in combined.columns:
            combined = combined.drop(columns=["cvss_score_raw"])
        combined["cvss_score"] = pd.to_numeric(combined["cvss_score"], errors="coerce")
    return combined, base, health


def refit_excluding_le1day(combined, base, health):
    """Re-run per-ecosystem propagation Cox + TOST after dropping <=1-day manual."""
    days = pd.to_numeric(combined["propagation_days"], errors="coerce")
    drop_mask = (combined["propagation_type"] == "manual") & (days <= ONE_DAY)
    kept = combined[~drop_mask].copy()
    pout(f"  Dropped {int(drop_mask.sum())} <=1-day manual events "
         f"({len(combined)} -> {len(kept)})")

    surv = build_propagation_df(kept, base, health)
    covs_a = ["is_permissive", "is_restrictive", "is_runtime",
              "requires_major", "cvss_score"]
    covs_b = covs_a + HEALTH_COVS

    R = {"margins": MARGINS}
    df_pooled = surv[covs_b + ["propagation_days", "prop_event",
                               "ecosystem", "osv_id"]].dropna()
    df_pooled = df_pooled[df_pooled["propagation_days"] > 0]
    R["pooled"] = fit_and_tost(df_pooled, covs_b, "pooled",
                               strata=["ecosystem"], cluster_col="osv_id")

    for eco, _ in ECOS:
        eco_surv = surv[surv["ecosystem"] == eco].copy()
        eco_covs = []
        for cov in covs_b:
            if cov in eco_surv.columns:
                vals = eco_surv[cov].dropna()
                if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                    eco_covs.append(cov)
        df_eco = eco_surv[eco_covs + ["propagation_days", "prop_event",
                                      "osv_id"]].dropna()
        df_eco = df_eco[df_eco["propagation_days"] > 0]
        R[eco] = fit_and_tost(df_eco, eco_covs, eco, strata=None,
                              cluster_col="osv_id")
    return R


def main():
    pout("=" * 70)
    pout("MAVEN BOT-CONTAMINATION PROXY (one-day propagation)")
    pout("=" * 70)

    combined, base, health = build_combined()
    pout(f"  Combined propagation: {len(combined)} events")

    desc = descriptive_bounds(combined)
    measured = load_measured_bot_rates()
    ext = extreme_bound(desc)

    pout("\n--- Descriptive proxy bounds ---")
    for eco in ["npm", "PyPI", "Maven"]:
        d = desc[eco]
        pout(f"  {eco}: manual={d['manual']}/{d['total_events']} "
             f"({d['manual_rate_of_total']:.1%}); "
             f"<=1day manual={d['manual_le1day']} "
             f"({d['le1day_frac_of_manual']:.1%} of manual)")

    pout("\n--- Extreme upper bound (all <=1-day manual are bots) ---")
    for eco in ["npm", "PyPI", "Maven"]:
        pout(f"  {eco}: worst-case manual rate = "
             f"{ext[eco]['manual_rate_after_removing_le1day']:.1%}")
    pout(f"  Ordering Maven>=PyPI>npm preserved: "
         f"{ext['ordering']['ordering_preserved']}")

    pout("\n--- Re-fit per-ecosystem TOST excluding <=1-day manual ---")
    refit = refit_excluding_le1day(combined, base, health)

    R = {
        "description": "Maven bot-contamination proxy (<=1-day propagation) and "
                       "robustness re-fit excluding <=1-day manual events.",
        "freeze_date": "2026-03-07",
        "one_day_threshold": ONE_DAY,
        "descriptive_proxy": desc,
        "measured_bot_rates": measured,
        "extreme_bound_manual_rate": ext,
        "refit_excluding_le1day_manual": refit,
        "baseline_reference": "sensitivity_tost_margin.json",
        "notes": [
            "Maven manual/total = 43855/75342 = 58.2%; same-day manual = 19792 "
            "(45.1% of manual); <=1-day manual = 20356 (46.4% of manual).",
            "The main propagation Cox (analyze_cross_ecosystem.py) already "
            "restricts to propagation_days>0, so same-day events never entered "
            "the survival estimates; the re-fit here additionally removes the "
            "day==1 manual events.",
            "Measured bot rates: npm 8.0% / PyPI 16.2% among classified "
            "(7.1% / 12.4% on the full sampled denominator); Maven "
            "unclassifiable (no SCM URLs).",
            "Direction of bias: if Maven 'manual' is partly bots, the Maven "
            "manual rate is OVERSTATED. In the extreme worst case (every "
            "<=1-day manual event is a bot) Maven's human-driven manual rate "
            "falls to 31.2%, which remains an order of magnitude above npm "
            "(3.6%) but is NO LONGER guaranteed to exceed PyPI (34.5%); the "
            "qualitative contrast with npm and the three-orders-of-magnitude "
            "AUTO-propagation gap (an auto-, not manual-, quantity) are "
            "untouched.",
            "Robustness: health-agnostic bot updates attenuate health hazard "
            "ratios toward 1 (making TOST EASIER to pass), so contamination "
            "cannot manufacture the decoupling. Re-fitting per-ecosystem TOST "
            "after excluding <=1-day manual events leaves the decoupling intact "
            "(Maven log_contributors HR moves 0.886->0.874, still equivalent "
            "only at delta=0.30; npm/PyPI remain equivalent at small margins) - "
            "the borderline Maven contributor result is NOT an artifact of "
            "bot contamination.",
        ],
    }

    out = DATA_DIR / "sensitivity_maven_bot_proxy.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")
    pout("DONE")
    print(json.dumps({k: R[k] for k in
                      ["descriptive_proxy", "extreme_bound_manual_rate"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
