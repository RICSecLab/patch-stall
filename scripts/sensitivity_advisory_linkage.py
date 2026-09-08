#!/usr/bin/env python3
"""Advisory-linkage sensitivity: RQ1 patch availability and the Cox model
restricted to repository-linked advisories only, to quantify how much of
the cross-ecosystem patch-rate gap reflects GHSA repository linkage
completeness rather than maintainer behavior.

Output: data/sensitivity_advisory_linkage.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_cross_ecosystem import (
    load_base, load_health, load_patch_dates_merged,
    build_patch_latency_merged,
)

R: dict = {}


def pout(msg):
    print(msg, file=sys.stderr)


def main():
    base = load_base()
    health = load_health()
    patch_merged = load_patch_dates_merged()

    pout(f"Base: {len(base)} advisories")

    # ── Patch availability: all vs linked-only ──
    pout("\n" + "=" * 70)
    pout("PATCH AVAILABILITY: ALL vs LINKED-ONLY")
    pout("=" * 70)

    for label, subset in [("all", base), ("linked_only", base[base["has_repo"]])]:
        pout(f"\n--- {label} (N={len(subset)}) ---")
        rates = {}
        for eco in ["npm", "PyPI", "Maven"]:
            eco_sub = subset[subset["ecosystem"] == eco]
            n = len(eco_sub)
            n_patched = eco_sub["has_patch"].sum()
            rate = n_patched / n if n > 0 else 0
            rates[eco] = {"n": int(n), "patched": int(n_patched),
                          "rate": round(rate, 3)}
            pout(f"  {eco:8s}: {n_patched}/{n} = {rate:.1%}")

        # By severity
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            sev_sub = subset[subset["severity_label"] == sev]
            for eco in ["npm", "PyPI", "Maven"]:
                eco_sev = sev_sub[sev_sub["ecosystem"] == eco]
                n = len(eco_sev)
                rate = eco_sev["has_patch"].mean() if n > 0 else 0
                rates[f"{eco}_{sev}"] = {"n": int(n), "rate": round(rate, 3)}
            pout(f"  {sev:12s}: npm={rates[f'npm_{sev}']['rate']:.1%} "
                 f"PyPI={rates[f'PyPI_{sev}']['rate']:.1%} "
                 f"Maven={rates[f'Maven_{sev}']['rate']:.1%}")

        R[f"patch_availability_{label}"] = rates

    # ── Cox model: all vs linked-only ──
    pout("\n" + "=" * 70)
    pout("COX MODEL: ALL vs LINKED-ONLY")
    pout("=" * 70)

    for label, base_subset in [("all", base),
                                ("linked_only", base[base["has_repo"]])]:
        pout(f"\n--- {label} ---")
        valid = build_patch_latency_merged(base_subset, patch_merged)
        pout(f"  Patch latency: {len(valid)} rows, {valid['patch_event'].sum()} events")

        cox_df = valid.copy()
        cox_df["is_critical"] = (cox_df["severity_label"] == "CRITICAL").astype(int)
        cox_df["is_high"] = (cox_df["severity_label"] == "HIGH").astype(int)
        cox_df["is_low"] = (cox_df["severity_label"] == "LOW").astype(int)

        if not health.empty and "github_repo" in cox_df.columns:
            h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
            h = h.rename(columns={"repo": "github_repo"})
            cox_df = cox_df.merge(h, on="github_repo", how="left")
            cox_df = cox_df.dropna(subset=["log_contributors", "log_stars", "age_years"])

        sev_covs = ["is_critical", "is_high", "is_low"]
        covs = sev_covs + ["log_contributors", "log_stars", "age_years"]
        cols = covs + ["patch_latency_days", "patch_event", "ecosystem"]
        cox_df = cox_df[cols].dropna()
        cox_df = cox_df[cox_df["patch_latency_days"] > 0]

        pout(f"  Cox subset: N={len(cox_df)}, events={cox_df['patch_event'].sum()}")
        for eco in ["npm", "PyPI", "Maven"]:
            eco_sub = cox_df[cox_df["ecosystem"] == eco]
            pout(f"    {eco}: n={len(eco_sub)}, events={int(eco_sub['patch_event'].sum())}")

        if len(cox_df) < 100:
            pout("  Insufficient data")
            continue

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cox_df, duration_col="patch_latency_days", event_col="patch_event",
                strata=["ecosystem"])

        s = cph.summary
        model_results = {
            "n": len(cox_df),
            "events": int(cox_df["patch_event"].sum()),
            "concordance": round(cph.concordance_index_, 3),
            "covariates": {},
        }
        for cov in covs:
            hr = np.exp(s.loc[cov, "coef"])
            p_val = s.loc[cov, "p"]
            lo = np.exp(s.loc[cov, "coef lower 95%"])
            hi = np.exp(s.loc[cov, "coef upper 95%"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
            model_results["covariates"][cov] = {
                "HR": round(hr, 3), "p": round(p_val, 4),
                "CI": [round(lo, 3), round(hi, 3)],
            }

        R[f"cox_{label}"] = model_results

    # ── Log-rank: linked-only ──
    pout("\n--- Log-rank tests (linked-only) ---")
    valid_linked = build_patch_latency_merged(base[base["has_repo"]], patch_merged)
    logrank_results = {}
    for a, b in [("npm", "PyPI"), ("npm", "Maven"), ("PyPI", "Maven")]:
        sa = valid_linked[valid_linked["ecosystem"] == a]
        sb = valid_linked[valid_linked["ecosystem"] == b]
        if len(sa) < 30 or len(sb) < 30:
            continue
        lr = logrank_test(
            sa["patch_latency_days"], sb["patch_latency_days"],
            event_observed_A=sa["patch_event"], event_observed_B=sb["patch_event"]
        )
        pout(f"  {a} vs {b}: chi2={lr.test_statistic:.1f}, p={lr.p_value:.6f}")
        logrank_results[f"{a}_vs_{b}"] = {
            "chi2": round(lr.test_statistic, 1),
            "p": round(lr.p_value, 6),
        }
    R["logrank_linked_only"] = logrank_results

    # ── Selection model: does linkage predict patch speed? ──
    pout("\n--- Selection model: linkage as covariate ---")
    valid_all = build_patch_latency_merged(base, patch_merged)
    sel_df = valid_all.copy()
    sel_df["is_critical"] = (sel_df["severity_label"] == "CRITICAL").astype(int)
    sel_df["is_high"] = (sel_df["severity_label"] == "HIGH").astype(int)
    sel_df["is_low"] = (sel_df["severity_label"] == "LOW").astype(int)
    sel_df["has_repo_int"] = sel_df["has_repo"].astype(int)

    sel_covs = ["is_critical", "is_high", "is_low", "has_repo_int"]
    sel_cols = sel_covs + ["patch_latency_days", "patch_event", "ecosystem"]
    sel_df = sel_df[sel_cols].dropna()
    sel_df = sel_df[sel_df["patch_latency_days"] > 0]

    cph_sel = CoxPHFitter(penalizer=0.01)
    cph_sel.fit(sel_df, duration_col="patch_latency_days", event_col="patch_event",
                strata=["ecosystem"])
    s_sel = cph_sel.summary
    pout(f"  N={len(sel_df)}, events={sel_df['patch_event'].sum()}")
    for cov in sel_covs:
        hr = np.exp(s_sel.loc[cov, "coef"])
        p_val = s_sel.loc[cov, "p"]
        lo = np.exp(s_sel.loc[cov, "coef lower 95%"])
        hi = np.exp(s_sel.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")

    R["selection_model"] = {
        "n": len(sel_df),
        "events": int(sel_df["patch_event"].sum()),
        "concordance": round(cph_sel.concordance_index_, 3),
        "has_repo_HR": round(np.exp(s_sel.loc["has_repo_int", "coef"]), 3),
        "has_repo_p": round(s_sel.loc["has_repo_int", "p"], 4),
        "has_repo_CI": [
            round(np.exp(s_sel.loc["has_repo_int", "coef lower 95%"]), 3),
            round(np.exp(s_sel.loc["has_repo_int", "coef upper 95%"]), 3),
        ],
    }

    out = DATA_DIR / "sensitivity_advisory_linkage.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
