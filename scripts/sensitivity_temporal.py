#!/usr/bin/env python3
"""Temporal and stratification sensitivity.

  1. Year-stratified Cox model: health metrics are measured at the data
     freeze, so this checks whether the estimates are stable across
     advisory publication years
  2. Unstratified propagation model: recovers the between-ecosystem
     constraint effect that ecosystem stratification absorbs into the
     baseline hazard

Output: data/sensitivity_temporal.json
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
from lifelines import CoxPHFitter

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

    # ================================================================
    # W3: Year-stratified Cox model for patch speed
    # ================================================================
    pout("=" * 70)
    pout("W3: YEAR-STRATIFIED COX MODEL (PATCH SPEED)")
    pout("=" * 70)

    valid = build_patch_latency_merged(base, patch_merged)
    cox_df = valid.copy()
    cox_df["is_critical"] = (cox_df["severity_label"] == "CRITICAL").astype(int)
    cox_df["is_high"] = (cox_df["severity_label"] == "HIGH").astype(int)
    cox_df["is_low"] = (cox_df["severity_label"] == "LOW").astype(int)

    if not health.empty and "github_repo" in cox_df.columns:
        h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        cox_df = cox_df.merge(h, on="github_repo", how="left")
        cox_df = cox_df.dropna(subset=["log_contributors", "log_stars", "age_years"])

    covs = ["is_critical", "is_high", "is_low",
            "log_contributors", "log_stars", "age_years"]
    cols = covs + ["patch_latency_days", "patch_event", "ecosystem"]
    cox_df = cox_df[cols + ["published_at"]].dropna()
    cox_df = cox_df[cox_df["patch_latency_days"] > 0]

    # Parse year
    cox_df["pub_year"] = pd.to_datetime(cox_df["published_at"]).dt.year

    # Define bins
    bins = [
        ("≤2020", cox_df["pub_year"] <= 2020),
        ("2021-2022", cox_df["pub_year"].between(2021, 2022)),
        ("2023-2024", cox_df["pub_year"].between(2023, 2024)),
        ("2025-2026", cox_df["pub_year"] >= 2025),
    ]

    year_results = {}
    for label, mask in bins:
        subset = cox_df[mask].copy()
        pout(f"\n--- {label} (N={len(subset)}, events={int(subset['patch_event'].sum())}) ---")

        if len(subset) < 100 or subset["patch_event"].sum() < 50:
            pout("  Insufficient data, skipping")
            year_results[label] = {"n": len(subset), "skip": True}
            continue

        fit_cols = covs + ["patch_latency_days", "patch_event", "ecosystem"]
        fit_df = subset[fit_cols]

        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(fit_df, duration_col="patch_latency_days",
                    event_col="patch_event", strata=["ecosystem"])
        except Exception as e:
            pout(f"  Failed: {e}")
            year_results[label] = {"n": len(subset), "error": str(e)}
            continue

        s = cph.summary
        model_res = {
            "n": len(fit_df),
            "events": int(fit_df["patch_event"].sum()),
            "concordance": round(cph.concordance_index_, 3),
            "covariates": {},
        }
        for cov in covs:
            hr = np.exp(s.loc[cov, "coef"])
            p_val = s.loc[cov, "p"]
            lo = np.exp(s.loc[cov, "coef lower 95%"])
            hi = np.exp(s.loc[cov, "coef upper 95%"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            pout(f"  {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
            model_res["covariates"][cov] = {
                "HR": round(hr, 3), "p": round(p_val, 4),
                "CI": [round(lo, 3), round(hi, 3)],
            }
        year_results[label] = model_res

    R["w3_year_stratified"] = year_results

    # Summary: HR stability across bins
    pout("\n--- HR stability across year bins ---")
    for cov in ["log_contributors", "log_stars", "age_years"]:
        hrs = []
        for label, res in year_results.items():
            if "covariates" in res and cov in res["covariates"]:
                hrs.append((label, res["covariates"][cov]["HR"]))
        if hrs:
            pout(f"  {cov}: " + ", ".join(f"{l}={h:.3f}" for l, h in hrs))

    # ================================================================
    # W6: Unstratified propagation model
    # ================================================================
    pout("\n" + "=" * 70)
    pout("W6: UNSTRATIFIED PROPAGATION MODEL")
    pout("=" * 70)

    # Load propagation data
    prop_files = {
        "npm": DATA_DIR / "npm_propagation.csv",
        "PyPI": DATA_DIR / "pypi_propagation.csv",
        "Maven": DATA_DIR / "maven_propagation.csv",
    }
    frames = []
    for eco, path in prop_files.items():
        if path.exists():
            df = pd.read_csv(path)
            df["ecosystem"] = eco
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    pout(f"Propagation events: {len(combined)}")

    # Map constraint types
    combined["constraint_abstract"] = combined["constraint_type"].map({
        "caret": "permissive", "tilde": "restrictive", "compatible": "restrictive",
        "exact": "pinned", "pinned": "pinned", "star": "wildcard",
        "lower_bound": "permissive", "range": "restrictive",
        "hard": "permissive", "soft": "pinned",
    }).fillna("other")

    combined["is_permissive"] = (combined["constraint_abstract"] == "permissive").astype(int)
    combined["is_restrictive"] = (combined["constraint_abstract"] == "restrictive").astype(int)
    combined["requires_major"] = combined.get("requires_major_upgrade",
                                              pd.Series(0, index=combined.index)).fillna(0).astype(int)

    # Propagation event
    combined["prop_event"] = (combined["propagation_type"] != "censored").astype(int)
    combined["prop_days"] = combined["propagation_days"].fillna(0).clip(lower=0.5)
    combined.loc[combined["propagation_type"] == "auto", "prop_days"] = 0.5

    # Merge health
    base_dedup = base.drop_duplicates(subset=["ghsa_id", "ecosystem", "package_name"])
    combined = combined.merge(
        base_dedup[["ghsa_id", "ecosystem", "package_name", "github_repo", "cvss_score"]],
        left_on=["osv_id", "ecosystem", "upstream_package"],
        right_on=["ghsa_id", "ecosystem", "package_name"],
        how="left",
    )
    if not health.empty:
        h = health[["repo", "log_contributors", "log_stars"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        combined = combined.merge(h, on="github_repo", how="left", suffixes=("", "_h"))

    # Ecosystem dummies
    combined["is_npm"] = (combined["ecosystem"] == "npm").astype(int)
    combined["is_pypi"] = (combined["ecosystem"] == "PyPI").astype(int)

    # --- Model A: Stratified (baseline) ---
    covs_a = ["is_permissive", "is_restrictive", "requires_major", "cvss_score",
              "log_contributors", "log_stars"]
    fit_a = combined[covs_a + ["prop_days", "prop_event", "ecosystem"]].dropna()
    fit_a = fit_a[fit_a["prop_days"] > 0]
    pout(f"\nStratified model: N={len(fit_a)}")

    cph_strat = CoxPHFitter(penalizer=0.01)
    cph_strat.fit(fit_a, duration_col="prop_days", event_col="prop_event",
                  strata=["ecosystem"])
    pout(f"  Concordance (stratified): {cph_strat.concordance_index_:.3f}")

    # --- Model B: Unstratified with ecosystem dummies ---
    covs_b = ["is_npm", "is_pypi", "is_permissive", "is_restrictive",
              "requires_major", "cvss_score", "log_contributors", "log_stars"]
    fit_b = combined[covs_b + ["prop_days", "prop_event"]].dropna()
    fit_b = fit_b[fit_b["prop_days"] > 0]
    pout(f"Unstratified model: N={len(fit_b)}")

    cph_unstrat = CoxPHFitter(penalizer=0.01)
    cph_unstrat.fit(fit_b, duration_col="prop_days", event_col="prop_event")
    pout(f"  Concordance (unstratified): {cph_unstrat.concordance_index_:.3f}")

    s = cph_unstrat.summary
    unstrat_results = {
        "n": len(fit_b),
        "events": int(fit_b["prop_event"].sum()),
        "concordance_stratified": round(cph_strat.concordance_index_, 3),
        "concordance_unstratified": round(cph_unstrat.concordance_index_, 3),
        "covariates": {},
    }
    for cov in covs_b:
        hr = np.exp(s.loc[cov, "coef"])
        p_val = s.loc[cov, "p"]
        lo = np.exp(s.loc[cov, "coef lower 95%"])
        hi = np.exp(s.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"  {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        unstrat_results["covariates"][cov] = {
            "HR": round(hr, 3), "p": round(p_val, 4),
            "CI": [round(lo, 3), round(hi, 3)],
        }

    R["w6_unstratified"] = unstrat_results

    out = DATA_DIR / "sensitivity_temporal.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
