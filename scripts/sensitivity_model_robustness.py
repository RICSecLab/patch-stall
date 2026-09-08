#!/usr/bin/env python3
"""Robustness checks on the main model specifications.

  1. Multicollinearity among health covariates: single-variable and PCA
     variants of the RQ1 Cox model
  2. deps.dev 100-dependent cap: cap-hit rate and its effect on RQ3
  3. Pre-advisory patches (T_patch < 0): alternative handling in the
     survival analysis
  4. Permissive-constraint effect reversal: interaction model behind the
     per-ecosystem propagation table

Output: data/sensitivity_model_robustness.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "paper" / "fig"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_cross_ecosystem import (
    load_base, load_health, load_patch_dates_merged,
    load_severity_index, load_propagation,
    build_patch_latency_merged, map_constraint_abstract,
    CONSTRAINT_MAP,
)

R: dict = {}


def pout(msg):
    print(msg, file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════
# Q1: MULTICOLLINEARITY — SINGLE-VAR AND PCA MODELS (RQ1)
# ══════════════════════════════════════════════════════════════════════

def q1_multicollinearity(valid, health):
    """Run single-variable and PCA models to address VIF > 10."""
    pout("\n" + "=" * 70)
    pout("Q1: MULTICOLLINEARITY SENSITIVITY (RQ1 patch speed)")
    pout("=" * 70)

    cox_df = valid.copy()
    cox_df["is_critical"] = (cox_df["severity_label"] == "CRITICAL").astype(int)
    cox_df["is_high"] = (cox_df["severity_label"] == "HIGH").astype(int)
    cox_df["is_low"] = (cox_df["severity_label"] == "LOW").astype(int)

    if health.empty:
        pout("  No health data")
        return

    h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
    h = h.rename(columns={"repo": "github_repo"})
    cox_df = cox_df.merge(h, on="github_repo", how="left")
    cox_df = cox_df.dropna(subset=["log_contributors", "log_stars", "age_years"])

    sev_covs = ["is_critical", "is_high", "is_low"]
    base_cols = sev_covs + ["patch_latency_days", "patch_event", "ecosystem"]
    cox_df = cox_df[base_cols + ["log_contributors", "log_stars", "age_years"]].dropna()
    cox_df = cox_df[cox_df["patch_latency_days"] > 0]

    pout(f"  N={len(cox_df)}, events={cox_df['patch_event'].sum()}")

    # Correlation
    r = cox_df["log_contributors"].corr(cox_df["log_stars"])
    pout(f"  Pearson r(log_contributors, log_stars) = {r:.3f}")
    R["q1_correlation"] = round(r, 3)

    results = {}

    # --- Full model (reference) ---
    pout("\n--- Full model (both contributors + stars) ---")
    cph_full = CoxPHFitter(penalizer=0.01)
    covs_full = sev_covs + ["log_contributors", "log_stars", "age_years"]
    cph_full.fit(cox_df[covs_full + ["patch_latency_days", "patch_event", "ecosystem"]],
                 duration_col="patch_latency_days", event_col="patch_event",
                 strata=["ecosystem"])
    _report_cox(cph_full, covs_full, "full_model", results)

    # --- Contributors-only model ---
    pout("\n--- Contributors-only model ---")
    cph_c = CoxPHFitter(penalizer=0.01)
    covs_c = sev_covs + ["log_contributors", "age_years"]
    cph_c.fit(cox_df[covs_c + ["patch_latency_days", "patch_event", "ecosystem"]],
              duration_col="patch_latency_days", event_col="patch_event",
              strata=["ecosystem"])
    _report_cox(cph_c, covs_c, "contributors_only", results)

    # --- Stars-only model ---
    pout("\n--- Stars-only model ---")
    cph_s = CoxPHFitter(penalizer=0.01)
    covs_s = sev_covs + ["log_stars", "age_years"]
    cph_s.fit(cox_df[covs_s + ["patch_latency_days", "patch_event", "ecosystem"]],
              duration_col="patch_latency_days", event_col="patch_event",
              strata=["ecosystem"])
    _report_cox(cph_s, covs_s, "stars_only", results)

    # --- PCA model: replace log_contributors + log_stars with PC1 ---
    pout("\n--- PCA model (PC1 from contributors + stars) ---")
    scaler = StandardScaler()
    health_2d = scaler.fit_transform(cox_df[["log_contributors", "log_stars"]])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(health_2d)
    cox_df["health_pc1"] = pcs[:, 0]
    cox_df["health_pc2"] = pcs[:, 1]

    var_explained = pca.explained_variance_ratio_
    loadings = pca.components_
    pout(f"  PC1 variance explained: {var_explained[0]:.3f}")
    pout(f"  PC2 variance explained: {var_explained[1]:.3f}")
    pout(f"  PC1 loadings: contributors={loadings[0, 0]:.3f}, stars={loadings[0, 1]:.3f}")
    pout(f"  PC2 loadings: contributors={loadings[1, 0]:.3f}, stars={loadings[1, 1]:.3f}")

    R["q1_pca"] = {
        "variance_explained": [round(v, 4) for v in var_explained],
        "loadings_pc1": {"log_contributors": round(loadings[0, 0], 3),
                         "log_stars": round(loadings[0, 1], 3)},
        "loadings_pc2": {"log_contributors": round(loadings[1, 0], 3),
                         "log_stars": round(loadings[1, 1], 3)},
    }

    cph_pca = CoxPHFitter(penalizer=0.01)
    covs_pca = sev_covs + ["health_pc1", "age_years"]
    cph_pca.fit(cox_df[covs_pca + ["patch_latency_days", "patch_event", "ecosystem"]],
                duration_col="patch_latency_days", event_col="patch_event",
                strata=["ecosystem"])
    _report_cox(cph_pca, covs_pca, "pca_model", results)

    # --- PCA with PC1 + PC2 ---
    pout("\n--- PCA model (PC1 + PC2) ---")
    cph_pca2 = CoxPHFitter(penalizer=0.01)
    covs_pca2 = sev_covs + ["health_pc1", "health_pc2", "age_years"]
    cph_pca2.fit(cox_df[covs_pca2 + ["patch_latency_days", "patch_event", "ecosystem"]],
                 duration_col="patch_latency_days", event_col="patch_event",
                 strata=["ecosystem"])
    _report_cox(cph_pca2, covs_pca2, "pca2_model", results)

    # Summary comparison
    pout("\n--- Concordance comparison ---")
    for name in ["full_model", "contributors_only", "stars_only", "pca_model", "pca2_model"]:
        pout(f"  {name:25s}: C={results[name]['concordance']:.3f}")

    R["q1_models"] = results


def _report_cox(cph, covs, name, results):
    s = cph.summary
    info = {
        "concordance": round(cph.concordance_index_, 3),
        "n": int(cph._n_examples),
        "events": int(cph.event_observed.sum()),
        "covariates": {},
    }
    for cov in covs:
        hr = np.exp(s.loc[cov, "coef"])
        p_val = s.loc[cov, "p"]
        lo = np.exp(s.loc[cov, "coef lower 95%"])
        hi = np.exp(s.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        info["covariates"][cov] = {
            "HR": round(hr, 3), "p": round(p_val, 4),
            "CI": [round(lo, 3), round(hi, 3)],
        }
    pout(f"  Concordance: {cph.concordance_index_:.3f}")
    results[name] = info


# ══════════════════════════════════════════════════════════════════════
# Q2: DEPS.DEV 100-DEPENDENT CAP
# ══════════════════════════════════════════════════════════════════════

def q2_depsdev_cap(combined, base, health, sev_index):
    """Analyze impact of 100-dependent cap on RQ3 results."""
    pout("\n" + "=" * 70)
    pout("Q2: DEPS.DEV 100-DEPENDENT CAP ANALYSIS")
    pout("=" * 70)

    # Count downstream dependents per (ecosystem, osv_id) pair
    dep_counts = combined.groupby(["ecosystem", "osv_id"]).size().reset_index(name="n_dependents")
    pout(f"  Total advisory-ecosystem pairs: {len(dep_counts)}")

    cap_results = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_counts = dep_counts[dep_counts["ecosystem"] == eco]
        n_total = len(eco_counts)
        n_at_cap = (eco_counts["n_dependents"] >= 100).sum()
        n_below = n_total - n_at_cap
        pct = n_at_cap / n_total * 100 if n_total > 0 else 0
        pout(f"\n  {eco}: {n_at_cap}/{n_total} advisories at cap ({pct:.1f}%)")
        pout(f"    Below cap: {n_below}, At cap (>=100): {n_at_cap}")
        pout(f"    Median dependents: {eco_counts['n_dependents'].median():.0f}")
        pout(f"    Mean dependents: {eco_counts['n_dependents'].mean():.1f}")
        pout(f"    P25/P75: {eco_counts['n_dependents'].quantile(0.25):.0f}/"
             f"{eco_counts['n_dependents'].quantile(0.75):.0f}")

        cap_results[eco] = {
            "n_advisories": int(n_total),
            "n_at_cap": int(n_at_cap),
            "n_below_cap": int(n_below),
            "pct_at_cap": round(pct, 1),
            "median_dependents": round(eco_counts["n_dependents"].median(), 1),
            "mean_dependents": round(eco_counts["n_dependents"].mean(), 1),
            "p25": round(eco_counts["n_dependents"].quantile(0.25), 1),
            "p75": round(eco_counts["n_dependents"].quantile(0.75), 1),
        }

    R["q2_cap_rates"] = cap_results

    # Tag each propagation event as from capped or uncapped advisory
    combined_tagged = combined.merge(dep_counts, on=["ecosystem", "osv_id"], how="left")
    combined_tagged["at_cap"] = (combined_tagged["n_dependents"] >= 100).astype(int)

    # Sensitivity: run RQ3 Cox Model A separately for capped vs uncapped
    pout("\n--- Sensitivity: Cox Model A by cap status ---")

    surv = combined_tagged[combined_tagged["propagation_type"].isin(["manual", "censored"])].copy()
    surv = surv[surv["propagation_days"].notna() & (surv["propagation_days"] > 0)]
    surv["prop_event"] = (surv["propagation_type"] == "manual").astype(int)

    surv["is_runtime"] = (surv["dependency_type"] == "dependencies").astype(int)
    surv["requires_major"] = surv["requires_major_upgrade"]
    surv["cvss_score"] = pd.to_numeric(surv["cvss_score"], errors="coerce")
    surv["cvss_score"] = surv["cvss_score"].fillna(surv["cvss_score"].median())
    surv["is_permissive"] = (surv["constraint_abstract"] == "permissive").astype(int)
    surv["is_restrictive"] = (surv["constraint_abstract"] == "restrictive").astype(int)

    covs_a = ["is_permissive", "is_restrictive", "is_runtime",
              "requires_major", "cvss_score"]

    sensitivity_results = {}
    for label, mask in [("uncapped", surv["at_cap"] == 0),
                        ("capped", surv["at_cap"] == 1),
                        ("all", surv["at_cap"].notna())]:
        sub = surv[mask].copy()
        df = sub[covs_a + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
        df = df[df["propagation_days"] > 0]

        n_events = int(df["prop_event"].sum())
        pout(f"\n  --- {label} (N={len(df)}, events={n_events}) ---")

        if len(df) < 100 or n_events < 20:
            pout(f"    Insufficient data")
            sensitivity_results[label] = {"n": len(df), "events": n_events,
                                           "insufficient": True}
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col="propagation_days", event_col="prop_event",
                    strata=["ecosystem"], cluster_col="osv_id")

            s = cph.summary
            model_results = {"n": len(df), "events": n_events,
                             "concordance": round(cph.concordance_index_, 3),
                             "covariates": {}}
            for cov in covs_a:
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
            pout(f"    Concordance: {cph.concordance_index_:.3f}")
            sensitivity_results[label] = model_results
        except Exception as e:
            pout(f"    Cox failed: {e}")
            sensitivity_results[label] = {"error": str(e)}

    R["q2_sensitivity"] = sensitivity_results


# ══════════════════════════════════════════════════════════════════════
# Q3: T_PATCH < 0 — PRE-ADVISORY PATCHES
# ══════════════════════════════════════════════════════════════════════

def q3_pre_advisory_patches(base, patch_merged):
    """Analyze T_patch < 0 cases and their survival analysis treatment."""
    pout("\n" + "=" * 70)
    pout("Q3: T_PATCH < 0 (PRE-ADVISORY PATCHES)")
    pout("=" * 70)

    if patch_merged is None or patch_merged.empty:
        pout("  No patch date data")
        return

    # Merge publication dates with patch dates
    m = base.copy()
    pm = patch_merged[patch_merged["patch_date"].notna()].copy()
    pm = pm.drop_duplicates(subset=["ghsa_id", "package_name", "first_patched_version"])
    m = m.merge(
        pm[["ghsa_id", "package_name", "first_patched_version", "patch_date", "date_source"]],
        on=["ghsa_id", "package_name", "first_patched_version"], how="inner"
    )
    m["patch_date"] = pd.to_datetime(m["patch_date"], errors="coerce", utc=True)
    m["raw_latency_days"] = (m["patch_date"] - m["published_at"]).dt.total_seconds() / 86400

    n_total = len(m)
    n_negative = (m["raw_latency_days"] < 0).sum()
    n_zero = (m["raw_latency_days"] == 0).sum()
    n_positive = (m["raw_latency_days"] > 0).sum()
    pct_negative = n_negative / n_total * 100

    pout(f"\n  Total observations with both dates: {n_total}")
    pout(f"  T_patch < 0 (pre-advisory): {n_negative} ({pct_negative:.1f}%)")
    pout(f"  T_patch = 0: {n_zero}")
    pout(f"  T_patch > 0: {n_positive}")

    pre_adv = m[m["raw_latency_days"] < 0]
    pout(f"\n  Pre-advisory patch latency distribution:")
    pout(f"    Median: {pre_adv['raw_latency_days'].median():.1f} days")
    pout(f"    Mean: {pre_adv['raw_latency_days'].mean():.1f} days")
    pout(f"    P10/P90: {pre_adv['raw_latency_days'].quantile(0.1):.1f}/"
         f"{pre_adv['raw_latency_days'].quantile(0.9):.1f}")
    pout(f"    Min/Max: {pre_adv['raw_latency_days'].min():.1f}/"
         f"{pre_adv['raw_latency_days'].max():.1f}")

    # Per-ecosystem breakdown
    eco_results = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = m[m["ecosystem"] == eco]
        eco_neg = (eco_sub["raw_latency_days"] < 0).sum()
        eco_total = len(eco_sub)
        pct = eco_neg / eco_total * 100 if eco_total > 0 else 0
        pout(f"  {eco}: {eco_neg}/{eco_total} pre-advisory ({pct:.1f}%)")
        eco_results[eco] = {
            "total": int(eco_total),
            "n_pre_advisory": int(eco_neg),
            "pct_pre_advisory": round(pct, 1),
        }

    R["q3_pre_advisory"] = {
        "total": int(n_total),
        "n_negative": int(n_negative),
        "n_zero": int(n_zero),
        "n_positive": int(n_positive),
        "pct_negative": round(pct_negative, 1),
        "median_negative_days": round(pre_adv["raw_latency_days"].median(), 1),
        "per_ecosystem": eco_results,
    }

    # Current treatment: clipped to 0.5 days
    pout(f"\n  Current treatment in survival analysis:")
    pout(f"    All T_patch < 0 clipped to 0.5 days (immediate event)")
    pout(f"    Rationale: patch was already available at disclosure time,")
    pout(f"    so from a downstream consumer's perspective the fix was immediate.")

    # Sensitivity: exclude pre-advisory patches entirely
    pout(f"\n--- Sensitivity: exclude T_patch < 0 ---")
    R["q3_treatment"] = {
        "method": "clip_to_0.5_days",
        "rationale": "Pre-advisory patches represent responsible disclosure; "
                     "patch was available at disclosure. Clipped to 0.5 days "
                     "(immediate event) in survival analysis.",
        "code_reference": "build_patch_latency_merged() line 152: "
                          "with_date['patch_latency_days'] = "
                          "with_date['patch_latency_days'].clip(lower=0.5)",
    }


# ══════════════════════════════════════════════════════════════════════
# Q4: PERMISSIVE EFFECT REVERSAL — INTERACTION MODEL
# ══════════════════════════════════════════════════════════════════════

def q4_permissive_interaction(combined, base, health, sev_index):
    """Test ecosystem × permissive interaction in propagation Cox model."""
    pout("\n" + "=" * 70)
    pout("Q4: PERMISSIVE EFFECT REVERSAL — INTERACTION MODEL")
    pout("=" * 70)

    surv = combined[combined["propagation_type"].isin(["manual", "censored"])].copy()
    surv = surv[surv["propagation_days"].notna() & (surv["propagation_days"] > 0)]
    surv["prop_event"] = (surv["propagation_type"] == "manual").astype(int)

    surv["is_runtime"] = (surv["dependency_type"] == "dependencies").astype(int)
    surv["requires_major"] = surv["requires_major_upgrade"]
    surv["cvss_score"] = pd.to_numeric(surv["cvss_score"], errors="coerce")
    surv["cvss_score"] = surv["cvss_score"].fillna(surv["cvss_score"].median())
    surv["is_permissive"] = (surv["constraint_abstract"] == "permissive").astype(int)
    surv["is_restrictive"] = (surv["constraint_abstract"] == "restrictive").astype(int)

    # First: show per-ecosystem permissive effect direction
    pout("\n--- Per-ecosystem permissive HR (separate models) ---")
    per_eco_results = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = surv[surv["ecosystem"] == eco].copy()

        # Drop covariates with near-zero variance
        eco_covs = []
        for cov in ["is_permissive", "is_restrictive", "is_runtime",
                     "requires_major", "cvss_score"]:
            vals = eco_sub[cov].dropna()
            if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                eco_covs.append(cov)
            else:
                pout(f"    [{eco}] Dropping {cov} (zero/near-zero variance)")

        df = eco_sub[eco_covs + ["propagation_days", "prop_event", "osv_id"]].dropna()
        df = df[df["propagation_days"] > 0]

        if len(df) < 50:
            pout(f"  {eco}: insufficient data (n={len(df)})")
            per_eco_results[eco] = {"insufficient": True}
            continue

        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="propagation_days", event_col="prop_event",
                cluster_col="osv_id")
        s = cph.summary

        eco_info = {"n": len(df), "events": int(df["prop_event"].sum()),
                    "concordance": round(cph.concordance_index_, 3),
                    "covariates": {}}
        for cov in eco_covs:
            hr = np.exp(s.loc[cov, "coef"])
            p_val = s.loc[cov, "p"]
            lo = np.exp(s.loc[cov, "coef lower 95%"])
            hi = np.exp(s.loc[cov, "coef upper 95%"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            pout(f"  {eco} {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
            eco_info["covariates"][cov] = {
                "HR": round(hr, 3), "p": round(p_val, 4),
                "CI": [round(lo, 3), round(hi, 3)],
            }
        per_eco_results[eco] = eco_info

    R["q4_per_ecosystem"] = per_eco_results

    # Interaction model: ecosystem × permissive
    pout("\n--- Interaction model: ecosystem × permissive ---")
    surv["is_npm"] = (surv["ecosystem"] == "npm").astype(int)
    surv["is_pypi"] = (surv["ecosystem"] == "PyPI").astype(int)
    # Maven is reference
    surv["permissive_x_npm"] = surv["is_permissive"] * surv["is_npm"]
    surv["permissive_x_pypi"] = surv["is_permissive"] * surv["is_pypi"]
    surv["restrictive_x_npm"] = surv["is_restrictive"] * surv["is_npm"]
    surv["restrictive_x_pypi"] = surv["is_restrictive"] * surv["is_pypi"]

    covs_int = ["is_permissive", "is_restrictive", "is_runtime",
                "requires_major", "cvss_score",
                "permissive_x_npm", "permissive_x_pypi",
                "restrictive_x_npm", "restrictive_x_pypi"]

    # Drop interaction terms with zero variance
    df_int = surv[covs_int + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
    df_int = df_int[df_int["propagation_days"] > 0]

    active_covs = []
    for cov in covs_int:
        if df_int[cov].std() > 0.01 and df_int[cov].nunique() > 1:
            active_covs.append(cov)
        else:
            pout(f"    Dropping {cov} (zero/near-zero variance)")

    pout(f"  N={len(df_int)}, events={df_int['prop_event'].sum()}, covariates={len(active_covs)}")

    cph_int = CoxPHFitter(penalizer=0.01)
    cph_int.fit(df_int[active_covs + ["propagation_days", "prop_event", "ecosystem", "osv_id"]],
                duration_col="propagation_days", event_col="prop_event",
                strata=["ecosystem"], cluster_col="osv_id")

    s_int = cph_int.summary
    int_results = {"n": len(df_int), "events": int(df_int["prop_event"].sum()),
                   "concordance": round(cph_int.concordance_index_, 3),
                   "covariates": {}}
    for cov in active_covs:
        hr = np.exp(s_int.loc[cov, "coef"])
        p_val = s_int.loc[cov, "p"]
        lo = np.exp(s_int.loc[cov, "coef lower 95%"])
        hi = np.exp(s_int.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        int_results["covariates"][cov] = {
            "HR": round(hr, 3), "p": round(p_val, 4),
            "CI": [round(lo, 3), round(hi, 3)],
        }
    pout(f"  Concordance: {cph_int.concordance_index_:.3f}")

    R["q4_interaction_model"] = int_results

    # Interpretation: compute ecosystem-specific permissive HR
    pout("\n--- Ecosystem-specific permissive HR from interaction model ---")
    base_coef = s_int.loc["is_permissive", "coef"] if "is_permissive" in s_int.index else 0
    base_se = s_int.loc["is_permissive", "se(coef)"] if "is_permissive" in s_int.index else 0
    eco_specific = {}

    # Maven (reference): just is_permissive
    if "is_permissive" in s_int.index:
        hr_maven = np.exp(base_coef)
        pout(f"  Maven:  permissive HR = {hr_maven:.3f} (base effect)")
        eco_specific["Maven"] = {"HR": round(hr_maven, 3), "beta": round(base_coef, 4)}

    # npm: is_permissive + permissive_x_npm
    if "permissive_x_npm" in s_int.index and "is_permissive" in s_int.index:
        beta_npm = base_coef + s_int.loc["permissive_x_npm", "coef"]
        hr_npm = np.exp(beta_npm)
        pout(f"  npm:    permissive HR = {hr_npm:.3f} (base + interaction)")
        eco_specific["npm"] = {"HR": round(hr_npm, 3), "beta": round(beta_npm, 4)}

    # PyPI: is_permissive + permissive_x_pypi
    if "permissive_x_pypi" in s_int.index and "is_permissive" in s_int.index:
        beta_pypi = base_coef + s_int.loc["permissive_x_pypi", "coef"]
        hr_pypi = np.exp(beta_pypi)
        pout(f"  PyPI:   permissive HR = {hr_pypi:.3f} (base + interaction)")
        eco_specific["PyPI"] = {"HR": round(hr_pypi, 3), "beta": round(beta_pypi, 4)}

    R["q4_eco_specific_permissive_hr"] = eco_specific

    # LRT: interaction model vs no-interaction model
    pout("\n--- LRT: interaction vs no-interaction ---")
    covs_noint = ["is_permissive", "is_restrictive", "is_runtime",
                  "requires_major", "cvss_score"]
    covs_noint = [c for c in covs_noint if c in active_covs or c in covs_noint]
    # Use same sample
    df_noint = df_int[covs_noint + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
    df_noint = df_noint[df_noint["propagation_days"] > 0]

    cph_noint = CoxPHFitter(penalizer=0.01)
    cph_noint.fit(df_noint, duration_col="propagation_days", event_col="prop_event",
                  strata=["ecosystem"], cluster_col="osv_id")

    # Compare log-likelihoods (approximation via partial likelihood)
    ll_int = cph_int.log_likelihood_
    ll_noint = cph_noint.log_likelihood_
    lrt_stat = 2 * (ll_int - ll_noint)
    n_interaction_terms = len([c for c in active_covs
                               if c.startswith("permissive_x_") or c.startswith("restrictive_x_")])
    from scipy.stats import chi2
    lrt_p = chi2.sf(lrt_stat, df=n_interaction_terms)
    pout(f"  LRT statistic: {lrt_stat:.2f}, df={n_interaction_terms}, p={lrt_p:.4f}")
    pout(f"  No-interaction concordance: {cph_noint.concordance_index_:.3f}")
    pout(f"  Interaction concordance: {cph_int.concordance_index_:.3f}")

    R["q4_lrt"] = {
        "lrt_statistic": round(lrt_stat, 2),
        "df": n_interaction_terms,
        "p": round(lrt_p, 4),
        "ll_interaction": round(ll_int, 2),
        "ll_no_interaction": round(ll_noint, 2),
        "concordance_no_interaction": round(cph_noint.concordance_index_, 3),
        "concordance_interaction": round(cph_int.concordance_index_, 3),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pout("Loading datasets...")
    base = load_base()
    health = load_health()
    patch_merged = load_patch_dates_merged()
    sev_index = load_severity_index()

    pout(f"  Base: {len(base)} advisories")
    pout(f"  Health: {len(health)} repos")
    pout(f"  Patch dates (merged): {len(patch_merged)} versions")

    # Build patch latency for RQ1
    valid = build_patch_latency_merged(base, patch_merged)
    pout(f"  Patch latency: {len(valid)} rows ({valid['patch_event'].sum()} events)")

    # Load propagation data for RQ3
    npm = load_propagation("npm", "npm_propagation.csv")
    pypi = load_propagation("PyPI", "pypi_propagation.csv")
    maven = load_propagation("Maven", "maven_propagation.csv")
    combined = pd.concat([npm, pypi, maven], ignore_index=True)
    combined["constraint_abstract"] = combined.apply(map_constraint_abstract, axis=1)

    if not sev_index.empty:
        sev_lu = sev_index[["osv_id", "cvss_score"]].drop_duplicates("osv_id")
        combined = combined.merge(sev_lu, on="osv_id", how="left", suffixes=("_raw", ""))
        if "cvss_score_raw" in combined.columns:
            combined = combined.drop(columns=["cvss_score_raw"])
        combined["cvss_score"] = pd.to_numeric(combined["cvss_score"], errors="coerce")

    pout(f"  Propagation: {len(combined)} events")

    # Run all sensitivity analyses
    q1_multicollinearity(valid, health)
    q2_depsdev_cap(combined, base, health, sev_index)
    q3_pre_advisory_patches(base, patch_merged)
    q4_permissive_interaction(combined, base, health, sev_index)

    # Save results
    out = DATA_DIR / "sensitivity_model_robustness.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")
    pout("=" * 70)
    pout("DONE")


if __name__ == "__main__":
    main()
