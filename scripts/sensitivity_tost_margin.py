#!/usr/bin/env python3
"""Equivalence-margin sensitivity: TOST at multiple delta_HR margins for
all per-ecosystem propagation models and the pooled model, so the
health-propagation decoupling does not rest on a single margin choice.

Margins tested: delta_HR = {0.10, 0.15, 0.20, 0.25, 0.30}

Output: data/sensitivity_tost_margin.json
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
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_cross_ecosystem import (
    load_base, load_health, load_propagation,
    load_severity_index, map_constraint_abstract,
)

MARGINS = [0.10, 0.15, 0.20, 0.25, 0.30]
HEALTH_COVS = ["log_contributors", "log_stars"]


def pout(msg):
    print(msg, file=sys.stderr)


def tost_test(beta, se, delta_hr):
    """Run TOST equivalence test for a Cox PH coefficient.

    H0: |beta| >= log(1 + delta_hr)  (i.e. HR is NOT equivalent to 1)
    H1: |beta| <  log(1 + delta_hr)  (i.e. HR IS equivalent to 1)

    Two one-sided tests:
      p_lower = P(Z > (beta + log(1+delta)) / SE)   [test beta > -delta]
      p_upper = P(Z > (log(1+delta) - beta) / SE)   ... wait, standard form:

    Actually the standard TOST for beta in [-delta, +delta]:
      Test 1 (lower): H0: beta <= -delta  =>  z1 = (beta - (-delta)) / SE
                       p1 = 1 - Phi(z1)   ... reject if z1 large (p1 small)
                       Actually p1 = P(Z > z1) ... for upper tail.
                       No: z1 = (beta + delta)/SE, reject lower H0 if z1 > z_alpha
                       p_lower = 1 - norm.cdf(z1)
      Test 2 (upper): H0: beta >= +delta  =>  z2 = (delta - beta) / SE
                       p_upper = 1 - norm.cdf(z2)

    p_tost = max(p_lower, p_upper).  Reject H0 (declare equivalence) if p_tost < alpha.
    """
    delta = np.log(1 + delta_hr)

    z_lower = (beta - (-delta)) / se   # = (beta + delta) / se
    z_upper = (delta - beta) / se

    p_lower = 1 - norm.cdf(z_lower)
    p_upper = 1 - norm.cdf(z_upper)
    p_tost = max(p_lower, p_upper)

    # 90% CI for equivalence assessment (corresponds to alpha=0.05 two one-sided)
    ci90_lo = beta - 1.645 * se
    ci90_hi = beta + 1.645 * se
    equivalent = (ci90_lo > -delta) and (ci90_hi < delta)

    return {
        "p_lower": round(p_lower, 6),
        "p_upper": round(p_upper, 6),
        "p_tost": round(p_tost, 6),
        "equivalent": bool(equivalent),
        "CI90_beta": [round(ci90_lo, 4), round(ci90_hi, 4)],
        "CI90_HR": [round(np.exp(ci90_lo), 4), round(np.exp(ci90_hi), 4)],
        "delta_log": round(delta, 4),
    }


def build_propagation_df(combined, base, health):
    """Build the RQ3 propagation DataFrame with health covariates (Model B)."""
    surv = combined[combined["propagation_type"].isin(["manual", "censored"])].copy()
    surv = surv[surv["propagation_days"].notna() & (surv["propagation_days"] > 0)]
    surv["prop_event"] = (surv["propagation_type"] == "manual").astype(int)

    surv["is_runtime"] = (surv["dependency_type"] == "dependencies").astype(int)
    surv["requires_major"] = surv["requires_major_upgrade"]
    surv["cvss_score"] = pd.to_numeric(surv["cvss_score"], errors="coerce")
    surv["cvss_score"] = surv["cvss_score"].fillna(surv["cvss_score"].median())
    surv["is_permissive"] = (surv["constraint_abstract"] == "permissive").astype(int)
    surv["is_restrictive"] = (surv["constraint_abstract"] == "restrictive").astype(int)

    # Link upstream health via osv_id -> base -> github_repo -> health
    if not health.empty:
        base_repo = base[["ghsa_id", "github_repo"]].drop_duplicates("ghsa_id")
        surv = surv.merge(base_repo, left_on="osv_id", right_on="ghsa_id", how="left")
        h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        if "github_repo_x" in surv.columns:
            surv["github_repo"] = surv["github_repo_x"].fillna(surv.get("github_repo_y"))
            surv = surv.drop(columns=["github_repo_x", "github_repo_y"], errors="ignore")
        surv = surv.merge(h, on="github_repo", how="left")

    return surv


def fit_and_tost(df, covs, label, strata=None, cluster_col="osv_id"):
    """Fit Cox PH and run TOST at all margins for health covariates.

    Returns dict with model info and TOST results per covariate.
    """
    result = {}

    n_events = int(df["prop_event"].sum())
    pout(f"\n  [{label}] n={len(df)}, events={n_events}, covs={covs}")

    if len(df) < 50 or n_events < 10:
        pout(f"    Insufficient data")
        return {"insufficient_data": True, "n": len(df), "events": n_events}

    try:
        cph = CoxPHFitter(penalizer=0.01)
        fit_kwargs = {"duration_col": "propagation_days", "event_col": "prop_event"}
        if strata:
            fit_kwargs["strata"] = strata
        if cluster_col and cluster_col in df.columns:
            fit_kwargs["cluster_col"] = cluster_col
        cph.fit(df, **fit_kwargs)

        s = cph.summary
        result["n"] = len(df)
        result["events"] = n_events
        result["concordance"] = round(cph.concordance_index_, 3)

        for cov in HEALTH_COVS:
            if cov not in s.index:
                pout(f"    {cov}: not in model (dropped)")
                continue

            beta = s.loc[cov, "coef"]
            se = s.loc[cov, "se(coef)"]
            hr = np.exp(beta)

            cov_result = {
                "HR": round(hr, 4),
                "beta": round(beta, 4),
                "SE": round(se, 4),
                "results": [],
            }

            pout(f"    {cov}: HR={hr:.4f}, beta={beta:.4f}, SE={se:.4f}")

            for delta_hr in MARGINS:
                t = tost_test(beta, se, delta_hr)
                t["delta"] = delta_hr
                cov_result["results"].append(t)
                flag = "EQUIV" if t["equivalent"] else "not equiv"
                pout(f"      delta={delta_hr:.2f}: p_tost={t['p_tost']:.4f} "
                     f"90%CI_HR={t['CI90_HR']} {flag}")

            result[cov] = cov_result

    except Exception as e:
        pout(f"    Cox failed: {e}")
        result["error"] = str(e)

    return result


def main():
    pout("=" * 70)
    pout("D3: TOST SENSITIVITY AT MULTIPLE MARGINS")
    pout("=" * 70)

    pout("\nLoading data...")
    base = load_base()
    health = load_health()
    sev_index = load_severity_index()

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

    pout(f"  Combined propagation: {len(combined)} events")

    surv = build_propagation_df(combined, base, health)
    pout(f"  Survival subset: {len(surv)} events")

    R = {"margins": MARGINS}

    # ── Structural + health covariates ──
    covs_a = ["is_permissive", "is_restrictive", "is_runtime",
              "requires_major", "cvss_score"]
    covs_b = covs_a + ["log_contributors", "log_stars"]

    # ── Pooled model (ecosystem-stratified) ──
    pout("\n" + "=" * 70)
    pout("POOLED MODEL (ecosystem-stratified)")
    pout("=" * 70)

    df_pooled = surv[
        covs_b + ["propagation_days", "prop_event", "ecosystem", "osv_id"]
    ].dropna()
    df_pooled = df_pooled[df_pooled["propagation_days"] > 0]

    R["pooled"] = fit_and_tost(
        df_pooled, covs_b, "pooled", strata=["ecosystem"], cluster_col="osv_id"
    )

    # ── Per-ecosystem models ──
    pout("\n" + "=" * 70)
    pout("PER-ECOSYSTEM MODELS")
    pout("=" * 70)

    for eco in ["npm", "PyPI", "Maven"]:
        pout(f"\n--- {eco} ---")
        eco_surv = surv[surv["ecosystem"] == eco].copy()

        # Drop covariates with near-zero variance
        eco_covs = []
        for cov in covs_b:
            if cov in eco_surv.columns:
                vals = eco_surv[cov].dropna()
                if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                    eco_covs.append(cov)
                else:
                    pout(f"    Dropping {cov} (zero/near-zero variance)")

        df_eco = eco_surv[
            eco_covs + ["propagation_days", "prop_event", "osv_id"]
        ].dropna()
        df_eco = df_eco[df_eco["propagation_days"] > 0]

        R[eco] = fit_and_tost(
            df_eco, eco_covs, eco, strata=None, cluster_col="osv_id"
        )

    # ── Summary table ──
    pout("\n" + "=" * 70)
    pout("SUMMARY TABLE")
    pout("=" * 70)

    header = f"{'Model':<10} {'Covariate':<20} {'HR':>8} {'beta':>8} {'SE':>8}"
    for d in MARGINS:
        header += f"  d={d:.2f}"
    pout(header)
    pout("-" * len(header))

    for model_name in ["pooled", "npm", "PyPI", "Maven"]:
        model = R.get(model_name, {})
        if "insufficient_data" in model or "error" in model:
            pout(f"{model_name:<10} INSUFFICIENT DATA or ERROR")
            continue
        for cov in HEALTH_COVS:
            if cov not in model:
                continue
            cov_data = model[cov]
            row = f"{model_name:<10} {cov:<20} {cov_data['HR']:>8.4f} {cov_data['beta']:>8.4f} {cov_data['SE']:>8.4f}"
            for res in cov_data["results"]:
                flag = "Y" if res["equivalent"] else "N"
                row += f"  {flag}({res['p_tost']:.3f})"
            pout(row)

    # Save
    out = DATA_DIR / "sensitivity_tost_margin.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")
    pout("DONE")

    # Also print to stdout for easy capture
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
