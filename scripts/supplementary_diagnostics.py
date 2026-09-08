#!/usr/bin/env python3
"""Supplementary model diagnostics reported alongside the main analysis.

Computes:
  1. VIF and Pearson correlation for PyPI per-ecosystem model (multicollinearity)
  2. Joint LRT for health covariates in per-ecosystem models
  3. Per-ecosystem censoring rates
  4. Regenerated forest plot with per-ecosystem comparison (Figure 6)

Output: data/results_supplementary.json, paper/fig/forest_propagation.pdf (updated)
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
from scipy.stats import chi2 as chi2_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "paper" / "fig"

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "figure.figsize": (3.4, 2.5),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Constraint type abstraction (must match main script)
CONSTRAINT_MAP = {
    "caret": "permissive", "range": "permissive", "lower_bound": "permissive",
    "upper_bound": "permissive",
    "tilde": "restrictive", "compatible": "restrictive",
    "exact": "pinned", "pinned": "pinned", "soft": "pinned",
    "any": "wildcard", "other": "wildcard",
}


def pout(msg):
    print(msg, file=sys.stderr)


def map_constraint_abstract(row):
    ct = row.get("constraint_type", "")
    return CONSTRAINT_MAP.get(str(ct).lower(), "pinned")


def load_propagation(ecosystem, fname):
    p = DATA_DIR / fname
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["ecosystem"] = ecosystem
    df["propagation_days"] = pd.to_numeric(df["propagation_days"], errors="coerce")
    df["requires_major_upgrade"] = pd.to_numeric(
        df["requires_major_upgrade"], errors="coerce"
    ).fillna(0).astype(int)
    return df


def build_surv_data():
    """Build survival dataset exactly as in the main analysis script."""
    sev_index = pd.read_csv(DATA_DIR / "osv_severity_index.csv")
    base = pd.read_csv(DATA_DIR / "cross_ecosystem_base.csv")
    health = pd.read_csv(DATA_DIR / "github_repo_health.csv")

    # Health preprocessing
    if "contributors_count" in health.columns:
        health["contributors"] = health["contributors_count"]
    for col in ["stars", "contributors"]:
        if col in health.columns:
            health[f"log_{col}"] = np.log1p(health[col])
    if "created_at" in health.columns:
        created = pd.to_datetime(health["created_at"], utc=True)
        health["age_years"] = (pd.Timestamp("2026-03-07", tz="UTC") - created).dt.days / 365.25

    # Load propagation
    npm = load_propagation("npm", "npm_propagation.csv")
    pypi = load_propagation("PyPI", "pypi_propagation.csv")
    maven = load_propagation("Maven", "maven_propagation.csv")

    combined = pd.concat([npm, pypi, maven], ignore_index=True)
    combined["constraint_abstract"] = combined.apply(map_constraint_abstract, axis=1)

    # Add CVSS
    if not sev_index.empty:
        sev_lu = sev_index[["osv_id", "cvss_score"]].drop_duplicates("osv_id")
        combined = combined.merge(sev_lu, on="osv_id", how="left", suffixes=("_raw", ""))
        if "cvss_score_raw" in combined.columns:
            combined = combined.drop(columns=["cvss_score_raw"])
        combined["cvss_score"] = pd.to_numeric(combined["cvss_score"], errors="coerce")

    pout(f"  Combined: {len(combined)} events")
    pout(f"    npm: {len(npm)}, PyPI: {len(pypi)}, Maven: {len(maven)}")

    # Build survival subset (exactly as rq3_propagation_cox)
    surv = combined[combined["propagation_type"].isin(["manual", "censored"])].copy()
    surv = surv[surv["propagation_days"].notna() & (surv["propagation_days"] > 0)]
    surv["prop_event"] = (surv["propagation_type"] == "manual").astype(int)

    pout(f"  Survival subset: n={len(surv)}, events={surv['prop_event'].sum()}")

    # Covariates
    surv["is_runtime"] = (surv["dependency_type"] == "dependencies").astype(int)
    surv["requires_major"] = surv["requires_major_upgrade"]
    surv["cvss_score"] = surv["cvss_score"].fillna(surv["cvss_score"].median())
    surv["is_permissive"] = (surv["constraint_abstract"] == "permissive").astype(int)
    surv["is_restrictive"] = (surv["constraint_abstract"] == "restrictive").astype(int)

    # Add upstream health via osv_id -> base -> github_repo -> health
    if not health.empty:
        base_repo = base[["ghsa_id", "github_repo"]].drop_duplicates("ghsa_id")
        surv = surv.merge(base_repo, left_on="osv_id", right_on="ghsa_id", how="left")
        h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        if "github_repo_x" in surv.columns:
            surv["github_repo"] = surv["github_repo_x"].fillna(surv.get("github_repo_y"))
            surv = surv.drop(columns=["github_repo_x", "github_repo_y"], errors="ignore")
        surv = surv.merge(h, on="github_repo", how="left")
        health_n = surv["log_contributors"].notna().sum()
        pout(f"  Events with upstream health: {health_n}/{len(surv)}")

    return combined, surv


def compute_multicollinearity(surv, results):
    """Point 1: VIF and Pearson correlation for per-ecosystem models."""
    pout("\n" + "="*60)
    pout("MULTICOLLINEARITY DIAGNOSTICS")
    pout("="*60)

    all_covs = ["is_permissive", "is_restrictive", "is_runtime",
                "requires_major", "cvss_score", "log_contributors", "log_stars"]

    multicollinearity = {}

    for eco in ["npm", "PyPI", "Maven"]:
        eco_surv = surv[surv["ecosystem"] == eco].copy()

        # Identify usable covariates (same logic as _per_ecosystem_tost)
        eco_covs = []
        for cov in all_covs:
            if cov in eco_surv.columns:
                vals = eco_surv[cov].dropna()
                if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                    eco_covs.append(cov)

        df_eco = eco_surv[eco_covs + ["propagation_days", "prop_event"]].dropna()
        df_eco = df_eco[df_eco["propagation_days"] > 0]

        pout(f"\n--- {eco} (n={len(df_eco)}) ---")

        # Pearson correlation between health covariates
        r = None
        if "log_contributors" in eco_covs and "log_stars" in eco_covs:
            r = df_eco["log_contributors"].corr(df_eco["log_stars"])
            pout(f"  Pearson r(log_contributors, log_stars) = {r:.3f}")
        else:
            pout(f"  Cannot compute correlation (missing covariates)")

        # VIF for all covariates
        vif_data = {}
        if len(eco_covs) >= 2:
            X = df_eco[eco_covs].values
            for i, cov in enumerate(eco_covs):
                try:
                    vif = variance_inflation_factor(X, i)
                    vif_data[cov] = round(float(vif), 2)
                    pout(f"  VIF({cov}) = {vif:.2f}")
                except Exception as e:
                    pout(f"  VIF({cov}) failed: {e}")

        multicollinearity[eco] = {
            "n": len(df_eco),
            "pearson_r_health": round(r, 3) if r is not None else None,
            "vif": vif_data,
        }

    results["multicollinearity"] = multicollinearity


def compute_joint_lrt(surv, results):
    """Point 1: Joint likelihood ratio test for health covariates."""
    pout("\n" + "="*60)
    pout("JOINT LRT FOR HEALTH COVARIATES")
    pout("="*60)

    all_covs = ["is_permissive", "is_restrictive", "is_runtime",
                "requires_major", "cvss_score", "log_contributors", "log_stars"]

    joint_lrt = {}

    for eco in ["npm", "PyPI", "Maven"]:
        eco_surv = surv[surv["ecosystem"] == eco].copy()

        # Identify usable covariates
        eco_covs_full = []
        eco_covs_structural = []
        for cov in all_covs:
            if cov in eco_surv.columns:
                vals = eco_surv[cov].dropna()
                if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                    eco_covs_full.append(cov)
                    if cov not in ["log_contributors", "log_stars"]:
                        eco_covs_structural.append(cov)

        df_full = eco_surv[eco_covs_full + ["propagation_days", "prop_event", "osv_id"]].dropna()
        df_full = df_full[df_full["propagation_days"] > 0]

        pout(f"\n--- {eco} ---")

        if len(df_full) < 50 or df_full["prop_event"].sum() < 10:
            pout(f"  Insufficient data")
            continue

        try:
            # Full model (with health)
            cph_full = CoxPHFitter(penalizer=0.01)
            cph_full.fit(df_full, duration_col="propagation_days", event_col="prop_event",
                         cluster_col="osv_id")
            ll_full = cph_full.log_likelihood_

            # Restricted model (structural only)
            df_restr = df_full[eco_covs_structural + ["propagation_days", "prop_event", "osv_id"]]
            cph_restr = CoxPHFitter(penalizer=0.01)
            cph_restr.fit(df_restr, duration_col="propagation_days", event_col="prop_event",
                          cluster_col="osv_id")
            ll_restr = cph_restr.log_likelihood_

            # LRT: -2(ll_restricted - ll_full), df = number of dropped covariates
            n_health = len(eco_covs_full) - len(eco_covs_structural)
            lrt_stat = -2 * (ll_restr - ll_full)
            lrt_p = 1 - chi2_dist.cdf(lrt_stat, df=n_health)

            pout(f"  LL_full={ll_full:.2f}, LL_restr={ll_restr:.2f}")
            pout(f"  LRT chi2={lrt_stat:.2f}, df={n_health}, p={lrt_p:.6f}")
            sig = "***" if lrt_p < 0.001 else "**" if lrt_p < 0.01 else "*" if lrt_p < 0.05 else "ns"
            pout(f"  Joint health significance: {sig}")

            joint_lrt[eco] = {
                "ll_full": round(ll_full, 2),
                "ll_restricted": round(ll_restr, 2),
                "lrt_chi2": round(lrt_stat, 2),
                "lrt_df": n_health,
                "lrt_p": round(lrt_p, 6),
                "significant": lrt_p < 0.05,
            }
        except Exception as e:
            pout(f"  LRT failed: {e}")

    results["joint_lrt_health"] = joint_lrt


def compute_censoring_rates(combined, results):
    """Point 4: Per-ecosystem censoring rates."""
    pout("\n" + "="*60)
    pout("PER-ECOSYSTEM CENSORING RATES")
    pout("="*60)

    censoring = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_data = combined[combined["ecosystem"] == eco]
        total = len(eco_data)
        auto = int((eco_data["propagation_type"] == "auto").sum())
        manual = int((eco_data["propagation_type"] == "manual").sum())
        censored = int((eco_data["propagation_type"] == "censored").sum())

        # For non-auto events (used in Cox models)
        non_auto = eco_data[eco_data["propagation_type"] != "auto"]
        non_auto_total = len(non_auto)
        non_auto_censored = int((non_auto["propagation_type"] == "censored").sum())
        non_auto_censor_rate = non_auto_censored / non_auto_total if non_auto_total > 0 else 0

        pout(f"  {eco}: total={total}, auto={auto}, manual={manual}, censored={censored}")
        pout(f"    Non-auto: {non_auto_total}, censored={non_auto_censored} "
             f"({non_auto_censor_rate:.1%})")

        censoring[eco] = {
            "total": int(total),
            "auto": auto,
            "manual": manual,
            "censored": censored,
            "non_auto_total": int(non_auto_total),
            "non_auto_censored": non_auto_censored,
            "non_auto_censor_rate": round(non_auto_censor_rate, 3),
        }

    # Overall
    total = len(combined)
    censored_total = int((combined["propagation_type"] == "censored").sum())
    pout(f"  Overall: {total} total, {censored_total} censored ({censored_total/total:.1%})")

    results["censoring_rates"] = censoring


def regenerate_forest_plot(surv):
    """Point 6: Regenerate forest plot with pooled + per-ecosystem health covariates."""
    pout("\n" + "="*60)
    pout("REGENERATING FOREST PLOT (POOLED + PER-ECOSYSTEM)")
    pout("="*60)

    all_covs = ["is_permissive", "is_restrictive", "is_runtime",
                "requires_major", "cvss_score", "log_contributors", "log_stars"]
    health_covs = ["log_contributors", "log_stars"]

    # Fit per-ecosystem models
    eco_results = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_surv = surv[surv["ecosystem"] == eco].copy()
        eco_covs = []
        for cov in all_covs:
            if cov in eco_surv.columns:
                vals = eco_surv[cov].dropna()
                if len(vals) > 0 and vals.std() > 0.01 and vals.nunique() > 1:
                    eco_covs.append(cov)

        df_eco = eco_surv[eco_covs + ["propagation_days", "prop_event", "osv_id"]].dropna()
        df_eco = df_eco[df_eco["propagation_days"] > 0]

        if len(df_eco) < 50 or df_eco["prop_event"].sum() < 10:
            pout(f"  {eco}: insufficient data")
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df_eco, duration_col="propagation_days", event_col="prop_event",
                    cluster_col="osv_id")
            eco_results[eco] = cph.summary
            pout(f"  {eco}: fitted (n={len(df_eco)}, events={int(df_eco['prop_event'].sum())})")
        except Exception as e:
            pout(f"  {eco} failed: {e}")

    # Fit pooled model (ecosystem-stratified)
    covs_b = [c for c in all_covs if c in surv.columns]
    df_b = surv[covs_b + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
    df_b = df_b[df_b["propagation_days"] > 0]

    cph_pooled = CoxPHFitter(penalizer=0.01)
    cph_pooled.fit(df_b, duration_col="propagation_days", event_col="prop_event",
                   strata=["ecosystem"], cluster_col="osv_id")
    pooled_summary = cph_pooled.summary
    pout(f"  Pooled: fitted (n={len(df_b)}, events={int(df_b['prop_event'].sum())})")

    # Build figure: health covariates only, pooled + per-ecosystem
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    colors = {"Pooled": "#555555", "npm": "#e74c3c", "PyPI": "#3498db", "Maven": "#2ecc71"}
    markers = {"Pooled": "D", "npm": "o", "PyPI": "s", "Maven": "^"}
    model_order = ["Pooled", "npm", "PyPI", "Maven"]

    y_positions = []
    y_labels = []
    y_pos = 0
    group_gap = 1.2
    n_in_first_group = 0

    for ci, cov in enumerate(health_covs):
        for model_name in model_order:
            if model_name == "Pooled":
                summary = pooled_summary
            else:
                summary = eco_results.get(model_name)

            if summary is None or cov not in summary.index:
                continue

            coef = summary.loc[cov, "coef"]
            lo = summary.loc[cov, "coef lower 95%"]
            hi = summary.loc[cov, "coef upper 95%"]
            p_val = summary.loc[cov, "p"]

            ax.plot(coef, y_pos, marker=markers[model_name], color=colors[model_name],
                    markersize=7, zorder=5)
            ax.plot([lo, hi], [y_pos, y_pos], color=colors[model_name],
                    linewidth=2, zorder=4)

            # Significance annotation
            if p_val < 0.001:
                ax.annotate("***", xy=(hi + 0.02, y_pos), fontsize=7, va="center")
            elif p_val < 0.01:
                ax.annotate("**", xy=(hi + 0.02, y_pos), fontsize=7, va="center")
            elif p_val < 0.05:
                ax.annotate("*", xy=(hi + 0.02, y_pos), fontsize=7, va="center")

            y_labels.append(model_name)
            y_positions.append(y_pos)
            if ci == 0:
                n_in_first_group += 1
            y_pos -= 1

        y_pos -= group_gap

    # Reference line at 0 (HR=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("log(Hazard Ratio)", fontsize=9)

    # Divider line between groups
    if n_in_first_group > 0 and len(y_positions) > n_in_first_group:
        divider_y = (y_positions[n_in_first_group - 1] + y_positions[n_in_first_group]) / 2
        ax.axhline(y=divider_y, color="#cccccc", linestyle="-", linewidth=0.5, zorder=0)

    # Group labels on the left
    if n_in_first_group > 0:
        contrib_mid = np.mean(y_positions[:n_in_first_group])
        ax.annotate("log(Contributors)", xy=(-0.22, contrib_mid),
                    xycoords=("axes fraction", "data"),
                    fontsize=8, fontweight="bold", ha="center", va="center", rotation=90)
    if len(y_positions) > n_in_first_group:
        stars_mid = np.mean(y_positions[n_in_first_group:])
        ax.annotate("log(Stars)", xy=(-0.22, stars_mid),
                    xycoords=("axes fraction", "data"),
                    fontsize=8, fontweight="bold", ha="center", va="center", rotation=90)

    # Legend
    legend_elements = [plt.Line2D([0], [0], marker=markers[n], color=colors[n],
                                  label=n, markersize=6, linewidth=1.5)
                       for n in model_order if n == "Pooled" or n in eco_results]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, framealpha=0.9)

    plt.subplots_adjust(left=0.32)
    ax.set_title("")

    p = FIG_DIR / "forest_propagation.pdf"
    fig.savefig(p, bbox_inches="tight")
    pout(f"  Saved {p.name}")

    p_png = FIG_DIR / "forest_propagation.png"
    fig.savefig(p_png, bbox_inches="tight", dpi=300)
    pout(f"  Saved {p_png.name}")
    plt.close(fig)


def main():
    pout("Loading data...")
    combined, surv = build_surv_data()

    results = {}

    # Point 1: Multicollinearity
    compute_multicollinearity(surv, results)

    # Point 1: Joint LRT
    compute_joint_lrt(surv, results)

    # Point 4: Censoring rates
    compute_censoring_rates(combined, results)

    # Point 6: Regenerate forest plot
    regenerate_forest_plot(surv)

    # Save supplementary results
    out = DATA_DIR / "results_supplementary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    pout(f"\nResults saved to {out}")
    pout("Done.")


if __name__ == "__main__":
    main()
