#!/usr/bin/env python3
"""Main cross-ecosystem analysis: reproduces every result in the paper.

Covers:
  RQ1  Patch availability and patch speed (Kaplan-Meier, stratified Cox PH)
  RQ2  Upstream repository health and CVE incidence (negative binomial)
  RQ3  Downstream propagation speed (constraint classification, Cox PH, TOST)

Patch dates come from the dual-source (registry + git tag) metric; propagation
events are classified by the version constraint in force at the upstream patch.

Output: data/results_main.json, paper/fig/*.pdf
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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy.stats import chi2_contingency, norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

# Fixed reference date for the two quantities that need one: the right-censoring
# time of advisories that never received a patch, and repository age. Reading the
# wall clock here instead would make every run produce slightly different
# estimates. The default is the observation cutoff stated in the paper; the data
# freeze is six days later, which is immaterial for an age measured in years.
ANALYSIS_DATE = pd.Timestamp(
    os.environ.get("PATCH_STALL_ANALYSIS_DATE", "2026-03-01"), tz="UTC"
)

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

R: dict = {}  # Accumulated results


def pout(msg):
    print(msg, file=sys.stderr)


def savefig(fig, name):
    p = FIG_DIR / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    pout(f"  Saved {p.name}")


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_base():
    df = pd.read_csv(DATA_DIR / "cross_ecosystem_base.csv")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["cvss_score"] = pd.to_numeric(df["cvss_score"], errors="coerce")
    return df


def load_health():
    p = DATA_DIR / "github_repo_health.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["age_years"] = (ANALYSIS_DATE - df["created_at"]).dt.days / 365.25
    df["contributors_count"] = pd.to_numeric(df["contributors_count"], errors="coerce").fillna(1)
    df["log_contributors"] = np.log1p(df["contributors_count"])
    df["log_stars"] = np.log1p(pd.to_numeric(df["stars"], errors="coerce").fillna(0))
    return df


def load_patch_dates_merged():
    """Load unified dual-source patch dates (registry + git tag)."""
    p = DATA_DIR / "patch_dates_merged.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Use format='mixed' because registry dates have microseconds
    # (e.g. 2023-12-23 17:59:06.771000+00:00) while git_tag dates
    # do not (e.g. 2023-08-10 23:48:37+00:00). Without format='mixed',
    # pandas coerces the shorter format to NaT, losing 9,500+ rows.
    df["patch_date"] = pd.to_datetime(df["patch_date"], format="mixed", utc=True)
    return df


def load_severity_index():
    """Load osv_severity_index.csv for CVSS score lookup."""
    p = DATA_DIR / "osv_severity_index.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["cvss_score"] = pd.to_numeric(df["cvss_score"], errors="coerce")
    return df


def load_propagation(ecosystem, fname):
    """Load a propagation CSV and add ecosystem column."""
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


def build_patch_latency_merged(base, patch_merged):
    """Build patch latency using dual-source merged dates.

    patch_merged has columns: ghsa_id, ecosystem, package_name,
    first_patched_version, github_repo, patch_date, date_source
    """
    m = base.copy()

    if patch_merged is not None and not patch_merged.empty:
        pm = patch_merged[patch_merged["patch_date"].notna()].copy()
        pm = pm.drop_duplicates(subset=["ghsa_id", "package_name", "first_patched_version"])
        m = m.merge(
            pm[["ghsa_id", "package_name", "first_patched_version", "patch_date", "date_source"]],
            on=["ghsa_id", "package_name", "first_patched_version"], how="left"
        )
    else:
        m["patch_date"] = pd.NaT
        m["date_source"] = None

    m["patch_date"] = pd.to_datetime(m["patch_date"], errors="coerce", utc=True)
    m["patch_latency_days"] = (m["patch_date"] - m["published_at"]).dt.total_seconds() / 86400

    # Events: all advisories with resolved patch date (cap 730d)
    # Pre-advisory patches clipped to 0.5 days
    with_date = m[
        m["patch_latency_days"].notna() &
        (m["patch_latency_days"] <= 730)
    ].copy()
    with_date["patch_latency_days"] = with_date["patch_latency_days"].clip(lower=0.5)
    with_date["patch_event"] = 1

    # True censoring: no patch version at all
    no_patch = m[~m["has_patch"]].copy()
    no_patch["patch_latency_days"] = (
        ANALYSIS_DATE - no_patch["published_at"]
    ).dt.total_seconds() / 86400
    no_patch["patch_latency_days"] = no_patch["patch_latency_days"].clip(lower=0.5, upper=730)
    no_patch["patch_event"] = 0

    valid = pd.concat([with_date, no_patch], ignore_index=True)
    valid = valid[valid["patch_latency_days"].notna()]
    return valid


# ══════════════════════════════════════════════════════════════════════
# CONSTRAINT TYPE ABSTRACTION
# ══════════════════════════════════════════════════════════════════════

CONSTRAINT_MAP = {
    # npm
    "caret": "permissive",
    "tilde": "restrictive",
    "exact": "pinned",
    "any": "wildcard",
    # PyPI
    "lower_bound": "permissive",
    "range": "permissive",
    "upper_bound": "permissive",
    "compatible": "restrictive",
    "pinned": "pinned",
    # Maven
    "soft": "pinned",  # Maven bare versions are effectively pinned
    # shared
    "other": "wildcard",
}


def map_constraint_abstract(row):
    """Map ecosystem-specific constraint_type to abstract category."""
    ct = row.get("constraint_type", "")
    eco = row.get("ecosystem", "")
    if ct in CONSTRAINT_MAP:
        return CONSTRAINT_MAP[ct]
    # npm range
    if eco == "npm" and ct == "range":
        return "wildcard"
    # Maven range
    if eco == "Maven" and ct == "range":
        return "permissive"
    return "wildcard"


# ══════════════════════════════════════════════════════════════════════
# PART 1: RQ1 - PATCH AVAILABILITY AND SPEED
# ══════════════════════════════════════════════════════════════════════

def rq1_patch_availability(base):
    """Chi-squared test for patch availability across ecosystems x severity."""
    pout("\n" + "=" * 70)
    pout("RQ1: CROSS-ECOSYSTEM PATCH AVAILABILITY")
    pout("=" * 70)

    ct = pd.crosstab(base["ecosystem"], base["has_patch"])
    chi2, p, dof, _ = chi2_contingency(ct)
    pout(f"\n  Chi-squared: chi2={chi2:.1f}, p={p:.2e}, dof={dof}")
    R["rq1_chi2"] = {"chi2": round(chi2, 1), "p": float(p), "dof": dof}

    # Chi-squared across ecosystem x severity
    ct_sev = pd.crosstab(
        [base["ecosystem"], base["severity_label"]], base["has_patch"]
    )
    chi2_sev, p_sev, dof_sev, _ = chi2_contingency(ct_sev)
    pout(f"  Eco x Severity chi2: chi2={chi2_sev:.1f}, p={p_sev:.2e}, dof={dof_sev}")
    R["rq1_chi2_eco_sev"] = {"chi2": round(chi2_sev, 1), "p": float(p_sev), "dof": dof_sev}

    pout(f"\n  {'Severity':<12} {'npm':>8} {'PyPI':>8} {'Maven':>8} {'Overall':>8}")
    pout(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    patch_rates = {}
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "Overall"]:
        sub = base if sev == "Overall" else base[base["severity_label"] == sev]
        rates = {}
        for eco in ["npm", "PyPI", "Maven"]:
            eco_sub = sub[sub["ecosystem"] == eco]
            rates[eco] = eco_sub["has_patch"].mean() if len(eco_sub) > 0 else 0
        rates["Overall"] = sub["has_patch"].mean()
        patch_rates[sev] = rates
        pout(f"  {sev:<12} {rates['npm']:>7.1%} {rates['PyPI']:>7.1%} "
             f"{rates['Maven']:>7.1%} {rates['Overall']:>7.1%}")
    R["rq1_patch_rates"] = {k: {k2: round(v2, 3) for k2, v2 in v.items()}
                            for k, v in patch_rates.items()}

    # Sample sizes
    eco_counts = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_counts[eco] = int(len(base[base["ecosystem"] == eco]))
    R["rq1_sample_sizes"] = eco_counts
    R["rq1_total_advisories"] = len(base)


def rq1_km_by_ecosystem(valid):
    """K-M curves by ecosystem + log-rank tests (NEW: Maven participates)."""
    pout("\n--- K-M by ecosystem (dual-source dates) ---")
    fig, ax = plt.subplots()
    colors = {"npm": "#e74c3c", "PyPI": "#3498db", "Maven": "#2ecc71"}
    km_stats = {}

    for eco in ["npm", "PyPI", "Maven"]:
        sub = valid[valid["ecosystem"] == eco]
        if len(sub) < 30:
            pout(f"  {eco}: only {len(sub)} rows, skipping KM")
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["patch_latency_days"], event_observed=sub["patch_event"], label=eco)
        kmf.plot_survival_function(ax=ax, color=colors[eco], ci_show=True)
        med = kmf.median_survival_time_
        km_stats[eco] = {
            "n": len(sub),
            "events": int(sub["patch_event"].sum()),
            "median": float(med) if np.isfinite(med) else None,
        }
        pout(f"  {eco}: n={len(sub)}, events={sub['patch_event'].sum()}, "
             f"median={'%.1f' % med if np.isfinite(med) else 'inf'}")

    ax.set_xlabel("Days since vulnerability disclosure")
    ax.set_ylabel("Proportion unpatched")
    ax.set_xlim(0, 365)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("")
    savefig(fig, "km_ecosystem")
    R["rq1_km"] = km_stats

    # Log-rank tests
    logrank_results = {}
    for a, b in [("npm", "PyPI"), ("npm", "Maven"), ("PyPI", "Maven")]:
        sa = valid[valid["ecosystem"] == a]
        sb = valid[valid["ecosystem"] == b]
        if len(sa) < 30 or len(sb) < 30:
            continue
        lr = logrank_test(
            sa["patch_latency_days"], sb["patch_latency_days"],
            event_observed_A=sa["patch_event"], event_observed_B=sb["patch_event"]
        )
        pout(f"  Log-rank {a} vs {b}: chi2={lr.test_statistic:.1f}, p={lr.p_value:.4f}")
        logrank_results[f"{a}_vs_{b}"] = {
            "chi2": round(lr.test_statistic, 1),
            "p": round(lr.p_value, 6),
        }
    R["rq1_logrank"] = logrank_results


def rq1_stratified_cox(valid, health):
    """Ecosystem-STRATIFIED Cox PH with dual-source dates."""
    pout("\n--- Ecosystem-stratified Cox PH (dual-source dates) ---")

    cox_df = valid.copy()
    cox_df["is_critical"] = (cox_df["severity_label"] == "CRITICAL").astype(int)
    cox_df["is_high"] = (cox_df["severity_label"] == "HIGH").astype(int)
    cox_df["is_low"] = (cox_df["severity_label"] == "LOW").astype(int)

    covs = ["is_critical", "is_high", "is_low"]

    # Add health
    if not health.empty and "github_repo" in cox_df.columns:
        h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        cox_df = cox_df.merge(h, on="github_repo", how="left")
        cox_df_h = cox_df.dropna(subset=["log_contributors", "log_stars", "age_years"])
        covs_h = covs + ["log_contributors", "log_stars", "age_years"]
    else:
        cox_df_h = cox_df
        covs_h = covs

    cols = covs_h + ["patch_latency_days", "patch_event", "ecosystem"]
    cox_df_h = cox_df_h[cols].dropna()
    cox_df_h = cox_df_h[cox_df_h["patch_latency_days"] > 0]

    if len(cox_df_h) < 100:
        pout("  Not enough data for stratified Cox")
        return

    # Stratified Cox: ecosystem as stratum (separate baselines)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df_h, duration_col="patch_latency_days", event_col="patch_event",
            strata=["ecosystem"])
    pout(f"  N={len(cox_df_h)}, events={cox_df_h['patch_event'].sum()}, "
         f"concordance={cph.concordance_index_:.3f}")

    strat_results = {}
    s = cph.summary
    for cov in covs_h:
        hr = np.exp(s.loc[cov, "coef"])
        p_val = s.loc[cov, "p"]
        lo = np.exp(s.loc[cov, "coef lower 95%"])
        hi = np.exp(s.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        strat_results[cov] = {"HR": round(hr, 3), "p": round(p_val, 4),
                              "CI": [round(lo, 3), round(hi, 3)]}

    R["rq1_stratified_cox"] = {
        "n": len(cox_df_h), "events": int(cox_df_h["patch_event"].sum()),
        "concordance": round(cph.concordance_index_, 3),
        "covariates": strat_results,
    }

    # Per-ecosystem N breakdown
    eco_breakdown = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = cox_df_h[cox_df_h["ecosystem"] == eco]
        eco_breakdown[eco] = {"n": len(eco_sub), "events": int(eco_sub["patch_event"].sum())}
    R["rq1_stratified_cox"]["eco_breakdown"] = eco_breakdown
    for eco, info in eco_breakdown.items():
        pout(f"    {eco}: n={info['n']}, events={info['events']}")

    # Schoenfeld residual tests
    pout("\n--- Schoenfeld tests (patch speed) ---")
    try:
        ph = proportional_hazard_test(cph, cox_df_h, time_transform="rank")
        schoenfeld = {}
        for cov in covs_h:
            chi2_val = ph.summary.loc[cov, "test_statistic"]
            p_val = ph.summary.loc[cov, "p"]
            flag = " ** VIOLATION **" if p_val < 0.05 else ""
            pout(f"    {cov:25s}  chi2={chi2_val:.2f}  p={p_val:.4f}{flag}")
            schoenfeld[cov] = {"chi2": round(chi2_val, 3), "p": round(p_val, 4),
                               "violation": bool(p_val < 0.05)}
        R["rq1_schoenfeld"] = schoenfeld
    except Exception as e:
        pout(f"  Schoenfeld test failed: {e}")

    # Forest plot
    fig, ax = plt.subplots(figsize=(3.4, max(2.5, len(covs_h) * 0.4)))
    cph.plot(ax=ax)
    ax.set_title("")
    ax.set_xlabel("log(Hazard Ratio)")
    savefig(fig, "forest_patch_stratified")


# ══════════════════════════════════════════════════════════════════════
# PART 2: RQ2 - CVE INCIDENCE (Negative Binomial)
# ══════════════════════════════════════════════════════════════════════

def rq2_nb_regression(base, health):
    """Negative binomial regression with ecosystem dummies."""
    pout("\n" + "=" * 70)
    pout("RQ2: HEALTH -> CVE INCIDENCE (NB regression)")
    pout("=" * 70)

    if health.empty:
        pout("  No health data")
        return

    import statsmodels.api as sm

    merged = base[base["has_repo"]].merge(
        health.rename(columns={"repo": "github_repo"}),
        on="github_repo", how="inner"
    )
    repo_cves = merged.groupby("github_repo").agg(
        cve_count=("cve_id", "nunique"),
        stars=("stars", "first"),
        contributors_count=("contributors_count", "first"),
        age_years=("age_years", "first"),
        ecosystem=("ecosystem", "first"),
    ).reset_index()

    repo_cves["log_stars"] = np.log1p(pd.to_numeric(repo_cves["stars"], errors="coerce").fillna(0))
    repo_cves["log_contribs"] = np.log1p(pd.to_numeric(repo_cves["contributors_count"], errors="coerce").fillna(0))
    repo_cves["is_pypi"] = (repo_cves["ecosystem"] == "PyPI").astype(int)
    repo_cves["is_maven"] = (repo_cves["ecosystem"] == "Maven").astype(int)
    repo_cves = repo_cves.dropna(subset=["log_stars", "log_contribs", "age_years"])

    if len(repo_cves) < 50:
        pout("  Not enough repos")
        return

    # Model 1: Without ecosystem dummies
    pout("\n--- Model 1: Without ecosystem dummies ---")
    X1 = repo_cves[["log_stars", "log_contribs", "age_years"]].copy()
    X1 = sm.add_constant(X1)
    y = repo_cves["cve_count"]
    nb1 = sm.GLM(y, X1, family=sm.families.NegativeBinomial()).fit()
    pout(f"  N={len(repo_cves)}, AIC={nb1.aic:.1f}")

    m1_results = {}
    for var in ["log_stars", "log_contribs", "age_years"]:
        irr = np.exp(nb1.params[var])
        p_val = nb1.pvalues[var]
        ci_lo = np.exp(nb1.conf_int().loc[var, 0])
        ci_hi = np.exp(nb1.conf_int().loc[var, 1])
        pout(f"    {var:20s}  IRR={irr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_val:.4f}")
        m1_results[var] = {"IRR": round(irr, 3), "p": round(p_val, 4),
                           "CI": [round(ci_lo, 3), round(ci_hi, 3)]}

    # Model 2: With ecosystem dummies
    pout("\n--- Model 2: With ecosystem dummies ---")
    X2 = repo_cves[["log_stars", "log_contribs", "age_years", "is_pypi", "is_maven"]].copy()
    X2 = sm.add_constant(X2)
    nb2 = sm.GLM(y, X2, family=sm.families.NegativeBinomial()).fit()
    pout(f"  N={len(repo_cves)}, AIC={nb2.aic:.1f}")

    nb2_results = {}
    for var in ["log_stars", "log_contribs", "age_years", "is_pypi", "is_maven"]:
        irr = np.exp(nb2.params[var])
        p_val = nb2.pvalues[var]
        ci_lo = np.exp(nb2.conf_int().loc[var, 0])
        ci_hi = np.exp(nb2.conf_int().loc[var, 1])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {var:20s}  IRR={irr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_val:.4f} {sig}")
        nb2_results[var] = {"IRR": round(irr, 3), "p": round(p_val, 4),
                            "CI": [round(ci_lo, 3), round(ci_hi, 3)]}

    pout(f"\n  AIC comparison: without eco={nb1.aic:.1f}, with eco={nb2.aic:.1f}")
    R["rq2_nb"] = {
        "n": len(repo_cves),
        "model1_aic": round(nb1.aic, 1),
        "model1_covariates": m1_results,
        "model2_aic": round(nb2.aic, 1),
        "model2_covariates": nb2_results,
    }

    # Health distribution figure
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    repos = merged.drop_duplicates("github_repo")
    colors = {"npm": "#e74c3c", "PyPI": "#3498db", "Maven": "#2ecc71"}
    for i, (col, label) in enumerate([
        ("stars", "Stars (log scale)"),
        ("contributors_count", "Contributors"),
        ("age_years", "Repository age (years)"),
    ]):
        ax = axes[i]
        for eco in ["npm", "PyPI", "Maven"]:
            vals = pd.to_numeric(repos[repos["ecosystem"] == eco][col], errors="coerce").dropna()
            if len(vals) > 0:
                ax.hist(np.log1p(vals) if col == "stars" else vals,
                        bins=30, alpha=0.5, label=eco, color=colors[eco], density=True)
        ax.set_xlabel(label)
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "health_dist")


# ══════════════════════════════════════════════════════════════════════
# PART 3: RQ3 - CROSS-ECOSYSTEM PROPAGATION
# ══════════════════════════════════════════════════════════════════════

def rq3_combine_propagation(sev_index):
    """Load and combine all 3 ecosystem propagation CSVs."""
    pout("\n" + "=" * 70)
    pout("RQ3: CROSS-ECOSYSTEM PROPAGATION")
    pout("=" * 70)

    npm = load_propagation("npm", "npm_propagation.csv")
    pypi = load_propagation("PyPI", "pypi_propagation.csv")
    maven = load_propagation("Maven", "maven_propagation.csv")

    pout(f"  npm: {len(npm)}, PyPI: {len(pypi)}, Maven: {len(maven)}")

    combined = pd.concat([npm, pypi, maven], ignore_index=True)
    pout(f"  Combined: {len(combined)}")

    # Map constraint_abstract
    combined["constraint_abstract"] = combined.apply(map_constraint_abstract, axis=1)

    # Add CVSS score from osv_severity_index
    if not sev_index.empty:
        sev_lu = sev_index[["osv_id", "cvss_score"]].drop_duplicates("osv_id")
        combined = combined.merge(sev_lu, on="osv_id", how="left", suffixes=("_raw", ""))
        # If cvss_score_raw existed from the propagation CSV, drop it
        if "cvss_score_raw" in combined.columns:
            combined = combined.drop(columns=["cvss_score_raw"])
        combined["cvss_score"] = pd.to_numeric(combined["cvss_score"], errors="coerce")

    R["rq3_combined_n"] = {
        "total": len(combined),
        "npm": len(npm), "PyPI": len(pypi), "Maven": len(maven),
    }

    return combined


def rq3_autoprop_rates(combined):
    """Auto-propagation rates by ecosystem and constraint type."""
    pout("\n--- Auto-propagation rates ---")

    # Overall by ecosystem
    pout("\n  By ecosystem:")
    eco_rates = {}
    for eco in ["npm", "PyPI", "Maven"]:
        sub = combined[combined["ecosystem"] == eco]
        n_total = len(sub)
        n_auto = (sub["propagation_type"] == "auto").sum()
        n_manual = (sub["propagation_type"] == "manual").sum()
        n_censored = (sub["propagation_type"] == "censored").sum()
        rate = n_auto / n_total if n_total > 0 else 0
        pout(f"    {eco:8s}  auto={n_auto:>5} ({rate:.1%})  "
             f"manual={n_manual:>5}  censored={n_censored:>5}  total={n_total}")
        eco_rates[eco] = {
            "auto": int(n_auto), "manual": int(n_manual),
            "censored": int(n_censored), "total": int(n_total),
            "auto_rate": round(rate, 4),
        }
    R["rq3_autoprop_by_eco"] = eco_rates

    # By abstract constraint category x ecosystem
    pout("\n  By constraint_abstract x ecosystem:")
    cross_rates = {}
    for cat in ["permissive", "restrictive", "pinned", "wildcard"]:
        cross_rates[cat] = {}
        for eco in ["npm", "PyPI", "Maven"]:
            sub = combined[(combined["constraint_abstract"] == cat) &
                           (combined["ecosystem"] == eco)]
            n = len(sub)
            if n > 0:
                auto_rate = (sub["propagation_type"] == "auto").mean()
                cross_rates[cat][eco] = {"n": int(n), "auto_rate": round(auto_rate, 4)}
                pout(f"    {cat:12s} x {eco:6s}: n={n:>5}, auto={auto_rate:.1%}")
            else:
                cross_rates[cat][eco] = {"n": 0, "auto_rate": None}
    R["rq3_autoprop_cross"] = cross_rates

    # Figure: auto-propagation rates by ecosystem and constraint
    fig, ax = plt.subplots(figsize=(5, 3))
    categories = ["permissive", "restrictive", "pinned", "wildcard"]
    x = np.arange(len(categories))
    width = 0.25
    colors = {"npm": "#e74c3c", "PyPI": "#3498db", "Maven": "#2ecc71"}

    for i, eco in enumerate(["npm", "PyPI", "Maven"]):
        rates = []
        for cat in categories:
            info = cross_rates[cat].get(eco, {})
            r = info.get("auto_rate", 0)
            rates.append(r if r is not None else 0)
        bars = ax.bar(x + i * width, rates, width, label=eco, color=colors[eco], alpha=0.8)
        # Add N labels on bars
        for j, bar in enumerate(bars):
            n = cross_rates[categories[j]].get(eco, {}).get("n", 0)
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"n={n}", ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_xlabel("Constraint category")
    ax.set_ylabel("Auto-propagation rate")
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    savefig(fig, "autoprop_cross_eco")


def rq3_propagation_cox(combined, base, health, sev_index):
    """Cross-ecosystem propagation Cox PH analysis."""
    pout("\n--- Cross-ecosystem propagation Cox PH ---")

    # Filter to non-automatic events (manual + censored)
    surv = combined[combined["propagation_type"].isin(["manual", "censored"])].copy()
    surv = surv[surv["propagation_days"].notna() & (surv["propagation_days"] > 0)]
    surv["prop_event"] = (surv["propagation_type"] == "manual").astype(int)

    pout(f"  Survival subset: n={len(surv)}, events={surv['prop_event'].sum()}")
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = surv[surv["ecosystem"] == eco]
        pout(f"    {eco}: n={len(eco_sub)}, events={eco_sub['prop_event'].sum()}")

    # Covariates
    surv["is_runtime"] = (surv["dependency_type"] == "dependencies").astype(int)
    surv["requires_major"] = surv["requires_major_upgrade"]
    surv["cvss_score"] = pd.to_numeric(surv["cvss_score"], errors="coerce")
    surv["cvss_score"] = surv["cvss_score"].fillna(surv["cvss_score"].median())

    # Constraint abstract dummies (ref = pinned)
    surv["is_permissive"] = (surv["constraint_abstract"] == "permissive").astype(int)
    surv["is_restrictive"] = (surv["constraint_abstract"] == "restrictive").astype(int)
    # wildcard folded into reference (pinned) since very few events

    # Add upstream health via osv_id -> base -> github_repo -> health
    # Link: propagation osv_id -> base ghsa_id -> github_repo
    if not health.empty:
        # osv_id is the same as ghsa_id in the base dataset
        base_repo = base[["ghsa_id", "github_repo"]].drop_duplicates("ghsa_id")
        surv = surv.merge(base_repo, left_on="osv_id", right_on="ghsa_id", how="left")
        h = health[["repo", "log_contributors", "log_stars", "age_years"]].copy()
        h = h.rename(columns={"repo": "github_repo"})
        # Avoid duplicate github_repo columns
        if "github_repo_x" in surv.columns:
            surv["github_repo"] = surv["github_repo_x"].fillna(surv.get("github_repo_y"))
            surv = surv.drop(columns=["github_repo_x", "github_repo_y"], errors="ignore")
        surv = surv.merge(h, on="github_repo", how="left")
        health_n = surv["log_contributors"].notna().sum()
        pout(f"  Events with upstream health: {health_n}/{len(surv)}")

    # ── K-M by ecosystem ──
    pout("\n--- K-M propagation by ecosystem ---")
    fig, ax = plt.subplots()
    colors = {"npm": "#e74c3c", "PyPI": "#3498db", "Maven": "#2ecc71"}
    km_prop_stats = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = surv[surv["ecosystem"] == eco]
        if len(eco_sub) < 30:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(eco_sub["propagation_days"], event_observed=eco_sub["prop_event"], label=eco)
        kmf.plot_survival_function(ax=ax, color=colors[eco], ci_show=True)
        med = kmf.median_survival_time_
        km_prop_stats[eco] = {
            "n": len(eco_sub), "events": int(eco_sub["prop_event"].sum()),
            "median": float(med) if np.isfinite(med) else None,
        }
        pout(f"  {eco}: n={len(eco_sub)}, events={eco_sub['prop_event'].sum()}, "
             f"median={'%.1f' % med if np.isfinite(med) else 'inf'}")
    ax.set_xlabel("Days since upstream patch release")
    ax.set_ylabel("Proportion not yet updated")
    ax.set_xlim(0, 365)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("")
    savefig(fig, "km_propagation")
    R["rq3_km_propagation"] = km_prop_stats

    # Log-rank tests for propagation
    logrank_prop = {}
    for a, b in [("npm", "PyPI"), ("npm", "Maven"), ("PyPI", "Maven")]:
        sa = surv[surv["ecosystem"] == a]
        sb = surv[surv["ecosystem"] == b]
        if len(sa) < 30 or len(sb) < 30:
            continue
        lr = logrank_test(
            sa["propagation_days"], sb["propagation_days"],
            event_observed_A=sa["prop_event"], event_observed_B=sb["prop_event"]
        )
        pout(f"  Log-rank {a} vs {b}: chi2={lr.test_statistic:.1f}, p={lr.p_value:.6f}")
        logrank_prop[f"{a}_vs_{b}"] = {
            "chi2": round(lr.test_statistic, 1),
            "p": round(lr.p_value, 6),
        }
    R["rq3_logrank_propagation"] = logrank_prop

    # ── Model A: Structural only (ecosystem-stratified) ──
    pout("\n--- Model A: Structural covariates (ecosystem-stratified) ---")
    covs_a = ["is_permissive", "is_restrictive", "is_runtime",
              "requires_major", "cvss_score"]

    # Use osv_id as cluster variable for all ecosystems
    df_a = surv[covs_a + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
    df_a = df_a[df_a["propagation_days"] > 0]

    if len(df_a) < 100:
        pout("  Not enough data for Model A")
        return

    cph_a = CoxPHFitter(penalizer=0.01)
    cph_a.fit(df_a, duration_col="propagation_days", event_col="prop_event",
              strata=["ecosystem"], cluster_col="osv_id")
    pout(f"  N={len(df_a)}, events={df_a['prop_event'].sum()}, "
         f"concordance={cph_a.concordance_index_:.3f}")

    model_a_results = {}
    sa = cph_a.summary
    for cov in covs_a:
        hr = np.exp(sa.loc[cov, "coef"])
        p_val = sa.loc[cov, "p"]
        lo = np.exp(sa.loc[cov, "coef lower 95%"])
        hi = np.exp(sa.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        model_a_results[cov] = {"HR": round(hr, 3), "p": round(p_val, 4),
                                "CI": [round(lo, 3), round(hi, 3)]}

    R["rq3_model_a"] = {
        "n": len(df_a), "events": int(df_a["prop_event"].sum()),
        "concordance": round(cph_a.concordance_index_, 3),
        "covariates": model_a_results,
    }

    # ── Model B: + health covariates (ecosystem-stratified) ──
    pout("\n--- Model B: + upstream health covariates (ecosystem-stratified) ---")
    covs_b = covs_a + ["log_contributors", "log_stars"]
    df_b = surv[covs_b + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].dropna()
    df_b = df_b[df_b["propagation_days"] > 0]

    if len(df_b) < 100 or df_b["prop_event"].sum() < 20:
        pout(f"  Not enough data for Model B (n={len(df_b)}, events={df_b['prop_event'].sum()})")
        return

    cph_b = CoxPHFitter(penalizer=0.01)
    cph_b.fit(df_b, duration_col="propagation_days", event_col="prop_event",
              strata=["ecosystem"], cluster_col="osv_id")
    pout(f"  N={len(df_b)}, events={df_b['prop_event'].sum()}, "
         f"concordance={cph_b.concordance_index_:.3f}")

    model_b_results = {}
    sb = cph_b.summary
    for cov in covs_b:
        hr = np.exp(sb.loc[cov, "coef"])
        p_val = sb.loc[cov, "p"]
        lo = np.exp(sb.loc[cov, "coef lower 95%"])
        hi = np.exp(sb.loc[cov, "coef upper 95%"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f} {sig}")
        model_b_results[cov] = {"HR": round(hr, 3), "p": round(p_val, 4),
                                "CI": [round(lo, 3), round(hi, 3)]}

    # Model A restricted to same sample as B for fair comparison
    pout("\n--- Model A (restricted to Model B sample) ---")
    df_a_restr = df_b[covs_a + ["propagation_days", "prop_event", "ecosystem", "osv_id"]].copy()
    cph_a_restr = CoxPHFitter(penalizer=0.01)
    cph_a_restr.fit(df_a_restr, duration_col="propagation_days", event_col="prop_event",
                     strata=["ecosystem"], cluster_col="osv_id")
    pout(f"  N={len(df_a_restr)}, events={int(df_a_restr['prop_event'].sum())}, "
         f"concordance={cph_a_restr.concordance_index_:.3f}")

    model_a_restr_results = {}
    sa_r = cph_a_restr.summary
    for cov in covs_a:
        hr = np.exp(sa_r.loc[cov, "coef"])
        p_val = sa_r.loc[cov, "p"]
        lo = np.exp(sa_r.loc[cov, "coef lower 95%"])
        hi = np.exp(sa_r.loc[cov, "coef upper 95%"])
        pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f}")
        model_a_restr_results[cov] = {"HR": round(hr, 3), "p": round(p_val, 4),
                                       "CI": [round(lo, 3), round(hi, 3)]}

    pout(f"\n  Concordance: A(restr)={cph_a_restr.concordance_index_:.3f} -> "
         f"B={cph_b.concordance_index_:.3f}")

    R["rq3_model_b"] = {
        "n": len(df_b), "events": int(df_b["prop_event"].sum()),
        "concordance": round(cph_b.concordance_index_, 3),
        "covariates": model_b_results,
    }
    R["rq3_model_a_restricted"] = {
        "n": len(df_a_restr), "events": int(df_a_restr["prop_event"].sum()),
        "concordance": round(cph_a_restr.concordance_index_, 3),
        "covariates": model_a_restr_results,
    }

    # Per-ecosystem breakdown for Model B
    eco_b_breakdown = {}
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = df_b[df_b["ecosystem"] == eco]
        eco_b_breakdown[eco] = {"n": len(eco_sub), "events": int(eco_sub["prop_event"].sum())}
    R["rq3_model_b"]["eco_breakdown"] = eco_b_breakdown
    for eco, info in eco_b_breakdown.items():
        pout(f"    {eco}: n={info['n']}, events={info['events']}")

    # Schoenfeld tests for propagation Model B
    pout("\n--- Schoenfeld tests (propagation Model B) ---")
    try:
        ph_prop = proportional_hazard_test(cph_b, df_b, time_transform="rank")
        schoenfeld_prop = {}
        for cov in covs_b:
            chi2_val = ph_prop.summary.loc[cov, "test_statistic"]
            p_val = ph_prop.summary.loc[cov, "p"]
            flag = " ** VIOLATION **" if p_val < 0.05 else ""
            pout(f"    {cov:25s}  chi2={chi2_val:.2f}  p={p_val:.4f}{flag}")
            schoenfeld_prop[cov] = {"chi2": round(chi2_val, 3), "p": round(p_val, 4),
                                     "violation": bool(p_val < 0.05)}
        R["rq3_schoenfeld"] = schoenfeld_prop
    except Exception as e:
        pout(f"  Schoenfeld test failed: {e}")

    # Forest plot for Model B
    fig, ax = plt.subplots(figsize=(3.4, max(2.5, len(covs_b) * 0.35)))
    cph_b.plot(ax=ax)
    ax.set_title("")
    ax.set_xlabel("log(Hazard Ratio)")
    savefig(fig, "forest_propagation")

    # ── TOST equivalence test on health covariates ──
    _tost_equivalence(sb, df_b, covs_b, surv)

    # ── Per-ecosystem TOST ──
    _per_ecosystem_tost(surv, covs_a, covs_b)


def _tost_equivalence(sb, df_b, covs_b, surv):
    """TOST equivalence test for health covariates in pooled model."""
    pout("\n--- TOST equivalence test (pooled cross-ecosystem) ---")
    health_covs = ["log_contributors", "log_stars"]
    n_events_b = int(df_b["prop_event"].sum())
    tost_results = {}

    for delta_hr in [0.25, 0.20, 0.15]:
        delta = np.log(1 + delta_hr)
        for cov in health_covs:
            beta = sb.loc[cov, "coef"]
            se = sb.loc[cov, "se(coef)"]
            hr = np.exp(beta)
            ci90_lo = beta - 1.645 * se
            ci90_hi = beta + 1.645 * se
            z1 = (beta - (-delta)) / se
            z2 = (delta - beta) / se
            p1 = 1 - norm.cdf(z1)
            p2 = 1 - norm.cdf(z2)
            p_tost = max(p1, p2)
            equiv = (ci90_lo > -delta) and (ci90_hi < delta)
            key = f"pooled_{cov}_delta{int(delta_hr*100)}"
            tost_results[key] = {
                "HR": round(hr, 3), "beta": round(beta, 4), "SE": round(se, 4),
                "CI90": [round(np.exp(ci90_lo), 3), round(np.exp(ci90_hi), 3)],
                "p_tost": round(p_tost, 4), "equivalent": bool(equiv),
                "delta_hr": delta_hr,
            }
            flag = "EQUIV" if equiv else "NOT equiv"
            pout(f"  pooled {cov} delta={int(delta_hr*100)}%: HR={hr:.3f} "
                 f"90%CI=[{np.exp(ci90_lo):.3f},{np.exp(ci90_hi):.3f}] "
                 f"p_tost={p_tost:.4f} {flag}")

    # Post-hoc power analysis
    pout("\n--- Post-hoc power analysis (pooled) ---")
    power_results = {}
    for cov in health_covs:
        sigma = df_b[cov].std()
        se = sb.loc[cov, "se(coef)"]
        z_sum = 1.96 + 0.842  # 80% power, two-sided alpha=0.05
        min_loghr = z_sum / np.sqrt(n_events_b * sigma**2)
        min_hr_upper = np.exp(min_loghr)
        min_hr_lower = np.exp(-min_loghr)
        events_needed_109 = int(np.ceil((z_sum**2) / (sigma**2 * np.log(1.09)**2)))
        events_needed_080 = int(np.ceil((z_sum**2) / (sigma**2 * np.log(0.80)**2)))
        pout(f"  {cov}: min detectable HR=[{min_hr_lower:.3f},{min_hr_upper:.3f}], "
             f"events for HR=1.09: {events_needed_109}, for HR=0.80: {events_needed_080}")
        power_results[cov] = {
            "sd": round(sigma, 3),
            "min_detectable_hr_upper": round(min_hr_upper, 3),
            "min_detectable_hr_lower": round(min_hr_lower, 3),
            "events_needed_hr109": events_needed_109,
            "events_needed_hr080": events_needed_080,
        }

    R["rq3_tost_pooled"] = tost_results
    R["rq3_power_pooled"] = power_results


def _per_ecosystem_tost(surv, covs_a, covs_b):
    """Run TOST separately for each ecosystem."""
    pout("\n--- Per-ecosystem TOST tests ---")
    health_covs = ["log_contributors", "log_stars"]
    per_eco_tost = {}

    for eco in ["npm", "PyPI", "Maven"]:
        pout(f"\n  === {eco} ===")
        eco_surv = surv[surv["ecosystem"] == eco].copy()

        # Drop covariates with near-zero variance for this ecosystem
        # (e.g., Maven has 0 restrictive constraints, PyPI has very few permissive)
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

        n_events = int(df_eco["prop_event"].sum())
        pout(f"    n={len(df_eco)}, events={n_events}, covariates={eco_covs}")

        if len(df_eco) < 50 or n_events < 10:
            pout(f"    Insufficient data for per-ecosystem Cox")
            per_eco_tost[eco] = {"insufficient_data": True, "n": len(df_eco), "events": n_events}
            continue

        try:
            cph_eco = CoxPHFitter(penalizer=0.01)
            cph_eco.fit(df_eco, duration_col="propagation_days", event_col="prop_event",
                        cluster_col="osv_id")
            se_eco = cph_eco.summary

            eco_tost = {"n": len(df_eco), "events": n_events,
                        "concordance": round(cph_eco.concordance_index_, 3),
                        "covariates_used": eco_covs}
            for cov in eco_covs:
                if cov in se_eco.index:
                    hr = np.exp(se_eco.loc[cov, "coef"])
                    p_val = se_eco.loc[cov, "p"]
                    lo = np.exp(se_eco.loc[cov, "coef lower 95%"])
                    hi = np.exp(se_eco.loc[cov, "coef upper 95%"])
                    pout(f"    {cov:25s}  HR={hr:.3f} [{lo:.3f}, {hi:.3f}]  p={p_val:.4f}")
                    eco_tost[f"coef_{cov}"] = {
                        "HR": round(hr, 3), "p": round(p_val, 4),
                        "CI": [round(lo, 3), round(hi, 3)],
                    }

            # TOST for health covariates
            for delta_hr in [0.25, 0.20]:
                delta = np.log(1 + delta_hr)
                for cov in health_covs:
                    if cov not in se_eco.index:
                        continue
                    beta = se_eco.loc[cov, "coef"]
                    se = se_eco.loc[cov, "se(coef)"]
                    hr = np.exp(beta)
                    ci90_lo = beta - 1.645 * se
                    ci90_hi = beta + 1.645 * se
                    z1 = (beta - (-delta)) / se
                    z2 = (delta - beta) / se
                    p1 = 1 - norm.cdf(z1)
                    p2 = 1 - norm.cdf(z2)
                    p_tost = max(p1, p2)
                    equiv = (ci90_lo > -delta) and (ci90_hi < delta)
                    key = f"{cov}_delta{int(delta_hr*100)}"
                    eco_tost[key] = {
                        "HR": round(hr, 3), "beta": round(beta, 4), "SE": round(se, 4),
                        "CI90": [round(np.exp(ci90_lo), 3), round(np.exp(ci90_hi), 3)],
                        "p_tost": round(p_tost, 4), "equivalent": bool(equiv),
                    }
                    flag = "EQUIV" if equiv else "NOT equiv"
                    pout(f"    TOST {cov} delta={int(delta_hr*100)}%: "
                         f"HR={hr:.3f} 90%CI=[{np.exp(ci90_lo):.3f},{np.exp(ci90_hi):.3f}] "
                         f"p={p_tost:.4f} {flag}")

            per_eco_tost[eco] = eco_tost
        except Exception as e:
            pout(f"    Per-ecosystem Cox failed for {eco}: {e}")
            per_eco_tost[eco] = {"error": str(e)}

    R["rq3_tost_per_ecosystem"] = per_eco_tost


# ══════════════════════════════════════════════════════════════════════
# CONSTRAINT ABSTRACTION SUMMARY
# ══════════════════════════════════════════════════════════════════════

def constraint_abstraction_summary(combined):
    """Report the mapping from raw constraint types to abstract categories."""
    pout("\n--- Constraint type abstraction mapping ---")
    mapping_table = combined.groupby(["ecosystem", "constraint_type", "constraint_abstract"]).size()
    mapping_table = mapping_table.reset_index(name="count")
    pout(mapping_table.to_string(index=False))

    # Summary by abstract category
    summary = combined.groupby("constraint_abstract").size().to_dict()
    pout(f"\n  Abstract category totals: {summary}")
    R["rq3_constraint_mapping"] = {
        eco: combined[combined["ecosystem"] == eco].groupby(
            ["constraint_type", "constraint_abstract"]
        ).size().reset_index(name="count").to_dict("records")
        for eco in ["npm", "PyPI", "Maven"]
    }
    R["rq3_constraint_abstract_totals"] = {k: int(v) for k, v in summary.items()}


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
    pout(f"  Severity index: {len(sev_index)} entries")

    # ── PART 1: RQ1 ──
    rq1_patch_availability(base)

    # Build patch latency with dual-source dates
    valid = build_patch_latency_merged(base, patch_merged)
    pout(f"\n  Patch latency dataset: {len(valid)} rows "
         f"({valid['patch_event'].sum()} events, "
         f"{(~valid['patch_event'].astype(bool)).sum()} censored)")

    # Per-ecosystem patch latency breakdown
    for eco in ["npm", "PyPI", "Maven"]:
        eco_sub = valid[valid["ecosystem"] == eco]
        pout(f"    {eco}: {len(eco_sub)} rows, {eco_sub['patch_event'].sum()} events")

    # Date source breakdown
    if "date_source" in valid.columns:
        source_counts = valid[valid["patch_event"] == 1].groupby(
            ["ecosystem", "date_source"]
        ).size()
        pout(f"\n  Date source breakdown (events only):\n{source_counts.to_string()}")
        R["rq1_date_source_breakdown"] = {
            f"{eco}_{src}": int(cnt)
            for (eco, src), cnt in source_counts.items()
        }

    rq1_km_by_ecosystem(valid)
    rq1_stratified_cox(valid, health)

    # ── PART 2: RQ2 ──
    rq2_nb_regression(base, health)

    # ── PART 3: RQ3 ──
    combined = rq3_combine_propagation(sev_index)
    constraint_abstraction_summary(combined)
    rq3_autoprop_rates(combined)
    rq3_propagation_cox(combined, base, health, sev_index)

    # ── Save results ──
    out = DATA_DIR / "results_main.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved to {out}")
    pout("Done.")


if __name__ == "__main__":
    main()
