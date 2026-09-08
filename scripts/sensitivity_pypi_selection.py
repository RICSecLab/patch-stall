#!/usr/bin/env python3
"""Within-PyPI constraint selection bias analysis.

Compares upstream project characteristics between packages using
permissive constraints (>=, range) vs pinned constraints (==)
to detect systematic selection bias.

Output: data/sensitivity_pypi_selection.json
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
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_cross_ecosystem import (
    load_base, load_health, load_propagation,
    load_severity_index,
)


def pout(msg):
    print(msg, file=sys.stderr)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (m1 - m2) / s_pooled


def smd(group1, group2):
    """Compute standardized mean difference (absolute value of Cohen's d).

    SMD < 0.1: balanced
    0.1 <= SMD < 0.25: mild imbalance
    SMD >= 0.25: substantial imbalance
    """
    d = cohens_d(group1, group2)
    return abs(d) if not np.isnan(d) else np.nan


def smd_label(val):
    """Classify SMD into balance categories."""
    if np.isnan(val):
        return "N/A"
    if val < 0.1:
        return "balanced"
    elif val < 0.25:
        return "mild_imbalance"
    else:
        return "substantial_imbalance"


def compare_groups(permissive, pinned, col, col_label):
    """Compare a single covariate between permissive and pinned groups."""
    perm_vals = permissive[col].dropna()
    pin_vals = pinned[col].dropna()

    if len(perm_vals) < 5 or len(pin_vals) < 5:
        pout(f"    {col_label}: insufficient data "
             f"(permissive={len(perm_vals)}, pinned={len(pin_vals)})")
        return {"insufficient_data": True,
                "n_permissive": len(perm_vals),
                "n_pinned": len(pin_vals)}

    # Mann-Whitney U test
    try:
        u_stat, p_val = mannwhitneyu(perm_vals, pin_vals, alternative="two-sided")
    except Exception as e:
        pout(f"    {col_label}: Mann-Whitney failed: {e}")
        return {"error": str(e)}

    # Effect size
    d = cohens_d(perm_vals, pin_vals)
    smd_val = abs(d) if not np.isnan(d) else np.nan
    balance = smd_label(smd_val)

    result = {
        "n_permissive": len(perm_vals),
        "n_pinned": len(pin_vals),
        "permissive_mean": round(float(perm_vals.mean()), 4),
        "permissive_median": round(float(perm_vals.median()), 4),
        "permissive_std": round(float(perm_vals.std()), 4),
        "pinned_mean": round(float(pin_vals.mean()), 4),
        "pinned_median": round(float(pin_vals.median()), 4),
        "pinned_std": round(float(pin_vals.std()), 4),
        "mann_whitney_U": round(float(u_stat), 1),
        "mann_whitney_p": round(float(p_val), 6),
        "cohens_d": round(float(d), 4) if not np.isnan(d) else None,
        "SMD": round(float(smd_val), 4) if not np.isnan(smd_val) else None,
        "balance": balance,
    }

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    pout(f"    {col_label:<25s}  perm_med={perm_vals.median():.3f}  "
         f"pin_med={pin_vals.median():.3f}  U={u_stat:.0f}  p={p_val:.4f}{sig}  "
         f"SMD={smd_val:.3f} [{balance}]")

    return result


def main():
    pout("=" * 70)
    pout("D4: WITHIN-PYPI CONSTRAINT SELECTION BIAS ANALYSIS")
    pout("=" * 70)

    R = {}

    # ── Load data ──
    pout("\nLoading data...")
    base = load_base()
    health = load_health()
    sev_index = load_severity_index()

    pypi = load_propagation("PyPI", "pypi_propagation.csv")
    pout(f"  PyPI propagation events: {len(pypi)}")
    pout(f"  Columns: {list(pypi.columns)}")

    # ── Classify constraint types ──
    pout("\n--- Constraint type classification ---")
    PERMISSIVE_TYPES = {"lower_bound", "range", "upper_bound"}
    PINNED_TYPES = {"pinned"}
    RESTRICTIVE_TYPES = {"compatible"}

    def classify_constraint(ct):
        if ct in PERMISSIVE_TYPES:
            return "permissive"
        elif ct in PINNED_TYPES:
            return "pinned"
        elif ct in RESTRICTIVE_TYPES:
            return "restrictive"
        else:
            return "other"

    pypi["constraint_class"] = pypi["constraint_type"].apply(classify_constraint)

    class_counts = pypi["constraint_class"].value_counts()
    pout(f"  Constraint class distribution:")
    for cls, cnt in class_counts.items():
        pout(f"    {cls}: {cnt} ({cnt/len(pypi)*100:.1f}%)")
    R["constraint_class_counts"] = {k: int(v) for k, v in class_counts.items()}

    # Also show raw constraint_type distribution
    raw_counts = pypi["constraint_type"].value_counts()
    pout(f"\n  Raw constraint_type distribution:")
    for ct, cnt in raw_counts.items():
        pout(f"    {ct}: {cnt}")
    R["raw_constraint_type_counts"] = {k: int(v) for k, v in raw_counts.items()}

    # ── Link with upstream health ──
    pout("\n--- Linking with upstream health metrics ---")

    # PyPI propagation has upstream_package and osv_id
    # base has ghsa_id (= osv_id), github_repo
    # health has repo (= github_repo), stars, contributors_count, age_years
    base_repo = base[["ghsa_id", "github_repo"]].drop_duplicates("ghsa_id")
    pypi_linked = pypi.merge(base_repo, left_on="osv_id", right_on="ghsa_id", how="left")

    n_linked = pypi_linked["github_repo"].notna().sum()
    pout(f"  Events linked to GitHub repo: {n_linked}/{len(pypi_linked)} "
         f"({n_linked/len(pypi_linked)*100:.1f}%)")

    h = health[["repo", "log_contributors", "log_stars", "age_years",
                "stars", "contributors_count"]].copy()
    h = h.rename(columns={"repo": "github_repo"})
    pypi_health = pypi_linked.merge(h, on="github_repo", how="left")

    n_with_health = pypi_health["log_contributors"].notna().sum()
    pout(f"  Events with health data: {n_with_health}/{len(pypi_health)} "
         f"({n_with_health/len(pypi_health)*100:.1f}%)")

    # Add CVSS score
    if not sev_index.empty:
        sev_lu = sev_index[["osv_id", "cvss_score"]].drop_duplicates("osv_id")
        pypi_health = pypi_health.merge(sev_lu, on="osv_id", how="left",
                                         suffixes=("_raw", ""))
        if "cvss_score_raw" in pypi_health.columns:
            pypi_health = pypi_health.drop(columns=["cvss_score_raw"])
        pypi_health["cvss_score"] = pd.to_numeric(pypi_health["cvss_score"],
                                                    errors="coerce")

    # Count downstream dependents per advisory
    dep_counts = pypi_health.groupby("osv_id").size().reset_index(name="n_dependents")
    pypi_health = pypi_health.merge(dep_counts, on="osv_id", how="left")

    R["data_summary"] = {
        "total_pypi_events": len(pypi),
        "linked_to_repo": int(n_linked),
        "with_health_data": int(n_with_health),
    }

    # ── Compare permissive vs pinned ──
    pout("\n" + "=" * 70)
    pout("PERMISSIVE vs PINNED COMPARISON")
    pout("=" * 70)

    permissive = pypi_health[pypi_health["constraint_class"] == "permissive"].copy()
    pinned = pypi_health[pypi_health["constraint_class"] == "pinned"].copy()

    pout(f"\n  Group sizes: permissive={len(permissive)}, pinned={len(pinned)}")
    R["group_sizes"] = {
        "permissive": len(permissive),
        "pinned": len(pinned),
        "restrictive": len(pypi_health[pypi_health["constraint_class"] == "restrictive"]),
        "other": len(pypi_health[pypi_health["constraint_class"] == "other"]),
    }

    # Covariates to compare
    comparisons = {}

    pout("\n  --- Upstream project characteristics ---")

    # log(stars)
    comparisons["log_stars"] = compare_groups(
        permissive, pinned, "log_stars", "log(stars)"
    )

    # log(contributors)
    comparisons["log_contributors"] = compare_groups(
        permissive, pinned, "log_contributors", "log(contributors)"
    )

    # age_years
    comparisons["age_years"] = compare_groups(
        permissive, pinned, "age_years", "age_years"
    )

    pout("\n  --- Advisory characteristics ---")

    # Number of downstream dependents per advisory
    comparisons["n_dependents"] = compare_groups(
        permissive, pinned, "n_dependents", "n_dependents"
    )

    # CVSS score
    comparisons["cvss_score"] = compare_groups(
        permissive, pinned, "cvss_score", "cvss_score"
    )

    R["permissive_vs_pinned"] = comparisons

    # ── Also compare permissive vs restrictive (compatible) ──
    pout("\n" + "=" * 70)
    pout("PERMISSIVE vs RESTRICTIVE (compatible) COMPARISON")
    pout("=" * 70)

    restrictive = pypi_health[pypi_health["constraint_class"] == "restrictive"].copy()
    pout(f"\n  Group sizes: permissive={len(permissive)}, restrictive={len(restrictive)}")

    comparisons_pr = {}

    comparisons_pr["log_stars"] = compare_groups(
        permissive, restrictive, "log_stars", "log(stars)"
    )
    comparisons_pr["log_contributors"] = compare_groups(
        permissive, restrictive, "log_contributors", "log(contributors)"
    )
    comparisons_pr["age_years"] = compare_groups(
        permissive, restrictive, "age_years", "age_years"
    )
    comparisons_pr["n_dependents"] = compare_groups(
        permissive, restrictive, "n_dependents", "n_dependents"
    )
    comparisons_pr["cvss_score"] = compare_groups(
        permissive, restrictive, "cvss_score", "cvss_score"
    )

    R["permissive_vs_restrictive"] = comparisons_pr

    # ── Overall covariate balance summary ──
    pout("\n" + "=" * 70)
    pout("COVARIATE BALANCE SUMMARY (SMD)")
    pout("=" * 70)

    balance_summary = {}
    for comparison_name, comp_data in [
        ("permissive_vs_pinned", comparisons),
        ("permissive_vs_restrictive", comparisons_pr),
    ]:
        pout(f"\n  {comparison_name}:")
        summary = {}
        for cov_name, cov_data in comp_data.items():
            if "SMD" in cov_data and cov_data["SMD"] is not None:
                summary[cov_name] = {
                    "SMD": cov_data["SMD"],
                    "balance": cov_data["balance"],
                }
                pout(f"    {cov_name:<25s}  SMD={cov_data['SMD']:.4f}  [{cov_data['balance']}]")
        balance_summary[comparison_name] = summary

        # Overall assessment
        smd_vals = [v["SMD"] for v in summary.values() if v["SMD"] is not None]
        if smd_vals:
            max_smd = max(smd_vals)
            mean_smd = np.mean(smd_vals)
            n_balanced = sum(1 for v in smd_vals if v < 0.1)
            n_mild = sum(1 for v in smd_vals if 0.1 <= v < 0.25)
            n_substantial = sum(1 for v in smd_vals if v >= 0.25)
            pout(f"    Max SMD: {max_smd:.4f}, Mean SMD: {mean_smd:.4f}")
            pout(f"    Balanced: {n_balanced}, Mild: {n_mild}, Substantial: {n_substantial}")
            balance_summary[comparison_name]["_overall"] = {
                "max_SMD": round(max_smd, 4),
                "mean_SMD": round(mean_smd, 4),
                "n_balanced": n_balanced,
                "n_mild_imbalance": n_mild,
                "n_substantial_imbalance": n_substantial,
            }

    R["balance_summary"] = balance_summary

    # ── Propagation outcome comparison ──
    pout("\n" + "=" * 70)
    pout("PROPAGATION OUTCOME BY CONSTRAINT CLASS")
    pout("=" * 70)

    outcome_results = {}
    for cls in ["permissive", "pinned", "restrictive"]:
        cls_data = pypi_health[pypi_health["constraint_class"] == cls]
        n_total = len(cls_data)
        if n_total == 0:
            continue
        n_auto = (cls_data["propagation_type"] == "auto").sum()
        n_manual = (cls_data["propagation_type"] == "manual").sum()
        n_censored = (cls_data["propagation_type"] == "censored").sum()
        auto_rate = n_auto / n_total
        manual_of_nonaut = n_manual / (n_manual + n_censored) if (n_manual + n_censored) > 0 else 0

        outcome_results[cls] = {
            "n_total": int(n_total),
            "n_auto": int(n_auto),
            "n_manual": int(n_manual),
            "n_censored": int(n_censored),
            "auto_rate": round(auto_rate, 4),
            "manual_of_nonauto": round(manual_of_nonaut, 4),
        }
        pout(f"  {cls:15s}: n={n_total:>5}, auto={n_auto:>4} ({auto_rate:.1%}), "
             f"manual={n_manual:>4}, censored={n_censored:>4}")

    R["propagation_outcome_by_class"] = outcome_results

    # Compare propagation_days between permissive and pinned (manual+censored only)
    pout("\n  --- Propagation speed comparison (non-auto events) ---")
    perm_surv = permissive[
        permissive["propagation_type"].isin(["manual", "censored"]) &
        permissive["propagation_days"].notna() &
        (permissive["propagation_days"] > 0)
    ]
    pin_surv = pinned[
        pinned["propagation_type"].isin(["manual", "censored"]) &
        pinned["propagation_days"].notna() &
        (pinned["propagation_days"] > 0)
    ]

    comparisons_speed = compare_groups(
        perm_surv, pin_surv, "propagation_days", "propagation_days"
    )
    R["propagation_speed_comparison"] = comparisons_speed

    # Save
    out = DATA_DIR / "sensitivity_pypi_selection.json"
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    pout(f"\nResults saved: {out}")
    pout("DONE")

    # Also print to stdout
    print(json.dumps(R, indent=2, default=str))


if __name__ == "__main__":
    main()
