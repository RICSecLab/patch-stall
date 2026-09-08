#!/usr/bin/env python3
"""
Validate registry API dates against git tag dates for patch releases.

Joins patch_release_dates.csv (registry API) and git_tag_dates.csv (git tags)
on (ghsa_id, ecosystem, package_name, first_patched_version), computes delta,
reports per-ecosystem statistics, and produces a merged dataset.

Decision logic:
  - median |delta| < 1 day  → tag dates are valid substitute → unified metric
  - median |delta| 1-7 days → acceptable with caveat
  - median |delta| > 7 days → tag dates measure something different → source covariate

Outputs:
  - Console: per-ecosystem validation statistics
  - paper/fig/dual_source_validation.pdf: histogram of deltas
  - data/patch_dates_merged.csv: unified date source
"""

import os
import sys

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_CSV = os.path.join(BASE, "data", "patch_release_dates.csv")
GIT_TAG_CSV = os.path.join(BASE, "data", "git_tag_dates.csv")
FIG_DIR = os.path.join(BASE, "paper", "fig")
OUT_FIG = os.path.join(FIG_DIR, "dual_source_validation.pdf")
OUT_MERGED = os.path.join(BASE, "data", "patch_dates_merged.csv")

JOIN_KEYS = ["ghsa_id", "ecosystem", "package_name", "first_patched_version"]


def load_data():
    """Load and parse both CSV files."""
    reg = pd.read_csv(REGISTRY_CSV)
    git = pd.read_csv(GIT_TAG_CSV)

    # Parse dates
    reg["patch_release_date"] = pd.to_datetime(
        reg["patch_release_date"], errors="coerce", utc=True
    )
    git["tag_date"] = pd.to_datetime(
        git["tag_date"], errors="coerce", utc=True
    )

    return reg, git


def compute_overlap(reg, git):
    """Inner join on keys where both sources have valid dates."""
    # Filter to rows with valid dates
    reg_valid = reg.dropna(subset=["patch_release_date"]).copy()
    git_valid = git.dropna(subset=["tag_date"]).copy()

    # Also filter out git rows with no_match / no_tags (no actual tag found)
    git_valid = git_valid[~git_valid["match_type"].isin(["no_match", "no_tags"])]

    merged = reg_valid.merge(git_valid, on=JOIN_KEYS, how="inner")

    # Compute delta in days (registry - tag)
    merged["delta_days"] = (
        merged["patch_release_date"] - merged["tag_date"]
    ).dt.total_seconds() / 86400.0

    return merged


def report_per_ecosystem(merged):
    """Report per-ecosystem validation statistics."""
    print("=" * 78)
    print("DUAL-SOURCE VALIDATION: Registry API dates vs Git Tag dates")
    print("=" * 78)
    print()

    ecosystems = sorted(merged["ecosystem"].unique())
    results = {}

    for eco in ecosystems:
        sub = merged[merged["ecosystem"] == eco]
        n = len(sub)
        delta = sub["delta_days"]
        abs_delta = delta.abs()

        med = delta.median()
        mean = delta.mean()
        sd = delta.std()
        med_abs = abs_delta.median()
        mean_abs = abs_delta.mean()

        within_1d = (abs_delta <= 1).mean() * 100
        within_7d = (abs_delta <= 7).mean() * 100
        within_30d = (abs_delta <= 30).mean() * 100

        # Pearson correlation on raw timestamps
        reg_ts = sub["patch_release_date"].astype(np.int64) / 1e9
        tag_ts = sub["tag_date"].astype(np.int64) / 1e9
        r, p = stats.pearsonr(reg_ts, tag_ts)

        results[eco] = {
            "n": n,
            "median": med,
            "mean": mean,
            "std": sd,
            "median_abs": med_abs,
            "mean_abs": mean_abs,
            "within_1d": within_1d,
            "within_7d": within_7d,
            "within_30d": within_30d,
            "pearson_r": r,
            "pearson_p": p,
        }

        print(f"─── {eco} (N={n}) ───")
        print(f"  Delta (registry − tag), days:")
        print(f"    Median:  {med:+.2f}    Mean: {mean:+.2f}    Std: {sd:.2f}")
        print(f"    Median |Δ|: {med_abs:.2f}    Mean |Δ|: {mean_abs:.2f}")
        print(f"  Agreement:")
        print(f"    Within  1 day:  {within_1d:5.1f}%")
        print(f"    Within  7 days: {within_7d:5.1f}%")
        print(f"    Within 30 days: {within_30d:5.1f}%")
        print(f"  Pearson r = {r:.6f}  (p = {p:.2e})")

        # Decision
        if med_abs < 1:
            decision = "VALID SUBSTITUTE (median |Δ| < 1 day)"
        elif med_abs <= 7:
            decision = "ACCEPTABLE WITH CAVEAT (median |Δ| 1-7 days)"
        else:
            decision = "DIFFERENT MEASURE → source covariate needed (median |Δ| > 7 days)"
        print(f"  Decision: {decision}")
        print()

    # Overall
    n_all = len(merged)
    delta_all = merged["delta_days"]
    abs_all = delta_all.abs()
    print(f"─── OVERALL (N={n_all}) ───")
    print(f"  Median Δ: {delta_all.median():+.2f}  Mean Δ: {delta_all.mean():+.2f}  Std: {delta_all.std():.2f}")
    print(f"  Median |Δ|: {abs_all.median():.2f}")
    print(f"  Within 1d: {(abs_all <= 1).mean()*100:.1f}%  "
          f"7d: {(abs_all <= 7).mean()*100:.1f}%  "
          f"30d: {(abs_all <= 30).mean()*100:.1f}%")
    reg_ts = merged["patch_release_date"].astype(np.int64) / 1e9
    tag_ts = merged["tag_date"].astype(np.int64) / 1e9
    r, p = stats.pearsonr(reg_ts, tag_ts)
    print(f"  Pearson r = {r:.6f}  (p = {p:.2e})")
    overall_med_abs = abs_all.median()
    if overall_med_abs < 1:
        print(f"  OVERALL DECISION: UNIFIED METRIC (median |Δ| = {overall_med_abs:.2f} < 1 day)")
    elif overall_med_abs <= 7:
        print(f"  OVERALL DECISION: ACCEPTABLE WITH CAVEAT (median |Δ| = {overall_med_abs:.2f} days)")
    else:
        print(f"  OVERALL DECISION: SOURCE COVARIATE NEEDED (median |Δ| = {overall_med_abs:.2f} > 7 days)")
    print()

    return results


def report_maven_new_events(reg, git):
    """Report how many Maven events git tags provide that registry does not."""
    git_maven = git[
        (git["ecosystem"] == "Maven")
        & git["tag_date"].notna()
        & ~git["match_type"].isin(["no_match", "no_tags"])
    ].copy()

    reg_maven = reg[
        (reg["ecosystem"] == "Maven")
        & reg["patch_release_date"].notna()
    ].copy()

    # Find git tag rows with no matching registry date
    git_keys = set(
        git_maven[JOIN_KEYS].apply(tuple, axis=1)
    )
    reg_keys = set(
        reg_maven[JOIN_KEYS].apply(tuple, axis=1)
    )

    # Also count registry rows with no date (blank) that git fills in
    reg_maven_nodate = reg[
        (reg["ecosystem"] == "Maven")
        & (reg["patch_release_date"].isna())
    ]
    reg_nodate_keys = set(
        reg_maven_nodate[JOIN_KEYS].apply(tuple, axis=1)
    )

    new_events = git_keys - reg_keys
    rescued_from_blank = git_keys & reg_nodate_keys

    print("=" * 78)
    print("MAVEN: New events recovered from git tags")
    print("=" * 78)
    print(f"  Git tag rows (Maven, valid date, matched tag): {len(git_maven)}")
    print(f"  Registry rows (Maven, valid date):             {len(reg_maven)}")
    print(f"  Registry rows (Maven, blank date):             {len(reg_maven_nodate)}")
    print(f"  New events (git-only, no registry match):      {len(new_events)}")
    print(f"  Rescued (blank in registry, filled by git):    {len(rescued_from_blank)}")
    print(f"  Total new dates git provides for Maven:        {len(new_events | rescued_from_blank)}")
    print()

    return new_events, rescued_from_blank


def generate_figure(merged):
    """Generate histogram of delta (registry - tag) per ecosystem."""
    ecosystems = sorted(merged["ecosystem"].unique())
    n_eco = len(ecosystems)

    fig, axes = plt.subplots(1, n_eco + 1, figsize=(4 * (n_eco + 1), 3.5),
                              sharey=False)

    colors = {"Maven": "#2196F3", "npm": "#4CAF50", "PyPI": "#FF9800"}

    for i, eco in enumerate(ecosystems):
        ax = axes[i]
        delta = merged[merged["ecosystem"] == eco]["delta_days"]
        # Clip extreme outliers for visualization
        clip_val = 30
        clipped = delta.clip(-clip_val, clip_val)
        ax.hist(clipped, bins=61, color=colors.get(eco, "#888"),
                alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.axvline(delta.median(), color="red", linewidth=1.5,
                   linestyle="-", label=f"Median={delta.median():+.1f}d")
        ax.set_xlabel("Delta (registry − tag), days")
        ax.set_title(f"{eco} (N={len(delta)})")
        ax.set_xlim(-clip_val, clip_val)
        ax.legend(fontsize=7, loc="upper right")
        if i == 0:
            ax.set_ylabel("Count")

    # Overall panel
    ax = axes[-1]
    delta = merged["delta_days"]
    clip_val = 30
    clipped = delta.clip(-clip_val, clip_val)
    ax.hist(clipped, bins=61, color="#666", alpha=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.axvline(delta.median(), color="red", linewidth=1.5,
               linestyle="-", label=f"Median={delta.median():+.1f}d")
    ax.set_xlabel("Delta (registry − tag), days")
    ax.set_title(f"Overall (N={len(delta)})")
    ax.set_xlim(-clip_val, clip_val)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Dual-Source Validation: Registry API vs Git Tag Dates",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_FIG.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Figure saved: {OUT_FIG}")
    plt.close(fig)


def build_merged_dataset(reg, git):
    """Build unified patch_dates_merged.csv.

    Priority:
      1. Registry date (more precise, sub-second timestamps)
      2. Git tag date (if registry date is missing)

    Output columns:
      ghsa_id, ecosystem, package_name, first_patched_version,
      github_repo, patch_date, date_source, tag_match_type
    """
    # Prepare registry side
    reg_out = reg.copy()
    reg_out = reg_out.rename(columns={"patch_release_date": "patch_date"})
    reg_out["date_source"] = "registry"
    reg_out["github_repo"] = np.nan
    reg_out["tag_match_type"] = np.nan

    # Prepare git side
    git_valid = git[
        git["tag_date"].notna()
        & ~git["match_type"].isin(["no_match", "no_tags"])
    ].copy()
    git_out = git_valid.rename(columns={"tag_date": "patch_date",
                                         "match_type": "tag_match_type"})
    git_out["date_source"] = "git_tag"
    git_out = git_out[JOIN_KEYS + ["github_repo", "patch_date",
                                    "date_source", "tag_match_type"]]

    # Registry rows with valid dates
    reg_has_date = reg_out[reg_out["patch_date"].notna()].copy()
    reg_has_date = reg_has_date[JOIN_KEYS + ["github_repo", "patch_date",
                                              "date_source", "tag_match_type"]]

    # For registry rows with valid dates, also grab github_repo from git data
    if len(reg_has_date) > 0:
        repo_lookup = git[JOIN_KEYS + ["github_repo"]].drop_duplicates(subset=JOIN_KEYS)
        reg_has_date = reg_has_date.drop(columns=["github_repo"])
        reg_has_date = reg_has_date.merge(repo_lookup, on=JOIN_KEYS, how="left")
        # Reorder columns
        reg_has_date = reg_has_date[JOIN_KEYS + ["github_repo", "patch_date",
                                                   "date_source", "tag_match_type"]]

    # Registry keys with valid dates (for exclusion)
    reg_valid_keys = set(
        reg_has_date[JOIN_KEYS].apply(tuple, axis=1)
    )

    # Git-only rows: rows in git that aren't in registry-valid
    git_only = git_out[
        ~git_out[JOIN_KEYS].apply(tuple, axis=1).isin(reg_valid_keys)
    ]

    # Combine
    merged = pd.concat([reg_has_date, git_only], ignore_index=True)

    # Sort
    merged = merged.sort_values(JOIN_KEYS).reset_index(drop=True)

    merged.to_csv(OUT_MERGED, index=False)
    print(f"Merged dataset saved: {OUT_MERGED}")
    print(f"  Total rows: {len(merged)}")
    print()

    # Report by ecosystem and source
    print("─── Merged dataset summary ───")
    for eco in sorted(merged["ecosystem"].unique()):
        sub = merged[merged["ecosystem"] == eco]
        n_reg = (sub["date_source"] == "registry").sum()
        n_git = (sub["date_source"] == "git_tag").sum()
        print(f"  {eco:8s}: {len(sub):6d} total  ({n_reg:5d} registry + {n_git:5d} git_tag)")
    n_reg_all = (merged["date_source"] == "registry").sum()
    n_git_all = (merged["date_source"] == "git_tag").sum()
    print(f"  {'TOTAL':8s}: {len(merged):6d} total  ({n_reg_all:5d} registry + {n_git_all:5d} git_tag)")
    print()

    return merged


def main():
    reg, git = load_data()

    # Phase 1: Overlap validation
    overlap = compute_overlap(reg, git)
    print(f"Overlap (both sources have dates): {len(overlap)} rows\n")
    results = report_per_ecosystem(overlap)

    # Phase 2: Maven new events
    report_maven_new_events(reg, git)

    # Phase 3: Figure
    generate_figure(overlap)
    print()

    # Phase 4: Merged dataset
    merged = build_merged_dataset(reg, git)

    print("=" * 78)
    print("DONE. Validation complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
