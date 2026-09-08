#!/usr/bin/env python3
"""Build a cross-ecosystem base dataset from GHSA + NVD + OSV + cve_patch_commits.

This script creates the unified advisory-level dataset that ALL research
questions derive from.  One row per (GHSA advisory × affected package) for
npm, PyPI, and Maven ecosystems.

Output
------
  data/cross_ecosystem_base.csv
      ghsa_id, cve_id, ecosystem, package_name, severity_label, cvss_score,
      cwe_primary, published_at, first_patched_version, github_repo,
      commit_sha, has_patch, has_repo

  data/cross_ecosystem_repos.csv   (unique repos for health collection)
      repo, cve_count, ecosystem_list
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

GHSA_JSON = DATA_DIR / "ghsa_advisories.json"
NVD_JSON = DATA_DIR / "nvd_cves.json"
OSV_CSV = DATA_DIR / "osv_advisories.csv"
COMMITS_CSV = DATA_DIR / "cve_patch_commits_all.csv"
SEV_CSV = DATA_DIR / "osv_severity_index.csv"

TARGET_ECOSYSTEMS = {"npm", "PyPI", "Maven"}


# ── GHSA parsing ──────────────────────────────────────────────────────

def parse_ghsa() -> pd.DataFrame:
    """Parse GHSA advisories into (advisory × package) rows."""
    with open(GHSA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    advisories = data["advisories"]
    rows = []

    for adv in advisories:
        ghsa_id = adv.get("ghsa_id", "")
        cve_id = adv.get("cve_id", "")
        severity = (adv.get("severity") or "").upper()
        cvss_score = adv.get("cvss_score")
        cvss_vector = adv.get("cvss_vector", "")
        published_at = adv.get("published_at", "")
        cwes = adv.get("cwes", [])
        cwe_primary = cwes[0] if cwes else ""

        # Parse CVSS score from vector if missing
        if cvss_score is None and cvss_vector:
            # Try to extract base score from vector string
            # CVSS:3.x vectors don't embed the score, but we can map severity
            pass

        for pkg in adv.get("affected", []):
            eco = pkg.get("ecosystem", "")
            if eco not in TARGET_ECOSYSTEMS:
                continue

            package_name = pkg.get("name", "")
            patched = pkg.get("first_patched_version", "")
            vuln_range = pkg.get("vulnerable_version_range", "")

            rows.append({
                "ghsa_id": ghsa_id,
                "cve_id": cve_id,
                "ecosystem": eco,
                "package_name": package_name,
                "severity_label": severity,
                "cvss_score": cvss_score,
                "cvss_vector": cvss_vector,
                "cwe_primary": cwe_primary,
                "published_at": published_at,
                "first_patched_version": patched,
                "vulnerable_range": vuln_range,
                "has_patch": bool(patched),
            })

    df = pd.DataFrame(rows)
    print(f"GHSA: {len(df)} rows across {df['ecosystem'].nunique()} ecosystems")
    print(f"  By ecosystem: {dict(df['ecosystem'].value_counts())}")
    print(f"  With patch: {df['has_patch'].sum()}")
    print(f"  Unique CVEs: {df['cve_id'].nunique()}")
    print(f"  Unique GHSA IDs: {df['ghsa_id'].nunique()}")
    return df


# ── NVD enrichment ────────────────────────────────────────────────────

def load_nvd_scores() -> dict[str, dict]:
    """Load NVD CVSS scores and CWE, keyed by CVE ID."""
    if not NVD_JSON.exists():
        print("WARNING: NVD JSON not found, skipping NVD enrichment")
        return {}

    print("Loading NVD data (this may take a moment)...")
    with open(NVD_JSON, encoding="utf-8") as f:
        nvd_data = json.load(f)

    cve_map = {}

    # Handle different NVD JSON formats
    items = []
    if isinstance(nvd_data, list):
        items = nvd_data
    elif isinstance(nvd_data, dict):
        items = nvd_data.get("vulnerabilities", nvd_data.get("CVE_Items", []))

    for item in items:
        # Our NVD JSON is a flat list with keys: cve_id, published, cvss3_score, cvss3_severity, cwe_ids
        cve_id = item.get("cve_id", item.get("id", ""))
        if not cve_id:
            # Try nested format
            cve_item = item.get("cve", {})
            cve_id = cve_item.get("id", cve_item.get("CVE_data_meta", {}).get("ID", ""))
        if not cve_id:
            continue

        # Flat format (our data)
        cvss3 = item.get("cvss3_score")

        # Nested format fallback
        if cvss3 is None:
            metrics = item.get("cve", item).get("metrics", {})
            for key in ["cvssMetricV31", "cvssMetricV30"]:
                metric_list = metrics.get(key, [])
                if metric_list:
                    cvss3 = metric_list[0].get("cvssData", {}).get("baseScore")
                    break

        # Extract CWE
        cwe = ""
        cwe_ids = item.get("cwe_ids", [])
        if cwe_ids:
            for c in cwe_ids:
                if c.startswith("CWE-") and c != "CWE-noinfo":
                    cwe = c
                    break
        else:
            weaknesses = item.get("cve", item).get("weaknesses", [])
            for w in weaknesses:
                for desc in w.get("description", []):
                    val = desc.get("value", "")
                    if val.startswith("CWE-") and val != "CWE-noinfo":
                        cwe = val
                        break
                if cwe:
                    break

        # Published date
        published = item.get("published", "")

        cve_map[cve_id] = {
            "nvd_cvss": cvss3,
            "nvd_cwe": cwe,
            "nvd_published": published,
        }

    print(f"NVD: loaded {len(cve_map)} CVEs")
    return cve_map


# ── OSV severity enrichment ──────────────────────────────────────────

def load_osv_severity() -> dict[str, dict]:
    """Load OSV severity index keyed by (osv_id or cve_id)."""
    if not SEV_CSV.exists():
        return {}

    df = pd.read_csv(SEV_CSV)
    df["cvss_score"] = pd.to_numeric(df["cvss_score"], errors="coerce")

    result = {}
    for _, row in df.iterrows():
        osv_id = str(row.get("osv_id", ""))
        cve_id = str(row.get("cve_id", ""))
        entry = {
            "osv_cvss": row.get("cvss_score"),
            "osv_severity": str(row.get("severity_label", "")).replace("MODERATE", "MEDIUM"),
            "osv_cwe": str(row.get("cwe_id", "")),
        }
        if osv_id:
            result[osv_id] = entry
        if cve_id:
            result[cve_id] = entry

    print(f"OSV severity: loaded {len(result)} entries")
    return result


# ── Commit data ───────────────────────────────────────────────────────

def load_commits() -> pd.DataFrame:
    """Load cve_patch_commits, deduplicated to one repo per CVE."""
    if not COMMITS_CSV.exists():
        print("WARNING: cve_patch_commits_all.csv not found")
        return pd.DataFrame(columns=["cve_id", "github_repo", "commit_sha"])

    df = pd.read_csv(COMMITS_CSV)
    # Extract owner/repo from commit URL or repo column
    if "repo" in df.columns:
        df["github_repo"] = df["repo"]
    elif "commit_url" in df.columns:
        df["github_repo"] = df["commit_url"].str.extract(
            r"github\.com/([^/]+/[^/]+)/commit"
        )[0]

    # Keep first commit per CVE (usually the primary fix)
    df = df.sort_values("commit_sha").drop_duplicates(subset=["cve_id"], keep="first")
    df = df[["cve_id", "github_repo", "commit_sha"]].copy()
    df = df.dropna(subset=["cve_id"])

    print(f"Commits: {len(df)} CVEs with GitHub repos ({df['github_repo'].nunique()} unique repos)")
    return df


# ── Build unified dataset ─────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """Build the cross-ecosystem base dataset."""

    # 1. Parse GHSA
    base = parse_ghsa()

    # 2. Enrich with NVD
    nvd = load_nvd_scores()
    if nvd:
        nvd_cvss = []
        nvd_cwe = []
        nvd_published = []
        for cve_id in base["cve_id"]:
            entry = nvd.get(cve_id, {})
            nvd_cvss.append(entry.get("nvd_cvss"))
            nvd_cwe.append(entry.get("nvd_cwe", ""))
            nvd_published.append(entry.get("nvd_published", ""))
        base["nvd_cvss"] = nvd_cvss
        base["nvd_cwe"] = nvd_cwe
        base["nvd_published"] = nvd_published

        # Fill in missing CVSS from NVD
        base["cvss_score"] = base["cvss_score"].fillna(base["nvd_cvss"])
        # Fill in missing CWE from NVD
        mask = base["cwe_primary"].isin(["", None]) | base["cwe_primary"].isna()
        base.loc[mask, "cwe_primary"] = base.loc[mask, "nvd_cwe"]

        matched = base["nvd_cvss"].notna().sum()
        print(f"NVD enrichment: {matched}/{len(base)} rows matched")

    # 3. Enrich with OSV severity
    osv_sev = load_osv_severity()
    if osv_sev:
        osv_cvss_vals = []
        osv_cwe_vals = []
        for _, row in base.iterrows():
            entry = osv_sev.get(row["ghsa_id"], osv_sev.get(row["cve_id"], {}))
            osv_cvss_vals.append(entry.get("osv_cvss"))
            osv_cwe_vals.append(entry.get("osv_cwe", ""))
        base["osv_cvss"] = osv_cvss_vals
        base["osv_cwe"] = osv_cwe_vals

        # Fill remaining CVSS gaps
        still_missing = base["cvss_score"].isna()
        base.loc[still_missing, "cvss_score"] = base.loc[still_missing, "osv_cvss"]
        mask = base["cwe_primary"].isin(["", None]) | base["cwe_primary"].isna()
        base.loc[mask, "cwe_primary"] = base.loc[mask, "osv_cwe"]

    # 4. Standardize severity labels from CVSS score
    def cvss_to_severity(score):
        if pd.isna(score):
            return ""
        score = float(score)
        if score >= 9.0:
            return "CRITICAL"
        elif score >= 7.0:
            return "HIGH"
        elif score >= 4.0:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        return ""

    base["cvss_score"] = pd.to_numeric(base["cvss_score"], errors="coerce")
    # Normalize MODERATE → MEDIUM
    base["severity_label"] = base["severity_label"].replace({"MODERATE": "MEDIUM"})
    # Use CVSS-derived severity where GHSA label is missing
    missing_sev = base["severity_label"].isin(["", "NONE"]) | base["severity_label"].isna()
    base.loc[missing_sev, "severity_label"] = base.loc[missing_sev, "cvss_score"].apply(cvss_to_severity)

    # 5. Join with commit data
    commits = load_commits()
    if not commits.empty:
        base = base.merge(commits, on="cve_id", how="left")
        base["has_repo"] = base["github_repo"].notna()
        print(f"After commit join: {base['has_repo'].sum()}/{len(base)} rows with GitHub repo")
    else:
        base["github_repo"] = None
        base["commit_sha"] = None
        base["has_repo"] = False

    # 6. Parse dates
    base["published_at"] = pd.to_datetime(base["published_at"], errors="coerce", utc=True)
    if "nvd_published" in base.columns:
        base["nvd_published"] = pd.to_datetime(base["nvd_published"], errors="coerce", utc=True)

    # 7. Clean up
    drop_cols = [c for c in ["nvd_cvss", "nvd_cwe", "osv_cvss", "osv_cwe"] if c in base.columns]
    base = base.drop(columns=drop_cols, errors="ignore")

    return base


def build_repo_list(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique repos for health metric collection."""
    repo_df = df[df["has_repo"]].copy()
    repos = repo_df.groupby("github_repo").agg(
        cve_count=("cve_id", "nunique"),
        ecosystem_list=("ecosystem", lambda x: ",".join(sorted(set(x)))),
    ).reset_index().rename(columns={"github_repo": "repo"})
    return repos


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Building cross-ecosystem base dataset")
    print("=" * 60)

    df = build_dataset()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Unique advisories (GHSA): {df['ghsa_id'].nunique()}")
    print(f"Unique CVEs: {df['cve_id'].nunique()}")
    print(f"\nBy ecosystem:")
    for eco in ["npm", "PyPI", "Maven"]:
        sub = df[df["ecosystem"] == eco]
        print(f"  {eco}: {len(sub)} rows, {sub['cve_id'].nunique()} CVEs, "
              f"{sub['has_patch'].sum()} with patch, {sub['has_repo'].sum()} with repo")
    print(f"\nCVSS coverage: {df['cvss_score'].notna().sum()}/{len(df)} ({df['cvss_score'].notna().mean():.1%})")
    print(f"CWE coverage: {(df['cwe_primary'].notna() & (df['cwe_primary'] != '')).sum()}/{len(df)}")
    print(f"Severity distribution:")
    print(df["severity_label"].value_counts().to_string())
    print(f"\nHas patch version: {df['has_patch'].sum()}")
    print(f"Has GitHub repo: {df['has_repo'].sum()}")

    # Save
    out_path = DATA_DIR / "cross_ecosystem_base.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")

    # Save repo list for health collection
    repos = build_repo_list(df)
    repo_path = DATA_DIR / "cross_ecosystem_repos.csv"
    repos.to_csv(repo_path, index=False)
    print(f"Saved: {repo_path} ({len(repos)} repos)")


if __name__ == "__main__":
    main()
