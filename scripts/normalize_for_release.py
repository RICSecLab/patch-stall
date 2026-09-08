#!/usr/bin/env python3
"""Normalize collected data into publication-ready artifact tables.

Reads raw CSV files from data/ and writes normalized tables to data/release/.
Idempotent: safe to re-run after additional data collection.

Output tables:
  advisories.csv        - Advisory master (1 row per GHSA-package pair)
  repositories.csv      - Repository health metrics (1 row per repo)
  patch_timing.csv      - Unified patch dates (1 row per advisory with date)
  propagation_events.csv - Unified propagation (npm + PyPI + Maven, ecosystem col)

Design:
  - propagation_events.csv contains only event-level columns + osv_id FK
  - Advisory-level attributes (severity, CWE, patch version) are in advisories.csv
  - Join on osv_id (= GHSA ID) to reconstruct denormalized view
  - All dates in ISO 8601 UTC
  - Missing values as empty string (CSV standard)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RELEASE_DIR = DATA_DIR / "release"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_advisories() -> int:
    """Normalize cross_ecosystem_base.csv → advisories.csv."""
    src = DATA_DIR / "cross_ecosystem_base.csv"
    dst = RELEASE_DIR / "advisories.csv"

    out_cols = [
        "osv_id",       # renamed from ghsa_id
        "cve_id",
        "ecosystem",
        "package_name",
        "severity_label",
        "cvss_score",
        "cwe_primary",
        "published_at",
        "first_patched_version",
        "has_patch",
        "github_repo",
        "has_repo",
    ]

    rows = 0
    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()

        for row in reader:
            out = {
                "osv_id": row["ghsa_id"],
                "cve_id": row.get("cve_id", ""),
                "ecosystem": row.get("ecosystem", ""),
                "package_name": row.get("package_name", ""),
                "severity_label": row.get("severity_label", ""),
                "cvss_score": row.get("cvss_score", ""),
                "cwe_primary": row.get("cwe_primary", ""),
                "published_at": row.get("published_at", ""),
                "first_patched_version": row.get("first_patched_version", ""),
                "has_patch": row.get("has_patch", ""),
                "github_repo": row.get("github_repo", ""),
                "has_repo": row.get("has_repo", ""),
            }
            writer.writerow(out)
            rows += 1

    return rows


def normalize_repositories() -> int:
    """Join cross_ecosystem_repos.csv + github_repo_health.csv → repositories.csv."""
    repos_file = DATA_DIR / "cross_ecosystem_repos.csv"
    health_file = DATA_DIR / "github_repo_health.csv"
    dst = RELEASE_DIR / "repositories.csv"

    # Load health data keyed by repo
    health = {}
    with open(health_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            health[row["repo"]] = row

    out_cols = [
        "repo",
        "ecosystem_list",
        "cve_count",
        "stars",
        "forks",
        "open_issues",
        "contributors_count",
        "language",
        "created_at",
        "pushed_at",
        "archived",
    ]

    rows = 0
    with open(repos_file, "r", encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()

        for row in reader:
            repo = row["repo"]
            h = health.get(repo, {})
            out = {
                "repo": repo,
                "ecosystem_list": row.get("ecosystem_list", ""),
                "cve_count": row.get("cve_count", ""),
                "stars": h.get("stars", ""),
                "forks": h.get("forks", ""),
                "open_issues": h.get("open_issues", ""),
                "contributors_count": h.get("contributors_count", ""),
                "language": h.get("language", ""),
                "created_at": h.get("created_at", ""),
                "pushed_at": h.get("pushed_at", ""),
                "archived": h.get("archived", ""),
            }
            writer.writerow(out)
            rows += 1

    return rows


def normalize_patch_timing() -> int:
    """Normalize patch_dates_merged.csv → patch_timing.csv."""
    src = DATA_DIR / "patch_dates_merged.csv"
    dst = RELEASE_DIR / "patch_timing.csv"

    out_cols = [
        "osv_id",       # renamed from ghsa_id
        "ecosystem",
        "package_name",
        "first_patched_version",
        "github_repo",
        "patch_date",
        "date_source",
        "tag_match_type",
    ]

    rows = 0
    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()

        for row in reader:
            out = {
                "osv_id": row["ghsa_id"],
                "ecosystem": row.get("ecosystem", ""),
                "package_name": row.get("package_name", ""),
                "first_patched_version": row.get("first_patched_version", ""),
                "github_repo": row.get("github_repo", ""),
                "patch_date": row.get("patch_date", ""),
                "date_source": row.get("date_source", ""),
                "tag_match_type": row.get("tag_match_type", ""),
            }
            writer.writerow(out)
            rows += 1

    return rows


def normalize_propagation() -> int:
    """Merge 3 ecosystem propagation CSVs → propagation_events.csv.

    Adds 'ecosystem' column, removes advisory-level attributes
    (severity, CWE, patched_version, patch_release_date) that
    belong in the advisories table. Retains osv_id and cve_id
    as foreign keys, and upstream_package for convenience.
    """
    dst = RELEASE_DIR / "propagation_events.csv"

    out_cols = [
        "ecosystem",
        "osv_id",
        "cve_id",
        "upstream_package",
        "downstream_package",
        "downstream_version",
        "downstream_release_date",
        "first_safe_release_date",
        "propagation_days",
        "propagation_type",
        "dependency_type",
        "version_constraint_at_t0",
        "constraint_type",
        "requires_major_upgrade",
    ]

    # Ecosystem labels must match advisories.csv and patch_timing.csv so the
    # documented joins work without case normalization.
    ecosystem_files = [
        ("npm", DATA_DIR / "npm_propagation.csv"),
        ("PyPI", DATA_DIR / "pypi_propagation.csv"),
        ("Maven", DATA_DIR / "maven_propagation.csv"),
    ]

    rows = 0
    with open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_cols, lineterminator="\n")
        writer.writeheader()

        for eco, src in ecosystem_files:
            if not src.exists():
                print(f"  WARN: {src.name} not found, skipping")
                continue

            with open(src, "r", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    out = {
                        "ecosystem": eco,
                        "osv_id": row.get("osv_id", ""),
                        "cve_id": row.get("cve_id", ""),
                        "upstream_package": row.get("upstream_package", ""),
                        "downstream_package": row.get("downstream_package", ""),
                        "downstream_version": row.get("downstream_version", ""),
                        "downstream_release_date": row.get("downstream_release_date", ""),
                        "first_safe_release_date": row.get("first_safe_release_date", ""),
                        "propagation_days": row.get("propagation_days", ""),
                        "propagation_type": row.get("propagation_type", ""),
                        "dependency_type": row.get("dependency_type", ""),
                        "version_constraint_at_t0": row.get("version_constraint_at_t0", ""),
                        "constraint_type": row.get("constraint_type", ""),
                        "requires_major_upgrade": row.get("requires_major_upgrade", ""),
                    }
                    writer.writerow(out)
                    rows += 1

    return rows


def generate_manifest(table_counts: dict[str, int]):
    """Generate manifest.json with file hashes, row counts, and metadata."""
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/normalize_for_release.py",
        "observation_cutoff": "2026-03-01",
        "data_freeze": "2026-03-07",
        "tables": {},
    }

    for name, count in table_counts.items():
        path = RELEASE_DIR / name
        manifest["tables"][name] = {
            "rows": count,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    # Add source file hashes for provenance
    manifest["source_hashes"] = {}
    for src in sorted(DATA_DIR.glob("*.csv")):
        manifest["source_hashes"][src.name] = sha256_file(src)

    dst = RELEASE_DIR / "manifest.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Normalizing data for release...")
    print(f"  Source: {DATA_DIR}")
    print(f"  Output: {RELEASE_DIR}")
    print()

    counts = {}

    print("  [1/4] advisories.csv")
    n = normalize_advisories()
    counts["advisories.csv"] = n
    print(f"         {n:,} rows")

    print("  [2/4] repositories.csv")
    n = normalize_repositories()
    counts["repositories.csv"] = n
    print(f"         {n:,} rows")

    print("  [3/4] patch_timing.csv")
    n = normalize_patch_timing()
    counts["patch_timing.csv"] = n
    print(f"         {n:,} rows")

    print("  [4/4] propagation_events.csv")
    n = normalize_propagation()
    counts["propagation_events.csv"] = n
    print(f"         {n:,} rows")

    print()
    print("  Generating manifest.json...")
    generate_manifest(counts)

    total = sum(counts.values())
    print()
    print(f"  Done. {len(counts)} tables, {total:,} total rows.")
    print(f"  Output: {RELEASE_DIR}/")


if __name__ == "__main__":
    main()
