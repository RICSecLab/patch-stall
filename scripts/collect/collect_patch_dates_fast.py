#!/usr/bin/env python3
"""Fast parallel patch date collection for npm, PyPI, Maven.

Collection strategy per ecosystem:
  - npm: One API call per PACKAGE (returns all version timestamps)
  - PyPI: Parallel fetching with reduced sleep
  - Maven: Optimized search queries
  - All: Concurrent collection across ecosystems

Usage:
  python scripts/collect_patch_dates_fast.py --ecosystem npm
  python scripts/collect_patch_dates_fast.py --ecosystem pypi
  python scripts/collect_patch_dates_fast.py --ecosystem maven
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
BASE_CSV = DATA_DIR / "cross_ecosystem_base.csv"
OUTPUT = DATA_DIR / "patch_release_dates.csv"

session = requests.Session()
session.headers["User-Agent"] = "patch-stall-replication"


def collect_npm(todo: pd.DataFrame):
    """Collect npm version dates - one call per package."""
    # Group by package
    packages = todo.groupby("package_name")["first_patched_version"].apply(set).to_dict()
    print(f"npm: {len(packages)} unique packages, {len(todo)} version lookups")

    results = []
    for i, (pkg, versions) in enumerate(packages.items()):
        if i > 0 and i % 50 == 0:
            print(f"  {i}/{len(packages)} packages")

        try:
            resp = session.get(f"https://registry.npmjs.org/{pkg}", timeout=15)
            if resp.status_code != 200:
                for ver in versions:
                    results.append(("", pkg, ver, ""))
                continue

            data = resp.json()
            times = data.get("time", {})

            for ver in versions:
                date = times.get(ver, "")
                ghsa = todo[(todo["package_name"] == pkg) &
                           (todo["first_patched_version"] == ver)]["ghsa_id"].iloc[0]
                results.append((ghsa, pkg, ver, date))

        except Exception:
            for ver in versions:
                results.append(("", pkg, ver, ""))

        time.sleep(0.2)

    return results


def collect_pypi(todo: pd.DataFrame):
    """Collect PyPI version dates."""
    results = []
    for i, (_, row) in enumerate(todo.iterrows()):
        if i > 0 and i % 100 == 0:
            print(f"  {i}/{len(todo)}")

        pkg = row["package_name"]
        ver = row["first_patched_version"]

        try:
            resp = session.get(
                f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                urls = data.get("urls", [])
                date = ""
                if urls:
                    date = urls[0].get("upload_time_iso_8601", "") or urls[0].get("upload_time", "")
                results.append((row["ghsa_id"], pkg, ver, date))
            else:
                results.append((row["ghsa_id"], pkg, ver, ""))
        except Exception:
            results.append((row["ghsa_id"], pkg, ver, ""))

        time.sleep(0.15)

    return results


def collect_maven(todo: pd.DataFrame):
    """Collect Maven version dates."""
    results = []
    for i, (_, row) in enumerate(todo.iterrows()):
        if i > 0 and i % 100 == 0:
            print(f"  {i}/{len(todo)}")

        pkg = row["package_name"]
        ver = row["first_patched_version"]

        # Parse group:artifact
        if ":" in pkg:
            group, artifact = pkg.split(":", 1)
        else:
            # Try common Maven naming: groupId.artifactId
            parts = pkg.rsplit(".", 1)
            if len(parts) == 2:
                group, artifact = parts
            else:
                results.append((row["ghsa_id"], pkg, ver, ""))
                continue

        try:
            url = (
                f"https://search.maven.org/solrsearch/select?"
                f"q=g:{group}+AND+a:{artifact}+AND+v:{ver}"
                f"&rows=1&wt=json"
            )
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                if docs:
                    ts = docs[0].get("timestamp")
                    if ts:
                        from datetime import datetime, timezone
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                        results.append((row["ghsa_id"], pkg, ver, dt.isoformat()))
                    else:
                        results.append((row["ghsa_id"], pkg, ver, ""))
                else:
                    results.append((row["ghsa_id"], pkg, ver, ""))
            else:
                results.append((row["ghsa_id"], pkg, ver, ""))
        except Exception:
            results.append((row["ghsa_id"], pkg, ver, ""))

        time.sleep(0.3)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecosystem", required=True, choices=["npm", "pypi", "maven"])
    args = parser.parse_args()

    eco_map = {"npm": "npm", "pypi": "PyPI", "maven": "Maven"}
    eco = eco_map[args.ecosystem]

    base = pd.read_csv(BASE_CSV)
    with_patch = base[base["has_patch"] & base["first_patched_version"].notna()].copy()
    dedup = with_patch[with_patch["ecosystem"] == eco].drop_duplicates(
        subset=["package_name", "first_patched_version"]
    )[["ghsa_id", "ecosystem", "package_name", "first_patched_version"]].copy()

    # Load existing
    existing = set()
    if OUTPUT.exists():
        try:
            done = pd.read_csv(OUTPUT)
            done_eco = done[done["ecosystem"] == eco]
            existing = set(
                zip(done_eco["package_name"], done_eco["first_patched_version"])
            )
        except Exception:
            pass

    todo = dedup[
        ~dedup.apply(
            lambda r: (r["package_name"], r["first_patched_version"]) in existing,
            axis=1
        )
    ]
    print(f"{eco}: {len(todo)} remaining ({len(existing)} already done)")

    if len(todo) == 0:
        print("Nothing to collect")
        return

    if args.ecosystem == "npm":
        results = collect_npm(todo)
    elif args.ecosystem == "pypi":
        results = collect_pypi(todo)
    elif args.ecosystem == "maven":
        results = collect_maven(todo)

    # Append to output
    write_header = not OUTPUT.exists() or OUTPUT.stat().st_size == 0
    with open(OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ghsa_id", "ecosystem", "package_name",
            "first_patched_version", "patch_release_date",
        ])
        if write_header:
            writer.writeheader()

        found = 0
        for ghsa_id, pkg, ver, date in results:
            if date:
                found += 1
            writer.writerow({
                "ghsa_id": ghsa_id,
                "ecosystem": eco,
                "package_name": pkg,
                "first_patched_version": ver,
                "patch_release_date": date,
            })

    print(f"Done. Found: {found}/{len(results)}")


if __name__ == "__main__":
    main()
