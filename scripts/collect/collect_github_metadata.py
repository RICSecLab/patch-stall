#!/usr/bin/env python3
"""Collect GitHub metadata for the cross-ecosystem dataset.

Two collection passes:
  1. Repo health metrics (stars, forks, contributors, open issues, language, age)
  2. Commit dates for patch commits (to compute patch latency)

Uses authenticated GitHub REST API (5,000 requests/hour).
Saves incrementally so the script can be resumed.

Output
------
  data/github_repo_health.csv    -- one row per repo
  data/github_commit_dates.csv   -- one row per (cve_id, commit_sha)

Usage
-----
  python scripts/collect_github_metadata.py --token ghp_xxx
  python scripts/collect_github_metadata.py --token ghp_xxx --repos-only
  python scripts/collect_github_metadata.py --token ghp_xxx --commits-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

BASE_CSV = DATA_DIR / "cross_ecosystem_base.csv"
REPOS_CSV = DATA_DIR / "cross_ecosystem_repos.csv"

HEALTH_OUT = DATA_DIR / "github_repo_health.csv"
COMMITS_OUT = DATA_DIR / "github_commit_dates.csv"

API_BASE = "https://api.github.com"
RATE_PAUSE = 2.0  # seconds between requests (conservative)


class GitHubCollector:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "patch-stall-replication",
        })
        self.calls = 0

    def _get(self, url: str) -> dict | list | None:
        """Make a GET request with rate limiting."""
        self.calls += 1
        try:
            resp = self.session.get(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"    Request error: {e}")
            return None

        if resp.status_code == 403:
            # Rate limited
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 60)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait + 1)
            return self._get(url)

        if resp.status_code == 404:
            return None

        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {url}")
            return None

        remaining = int(resp.headers.get("X-RateLimit-Remaining", 5000))
        if remaining < 100:
            print(f"  Rate limit low: {remaining} remaining")

        return resp.json()

    def check_rate(self):
        """Check current rate limit status."""
        data = self._get(f"{API_BASE}/rate_limit")
        if data:
            core = data.get("resources", {}).get("core", {})
            print(f"Rate limit: {core.get('remaining')}/{core.get('limit')} "
                  f"(resets at {core.get('reset')})")

    def _get_contributor_count(self, repo: str) -> int:
        """Get contributor count using Link header pagination trick."""
        self.calls += 1
        url = f"{API_BASE}/repos/{repo}/contributors?per_page=1&anon=true"
        try:
            resp = self.session.get(url, timeout=30)
        except requests.exceptions.RequestException:
            return 0

        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 60)
            time.sleep(wait + 1)
            return self._get_contributor_count(repo)

        if resp.status_code != 200:
            return 0

        # Parse Link header: <url>; rel="last" contains page count
        link = resp.headers.get("Link", "")
        if 'rel="last"' in link:
            import re
            match = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link)
            if match:
                return int(match.group(1))

        # No pagination = all fit on one page
        data = resp.json()
        return len(data) if isinstance(data, list) else 0

    # ── Repo health collection ────────────────────────────────────────

    def collect_repo_health(self, repos: list[str], existing: set[str] | None = None):
        """Collect health metrics for repos."""
        if existing is None:
            existing = set()

        todo = [r for r in repos if r not in existing]
        print(f"Repo health: {len(todo)} to collect ({len(existing)} already done)")

        # Open file for appending
        write_header = not HEALTH_OUT.exists() or HEALTH_OUT.stat().st_size == 0
        with open(HEALTH_OUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "repo", "stars", "forks", "open_issues", "size_kb",
                "language", "created_at", "pushed_at", "archived",
                "contributors_count", "default_branch",
            ])
            if write_header:
                writer.writeheader()

            for i, repo in enumerate(todo):
                if i > 0 and i % 100 == 0:
                    print(f"  {i}/{len(todo)} repos collected ({self.calls} API calls)")

                data = self._get(f"{API_BASE}/repos/{repo}")
                if data is None:
                    writer.writerow({"repo": repo})
                    continue

                # Get contributor count via Link header pagination trick
                contrib_count = self._get_contributor_count(repo)

                row = {
                    "repo": repo,
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0),
                    "open_issues": data.get("open_issues_count", 0),
                    "size_kb": data.get("size", 0),
                    "language": data.get("language", ""),
                    "created_at": data.get("created_at", ""),
                    "pushed_at": data.get("pushed_at", ""),
                    "archived": data.get("archived", False),
                    "contributors_count": contrib_count,
                    "default_branch": data.get("default_branch", ""),
                }
                writer.writerow(row)
                f.flush()

                time.sleep(0.8)  # ~2 calls per repo, stay under limit

        print(f"  Done. Total API calls: {self.calls}")

    # ── Commit date collection ────────────────────────────────────────

    def collect_commit_dates(self, commits: list[dict], existing: set[str] | None = None):
        """Collect commit dates for patch commits."""
        if existing is None:
            existing = set()

        todo = [c for c in commits if c["commit_sha"] not in existing]
        print(f"Commit dates: {len(todo)} to collect ({len(existing)} already done)")

        write_header = not COMMITS_OUT.exists() or COMMITS_OUT.stat().st_size == 0
        with open(COMMITS_OUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "cve_id", "repo", "commit_sha", "commit_date", "author_date",
            ])
            if write_header:
                writer.writeheader()

            for i, c in enumerate(todo):
                if i > 0 and i % 500 == 0:
                    print(f"  {i}/{len(todo)} commits ({self.calls} API calls)")

                repo = c["repo"]
                sha = c["commit_sha"]
                data = self._get(f"{API_BASE}/repos/{repo}/git/commits/{sha}")

                row = {
                    "cve_id": c["cve_id"],
                    "repo": repo,
                    "commit_sha": sha,
                }

                if data:
                    committer = data.get("committer", {})
                    author = data.get("author", {})
                    row["commit_date"] = committer.get("date", "")
                    row["author_date"] = author.get("date", "")

                writer.writerow(row)
                f.flush()

                time.sleep(0.75)

        print(f"  Done. Total API calls: {self.calls}")


def load_existing(path: Path, key_col: str) -> set[str]:
    """Load already-collected entries."""
    if not path.exists():
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(path)
        return set(df[key_col].dropna().astype(str))
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="GitHub PAT")
    parser.add_argument("--repos-only", action="store_true")
    parser.add_argument("--commits-only", action="store_true")
    parser.add_argument("--max-repos", type=int, default=0,
                        help="Max repos to collect (0=all)")
    parser.add_argument("--max-commits", type=int, default=0,
                        help="Max commits to collect (0=all)")
    args = parser.parse_args()

    collector = GitHubCollector(args.token)
    collector.check_rate()

    import pandas as pd

    do_repos = not args.commits_only
    do_commits = not args.repos_only

    if do_repos:
        print("\n=== Collecting repo health metrics ===")
        repos_df = pd.read_csv(REPOS_CSV)
        repo_list = repos_df["repo"].tolist()
        if args.max_repos > 0:
            repo_list = repo_list[:args.max_repos]
        existing = load_existing(HEALTH_OUT, "repo")
        collector.collect_repo_health(repo_list, existing)

    if do_commits:
        print("\n=== Collecting commit dates ===")
        base = pd.read_csv(BASE_CSV)
        commits_df = base[base["has_repo"] & base["commit_sha"].notna()][
            ["cve_id", "github_repo", "commit_sha"]
        ].drop_duplicates(subset=["commit_sha"])
        commits_df = commits_df.rename(columns={"github_repo": "repo"})
        commit_list = commits_df.to_dict("records")
        if args.max_commits > 0:
            commit_list = commit_list[:args.max_commits]
        existing = load_existing(COMMITS_OUT, "commit_sha")
        collector.collect_commit_dates(commit_list, existing)


if __name__ == "__main__":
    main()
