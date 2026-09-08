#!/usr/bin/env python3
"""Collect monthly contributor counts via GitHub Stats API.

For each repository in the dataset, query the contributor statistics endpoint
to retrieve all-time weekly commit data, then aggregate to monthly counts.
This enables time-varying health covariates in Cox proportional hazards models.

The stats/contributors endpoint returns all historical weekly data in a single
API call, so the total number of calls equals the number of unique repos
(~3,000 calls ≈ 36 min at the authenticated rate of 5,000 req/hr).

Usage:
    python collect_monthly_health.py --token ghp_xxx
    python collect_monthly_health.py --token ghp_xxx --resume
    python collect_monthly_health.py --token ghp_xxx --max-repos 100

Output: data/monthly_health.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

REPOS_CSV = DATA_DIR / "cross_ecosystem_repos.csv"
OUTPUT_CSV = DATA_DIR / "monthly_health.csv"
CHECKPOINT_CSV = DATA_DIR / "monthly_health_checkpoint.csv"

API_BASE = "https://api.github.com"

OUTPUT_COLUMNS = [
    "repo",
    "year_month",
    "contributor_count",
    "commit_count",
]


def _log(msg: str, file=sys.stderr):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=file, flush=True)


class GitHubClient:
    def __init__(self, token: str | None):
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "patch-stall-replication",
        }
        if token:
            headers["Authorization"] = f"token {token}"
        self.session.headers.update(headers)
        self.calls = 0
        self.rate_remaining = 5000
        self.rate_reset = 0

    def _update_rate(self, resp: requests.Response):
        self.rate_remaining = int(resp.headers.get("X-RateLimit-Remaining", self.rate_remaining))
        self.rate_reset = int(resp.headers.get("X-RateLimit-Reset", self.rate_reset))

    def _wait_if_limited(self):
        if self.rate_remaining < 30:
            wait = max(self.rate_reset - int(time.time()), 10) + 5
            _log(f"  Rate limit low ({self.rate_remaining} remaining). Sleeping {wait}s...")
            time.sleep(wait)

    def get_raw(self, url: str, retries: int = 3) -> requests.Response | None:
        """Make a GET request with retry logic. Returns raw Response."""
        self._wait_if_limited()
        self.calls += 1

        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    _log(f"    Network error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                _log(f"    Network error after {retries} attempts: {e}")
                return None

            self._update_rate(resp)
            return resp

        return None

    def check_rate_limit(self):
        """Query and log current rate limit status."""
        resp = self.get_raw(f"{API_BASE}/rate_limit")
        if resp and resp.status_code == 200:
            data = resp.json()
            core = data.get("resources", {}).get("core", {})
            reset_dt = datetime.fromtimestamp(core.get("reset", 0), tz=timezone.utc)
            _log(
                f"  Rate limit: {core.get('remaining')}/{core.get('limit')} "
                f"(resets at {reset_dt.strftime('%H:%M:%S UTC')})"
            )


def get_contributor_stats(client: GitHubClient, repo: str,
                          max_retries: int = 5) -> list[dict] | None:
    """Fetch /repos/{repo}/stats/contributors with 202-retry handling.

    The endpoint may return HTTP 202 while GitHub computes stats in the
    background. We retry up to max_retries times with exponential back-off.

    Returns a list of contributor objects, each with:
      - author: {login, ...}
      - weeks: [{w: unix_timestamp, a: additions, d: deletions, c: commits}, ...]

    Returns None if the repo is inaccessible (404/403) or stats unavailable.
    """
    url = f"{API_BASE}/repos/{repo}/stats/contributors"

    for attempt in range(max_retries):
        resp = client.get_raw(url)
        if resp is None:
            return None

        if resp.status_code == 404:
            # Repo deleted or private
            return None

        if resp.status_code in (401, 403):
            _log(f"    Access denied for {repo} (HTTP {resp.status_code})")
            return None

        if resp.status_code == 202:
            # Stats are being computed; wait and retry
            wait = min(5 * (attempt + 1), 30)
            _log(f"    202 for {repo} (attempt {attempt+1}/{max_retries}). "
                 f"Stats being computed, retrying in {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            _log(f"    HTTP {resp.status_code} for {repo}")
            return None

        try:
            data = resp.json()
        except ValueError:
            _log(f"    JSON parse error for {repo}")
            return None

        if not isinstance(data, list):
            return None

        return data

    _log(f"    Gave up waiting for stats on {repo} after {max_retries} retries")
    return None


def week_timestamp_to_year_month(unix_ts: int) -> str:
    """Convert a Unix week timestamp (Sunday start) to 'YYYY-MM' string."""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime("%Y-%m")


def aggregate_to_monthly(contributor_data: list[dict]) -> list[dict]:
    """Aggregate all-time weekly contributor stats into monthly rows.

    Each month row contains:
      - year_month: 'YYYY-MM'
      - contributor_count: unique contributors with at least one commit that month
      - commit_count: total commits across all contributors that month
    """
    # month_stats[year_month] = {"committers": set(), "commit_count": int}
    month_stats: dict[str, dict] = {}

    for contributor in contributor_data:
        author = contributor.get("author") or {}
        login = author.get("login", "anonymous")

        for week in contributor.get("weeks", []):
            commits_this_week = week.get("c", 0)
            if commits_this_week == 0:
                continue

            week_ts = week.get("w", 0)
            if week_ts == 0:
                continue

            ym = week_timestamp_to_year_month(week_ts)

            if ym not in month_stats:
                month_stats[ym] = {"committers": set(), "commit_count": 0}

            month_stats[ym]["committers"].add(login)
            month_stats[ym]["commit_count"] += commits_this_week

    rows = []
    for ym in sorted(month_stats):
        entry = month_stats[ym]
        rows.append({
            "year_month": ym,
            "contributor_count": len(entry["committers"]),
            "commit_count": entry["commit_count"],
        })

    return rows


def load_repos() -> list[str]:
    """Load unique repo slugs from cross_ecosystem_repos.csv."""
    import csv as _csv

    if not REPOS_CSV.exists():
        _log(f"ERROR: {REPOS_CSV} not found")
        sys.exit(1)

    repos: list[str] = []
    seen: set[str] = set()

    with open(REPOS_CSV, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            repo = row.get("repo", "").strip()
            if repo and repo not in seen:
                repos.append(repo)
                seen.add(repo)

    _log(f"Loaded {len(repos)} unique repos from {REPOS_CSV.name}")
    return repos


def load_completed_repos() -> set[str]:
    """Load set of repos already written to the output CSV."""
    done: set[str] = set()

    for path in (OUTPUT_CSV, CHECKPOINT_CSV):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    repo = row.get("repo", "").strip()
                    if repo:
                        done.add(repo)
        except (OSError, csv.Error):
            pass

    return done


def write_rows(rows: list[dict], path: Path, append: bool = True):
    """Write monthly health rows to CSV."""
    write_header = not path.exists() or path.stat().st_size == 0 or not append
    mode = "a" if append else "w"

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Collect monthly contributor counts from GitHub Stats API"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previously collected data (skip already-done repos)",
    )
    parser.add_argument(
        "--max-repos", type=int, default=0,
        help="Process at most N repos (0 = all)",
    )
    parser.add_argument(
        "--pause", type=float, default=0.75,
        help="Seconds to sleep between API calls (default: 0.75 ≈ ~4800 calls/hr)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Flush intermediate checkpoint every N repos (default: 100)",
    )
    args = parser.parse_args()

    token = args.token or None
    if not token:
        _log("WARNING: No GitHub token provided. Unauthenticated rate limit is 60 req/hr.")

    _log("Monthly Health Collector")
    _log(f"  Output: {OUTPUT_CSV}")
    _log(f"  Resume: {args.resume}")
    _log(f"  Max repos: {args.max_repos if args.max_repos > 0 else 'all'}")
    _log(f"  Pause between calls: {args.pause}s")

    client = GitHubClient(token)
    client.check_rate_limit()

    # Load repo list
    repos = load_repos()

    # Apply max-repos limit
    if args.max_repos > 0:
        repos = repos[:args.max_repos]
        _log(f"Limiting to first {len(repos)} repos")

    # Load already-completed repos for resume
    completed: set[str] = set()
    if args.resume:
        completed = load_completed_repos()
        _log(f"Resume: {len(completed)} repos already done")

    todo = [r for r in repos if r not in completed]
    _log(f"Repos to process: {len(todo)} (skipping {len(repos) - len(todo)} done)")

    if not todo:
        _log("Nothing to do.")
        return

    # Ensure output file has a header if starting fresh
    if not args.resume and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        _log(f"Removed existing {OUTPUT_CSV.name} for fresh run")

    stats = {
        "processed": 0,
        "skipped_no_data": 0,
        "skipped_inaccessible": 0,
        "rows_written": 0,
    }
    start_time = time.time()
    pending_rows: list[dict] = []

    for i, repo in enumerate(todo):
        _log(f"  [{i+1}/{len(todo)}] {repo}")

        contributor_data = get_contributor_stats(client, repo)
        if contributor_data is None:
            _log(f"    Skipping (inaccessible or no stats)")
            stats["skipped_inaccessible"] += 1
            stats["processed"] += 1
            # Write a sentinel row so we don't re-attempt on resume
            pending_rows.append({
                "repo": repo,
                "year_month": "",
                "contributor_count": -1,
                "commit_count": -1,
            })
        elif len(contributor_data) == 0:
            _log(f"    No contributor data returned")
            stats["skipped_no_data"] += 1
            stats["processed"] += 1
            pending_rows.append({
                "repo": repo,
                "year_month": "",
                "contributor_count": 0,
                "commit_count": 0,
            })
        else:
            monthly_rows = aggregate_to_monthly(contributor_data)
            _log(
                f"    {len(contributor_data)} contributors, "
                f"{len(monthly_rows)} months of data"
            )
            for row in monthly_rows:
                pending_rows.append({
                    "repo": repo,
                    "year_month": row["year_month"],
                    "contributor_count": row["contributor_count"],
                    "commit_count": row["commit_count"],
                })

            stats["rows_written"] += len(monthly_rows)
            stats["processed"] += 1

        # Periodic checkpoint flush
        if (i + 1) % args.checkpoint_interval == 0 or (i + 1) == len(todo):
            if pending_rows:
                # Write to checkpoint during collection, merge to output at end
                write_rows(pending_rows, CHECKPOINT_CSV, append=True)
                _log(
                    f"  Checkpoint: flushed {len(pending_rows)} rows "
                    f"({stats['processed']} repos done, "
                    f"{client.calls} API calls, "
                    f"{time.time() - start_time:.0f}s elapsed)"
                )
                pending_rows = []

        time.sleep(args.pause)

    # Merge checkpoint into final output
    _log("\nMerging checkpoint into final output...")
    if CHECKPOINT_CSV.exists() and CHECKPOINT_CSV.stat().st_size > 0:
        with open(CHECKPOINT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_checkpoint_rows = list(reader)

        write_rows(all_checkpoint_rows, OUTPUT_CSV, append=False)
        _log(f"Wrote {len(all_checkpoint_rows)} rows to {OUTPUT_CSV}")

        # Clean up checkpoint file
        CHECKPOINT_CSV.unlink()
        _log("Removed checkpoint file")
    else:
        _log("No checkpoint data to merge")

    # Final summary
    elapsed = time.time() - start_time
    _log(f"\n{'=' * 60}")
    _log(f"MONTHLY HEALTH COLLECTION COMPLETE")
    _log(f"{'=' * 60}")
    _log(f"  Repos processed     : {stats['processed']}")
    _log(f"  Inaccessible repos  : {stats['skipped_inaccessible']}")
    _log(f"  No-data repos       : {stats['skipped_no_data']}")
    _log(f"  Total rows written  : {stats['rows_written']}")
    _log(f"  Total API calls     : {client.calls}")
    _log(f"  Elapsed time        : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    _log(f"  Output file         : {OUTPUT_CSV}")

    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        _log(f"\n  Final CSV: {row_count} data rows")


if __name__ == "__main__":
    main()
