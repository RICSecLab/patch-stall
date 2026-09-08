#!/usr/bin/env python3
"""Classify bot-generated vs human-generated dependency updates.

For a sample of 'manual' propagation events, check GitHub commit/PR
metadata to identify Dependabot/Renovate-generated updates.

Usage:
    python classify_bot_updates.py [--sample-size 500] [--token GITHUB_TOKEN]
    python classify_bot_updates.py --dry-run
    python classify_bot_updates.py --resume --token ghp_xxx

Output: data/sensitivity_bot_classification.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

REPOS_CSV = DATA_DIR / "cross_ecosystem_repos.csv"
NPM_PROP_CSV = DATA_DIR / "npm_propagation.csv"
PYPI_PROP_CSV = DATA_DIR / "pypi_propagation.csv"
MAVEN_PROP_CSV = DATA_DIR / "maven_propagation.csv"

OUTPUT_JSON = DATA_DIR / "sensitivity_bot_classification.json"
INTERMEDIATE_JSON = DATA_DIR / "sensitivity_d1_intermediate.json"

API_BASE = "https://api.github.com"

# Commit author logins that indicate bot-generated updates
DEPENDABOT_LOGINS = {"dependabot[bot]", "dependabot-preview[bot]", "dependabot"}
RENOVATE_LOGINS = {"renovate[bot]", "renovate", "renovatebot"}
OTHER_BOT_PATTERNS = [
    "snyk-bot", "greenkeeper[bot]", "greenkeeper", "pyup-bot", "allcontributors[bot]",
]

# Keywords in commit messages that suggest a dependency bump
BUMP_KEYWORDS = [
    "bump", "update", "upgrade", "chore(deps)", "chore: deps",
    "build(deps)", "fix(deps)", "feat(deps)",
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
        if self.rate_remaining < 100:
            wait = max(self.rate_reset - int(time.time()), 10) + 5
            _log(f"  Rate limit low ({self.rate_remaining} remaining). Sleeping {wait}s...")
            time.sleep(wait)

    def get(self, url: str, params: dict | None = None, retries: int = 3) -> dict | list | None:
        """GET with rate limiting and retry."""
        self._wait_if_limited()
        self.calls += 1

        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                _log(f"    Request error after {retries} attempts: {e}")
                return None

            self._update_rate(resp)

            if resp.status_code == 403:
                # Rate limited or forbidden
                reset = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(reset - int(time.time()), 60) + 5
                _log(f"  403 response. Sleeping {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 202:
                # Stats being computed — caller should retry
                return {"_status": 202}

            if resp.status_code == 404:
                return None

            if resp.status_code != 200:
                _log(f"    HTTP {resp.status_code} for {url[:80]}")
                return None

            try:
                return resp.json()
            except ValueError:
                _log(f"    JSON parse error for {url[:80]}")
                return None

        return None

    def search_commits(self, repo: str, package_name: str, version: str) -> list[dict]:
        """Search commits in a repo that look like dependency bumps for package_name."""
        # GitHub commit search: repo:owner/repo message:keyword
        # We search for the package name in commit messages
        query = f"repo:{repo} {package_name} in:message"
        url = f"{API_BASE}/search/commits"
        params = {"q": query, "per_page": 30, "sort": "committer-date", "order": "desc"}

        data = self.get(url, params=params)
        if not data or not isinstance(data, dict):
            return []
        return data.get("items", [])

    def get_pr_for_commit(self, repo: str, commit_sha: str) -> dict | None:
        """Get the PR associated with a commit."""
        url = f"{API_BASE}/repos/{repo}/commits/{commit_sha}/pulls"
        data = self.get(url)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_repo_commits(self, repo: str, path: str | None = None,
                         since: str | None = None, until: str | None = None,
                         per_page: int = 30) -> list[dict]:
        """Get recent commits for a repo, optionally filtered."""
        url = f"{API_BASE}/repos/{repo}/commits"
        params: dict = {"per_page": per_page}
        if path:
            params["path"] = path
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        data = self.get(url, params=params)
        if data and isinstance(data, list):
            return data
        return []


def classify_actor(login: str | None, name: str | None, message: str | None) -> str:
    """Classify a commit/PR actor as dependabot/renovate/other_bot/human/unknown."""
    if login is None and name is None:
        return "unknown"

    login_lower = (login or "").lower()
    name_lower = (name or "").lower()

    if login_lower in DEPENDABOT_LOGINS or "dependabot" in login_lower:
        return "dependabot"

    if login_lower in RENOVATE_LOGINS or "renovate" in login_lower:
        return "renovate"

    for pattern in OTHER_BOT_PATTERNS:
        if pattern in login_lower or pattern in name_lower:
            return "other_bot"

    if "[bot]" in login_lower:
        return "other_bot"

    # Check commit message for bot-like patterns
    msg_lower = (message or "").lower()
    if "dependabot" in msg_lower:
        return "dependabot"
    if "renovate" in msg_lower:
        return "renovate"

    return "human"


def find_repo_for_package(package: str, ecosystem: str,
                          repos_lookup: dict[str, str],
                          client: "GitHubClient | None" = None) -> str | None:
    """Try to resolve a package name to a GitHub repo slug (owner/repo)."""
    import re

    # Try direct lookup from cross_ecosystem_repos
    key = f"{ecosystem}:{package}"
    if key in repos_lookup:
        return repos_lookup[key]

    # Try bare package name
    if package in repos_lookup:
        return repos_lookup[package]

    # Fallback: query package registry for repo URL
    repo = _registry_repo_lookup(package, ecosystem)
    if repo:
        # Cache for future lookups
        repos_lookup[key] = repo
        repos_lookup[package] = repo
    return repo


def _registry_repo_lookup(package: str, ecosystem: str) -> str | None:
    """Query npm/PyPI registry to find the GitHub repo for a package."""
    import re

    gh_pattern = re.compile(r"github\.com[/:]([^/]+/[^/.]+)")

    try:
        if ecosystem == "npm":
            resp = requests.get(
                f"https://registry.npmjs.org/{package}",
                timeout=(5, 10),
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                repo_info = data.get("repository", {})
                url = repo_info.get("url", "") if isinstance(repo_info, dict) else ""
                m = gh_pattern.search(url)
                if m:
                    return m.group(1).rstrip(".git")

        elif ecosystem == "pypi":
            resp = requests.get(
                f"https://pypi.org/pypi/{package}/json",
                timeout=(5, 10),
            )
            if resp.status_code == 200:
                data = resp.json()
                info = data.get("info", {})
                # Check project_urls and home_page
                for url in list((info.get("project_urls") or {}).values()) + [info.get("home_page", "")]:
                    if url:
                        m = gh_pattern.search(url)
                        if m:
                            return m.group(1).rstrip(".git")

        elif ecosystem == "maven":
            # Maven Central search is harder; try the group/artifact pattern
            parts = package.split(":")
            if len(parts) == 2:
                group_id, artifact_id = parts
                resp = requests.get(
                    f"https://search.maven.org/solrsearch/select?q=g:{group_id}+AND+a:{artifact_id}&rows=1&wt=json",
                    timeout=10,
                )
                if resp.status_code == 200:
                    docs = resp.json().get("response", {}).get("docs", [])
                    # Maven Central doesn't reliably give repo URLs; skip
                    pass

    except (requests.exceptions.RequestException, ValueError, KeyError):
        pass

    return None


def classify_event(
    event: dict,
    ecosystem: str,
    client: GitHubClient,
    repos_lookup: dict[str, str],
    dry_run: bool = False,
) -> str:
    """Classify a single propagation event. Returns classification label."""
    package = event.get("downstream_package", "")
    version = event.get("downstream_version", "")

    if not package:
        return "unknown"

    repo = find_repo_for_package(package, ecosystem, repos_lookup, client)
    if not repo:
        return "unknown"

    if dry_run:
        return "unknown"

    # Strategy: fetch repo commits around the release date (Core API, 5000/hr)
    # Avoids Search API which has a 30/min limit
    # Look for commits around the downstream_release_date
    release_date = event.get("downstream_release_date", "")
    if release_date:
        try:
            since = release_date + "T00:00:00Z"
            recent = client.get_repo_commits(repo, since=since, per_page=20)
        except Exception:
            recent = []

        for commit in recent:
            commit_data = commit.get("commit", {})
            author = commit.get("author") or {}
            committer = commit.get("committer") or {}
            message = commit_data.get("message", "")

            # Check if this commit looks like a dep bump
            msg_lower = message.lower()
            is_bump = any(kw in msg_lower for kw in BUMP_KEYWORDS)
            if not is_bump:
                continue

            for actor in [author, committer]:
                login = actor.get("login", "")
                classification = classify_actor(login, None, message)
                if classification not in ("human", "unknown"):
                    return classification

    # Could not determine from GitHub data — return human as default for
    # events where we found the repo but no bot signature
    if repo:
        return "human"

    return "unknown"


def load_repos_lookup() -> dict[str, str]:
    """Build a package -> repo mapping from cross_ecosystem_repos.csv."""
    import csv
    lookup: dict[str, str] = {}

    if not REPOS_CSV.exists():
        _log(f"  WARNING: {REPOS_CSV} not found, repo lookup will be empty")
        return lookup

    with open(REPOS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row.get("repo", "").strip()
            # Columns may vary; try common names
            for pkg_col in ("package", "package_name", "downstream_package"):
                pkg = row.get(pkg_col, "").strip()
                if pkg and repo:
                    lookup[pkg] = repo
            for eco_col in ("ecosystem", "system"):
                eco = row.get(eco_col, "").strip().lower()
                pkg = row.get("package", row.get("package_name", "")).strip()
                if eco and pkg and repo:
                    lookup[f"{eco}:{pkg}"] = repo

    _log(f"  Loaded {len(lookup)} entries from repos lookup")
    return lookup


def load_manual_events(sample_size: int, seed: int = 42) -> dict[str, list[dict]]:
    """Load manual propagation events per ecosystem and random-sample them."""
    import csv

    ecosystem_files = {
        "npm": NPM_PROP_CSV,
        "pypi": PYPI_PROP_CSV,
        "maven": MAVEN_PROP_CSV,
    }

    rng = random.Random(seed)
    result: dict[str, list[dict]] = {}

    for eco, csv_path in ecosystem_files.items():
        if not csv_path.exists():
            _log(f"  WARNING: {csv_path.name} not found, skipping {eco}")
            result[eco] = []
            continue

        manual_events: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("propagation_type", "") == "manual":
                    manual_events.append(row)

        _log(f"  {eco}: {len(manual_events)} manual events found")

        if len(manual_events) > sample_size:
            sampled = rng.sample(manual_events, sample_size)
        else:
            sampled = manual_events

        result[eco] = sampled
        _log(f"  {eco}: sampling {len(sampled)} events")

    return result


def load_intermediate() -> dict:
    """Load intermediate results if they exist."""
    if INTERMEDIATE_JSON.exists():
        try:
            with open(INTERMEDIATE_JSON, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_intermediate(results: dict):
    """Save intermediate results to disk."""
    with open(INTERMEDIATE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def aggregate_rates(classifications: list[str]) -> dict:
    """Compute rate breakdown from a list of classification strings."""
    total = len(classifications)
    if total == 0:
        return {
            "total": 0, "bot_rate": 0.0,
            "dependabot_rate": 0.0, "renovate_rate": 0.0,
            "other_bot_rate": 0.0, "human_rate": 0.0, "unknown_rate": 0.0,
        }

    counts = {
        "dependabot": sum(1 for c in classifications if c == "dependabot"),
        "renovate": sum(1 for c in classifications if c == "renovate"),
        "other_bot": sum(1 for c in classifications if c == "other_bot"),
        "human": sum(1 for c in classifications if c == "human"),
        "unknown": sum(1 for c in classifications if c == "unknown"),
    }
    bot_count = counts["dependabot"] + counts["renovate"] + counts["other_bot"]

    return {
        "total": total,
        "bot_rate": bot_count / total,
        "dependabot_rate": counts["dependabot"] / total,
        "renovate_rate": counts["renovate"] / total,
        "other_bot_rate": counts["other_bot"] / total,
        "human_rate": counts["human"] / total,
        "unknown_rate": counts["unknown"] / total,
        "counts": counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Classify bot vs human dependency updates in 'manual' propagation events"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of manual events to sample per ecosystem (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and sample events but do not make API calls; classify all as 'unknown'",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from intermediate results saved by a previous run",
    )
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="Seconds to sleep between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--ecosystem", choices=["npm", "pypi", "maven"],
        help="Process only one ecosystem (default: all three)",
    )
    args = parser.parse_args()

    token = args.token or None
    if not token and not args.dry_run:
        _log("WARNING: No GitHub token provided. Unauthenticated API allows only 60 req/hr.")

    _log("Bot Classification Script")
    _log(f"  Sample size per ecosystem: {args.sample_size}")
    _log(f"  Dry run: {args.dry_run}")
    _log(f"  Resume: {args.resume}")

    # Load repos lookup
    repos_lookup = load_repos_lookup()

    # Load manual events
    events_by_eco = load_manual_events(args.sample_size, seed=args.seed)

    # Filter to specified ecosystem if requested
    if args.ecosystem:
        events_by_eco = {args.ecosystem: events_by_eco.get(args.ecosystem, [])}

    # Load intermediate results for resume
    intermediate: dict = {}
    if args.resume:
        intermediate = load_intermediate()
        _log(f"  Loaded intermediate results for {list(intermediate.keys())}")

    client = GitHubClient(token)

    # Per-ecosystem classification results
    # Structure: {eco: {event_key: classification_label}}
    eco_results: dict[str, dict[str, str]] = {}
    for eco in events_by_eco:
        eco_results[eco] = dict(intermediate.get(eco, {}))

    total_events = sum(len(v) for v in events_by_eco.values())
    processed = 0

    for eco, events in events_by_eco.items():
        _log(f"\n--- Ecosystem: {eco.upper()} ({len(events)} events) ---")

        for i, event in enumerate(events):
            # Build a stable event key for resume capability
            event_key = (
                f"{event.get('downstream_package','?')}|"
                f"{event.get('downstream_version','?')}|"
                f"{event.get('osv_id', event.get('cve_id','?'))}"
            )

            if event_key in eco_results[eco]:
                processed += 1
                continue  # already classified in a previous run

            classification = classify_event(
                event, eco, client, repos_lookup, dry_run=args.dry_run
            )
            eco_results[eco][event_key] = classification
            processed += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(events):
                pct = processed / total_events * 100
                _log(
                    f"  [{eco}] {i+1}/{len(events)} done "
                    f"({pct:.1f}% overall, {client.calls} API calls)"
                )

                # Save intermediate checkpoint every 10 events
                save_intermediate(eco_results)

            if not args.dry_run:
                time.sleep(args.pause)

    # Aggregate per ecosystem
    summary: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "total_api_calls": client.calls,
        "ecosystems": {},
    }

    all_classifications: list[str] = []

    for eco in events_by_eco:
        classifications = list(eco_results[eco].values())
        all_classifications.extend(classifications)
        rates = aggregate_rates(classifications)
        summary["ecosystems"][eco] = rates

        _log(f"\n{eco.upper()} results:")
        _log(f"  total classified : {rates['total']}")
        _log(f"  bot_rate         : {rates['bot_rate']:.1%}")
        _log(f"  dependabot_rate  : {rates['dependabot_rate']:.1%}")
        _log(f"  renovate_rate    : {rates['renovate_rate']:.1%}")
        _log(f"  other_bot_rate   : {rates['other_bot_rate']:.1%}")
        _log(f"  human_rate       : {rates['human_rate']:.1%}")
        _log(f"  unknown_rate     : {rates['unknown_rate']:.1%}")

    summary["overall"] = aggregate_rates(all_classifications)

    # Save final output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _log(f"\nSaved results to {OUTPUT_JSON}")
    _log(f"Total API calls: {client.calls}")

    # Clean up intermediate file on successful completion
    if INTERMEDIATE_JSON.exists() and not args.dry_run:
        INTERMEDIATE_JSON.unlink()
        _log("Removed intermediate checkpoint file")


if __name__ == "__main__":
    main()
