#!/usr/bin/env python3
"""
collect_git_tags.py - Collect git tag dates from the GitHub API.

Git tags are the second source of the dual-source patch-date metric. They are
what makes Maven usable: only 14 of 1,559 Maven observations have a registry
date, so without tags the ecosystem drops out of the survival models.

Usage:
    GITHUB_TOKEN=ghp_xxx python scripts/collect_git_tags.py
    python scripts/collect_git_tags.py --token ghp_xxx
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = BASE_DIR / "data" / "cross_ecosystem_base.csv"
OUTPUT_CSV = BASE_DIR / "data" / "git_tag_dates.csv"
CHECKPOINT_CSV = BASE_DIR / "data" / "git_tag_dates_checkpoint.csv"

API_BASE = "https://api.github.com"
PER_PAGE = 100  # max for GitHub API

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.rate_remaining = 5000
        self.rate_reset = 0
        self.request_count = 0

    def _check_rate_limit(self):
        """Sleep if we're about to hit the rate limit."""
        if self.rate_remaining < 50:
            wait = max(0, self.rate_reset - time.time()) + 5
            if wait > 0:
                print(f"  [rate-limit] {self.rate_remaining} remaining, sleeping {wait:.0f}s until reset...")
                time.sleep(wait)

    def _update_rate_info(self, resp: requests.Response):
        """Update rate limit info from response headers."""
        self.rate_remaining = int(resp.headers.get("X-RateLimit-Remaining", self.rate_remaining))
        self.rate_reset = int(resp.headers.get("X-RateLimit-Reset", self.rate_reset))

    def get(self, url: str, params: dict = None, retries: int = 3) -> requests.Response:
        """Make a GET request with rate-limit awareness and retry logic."""
        self._check_rate_limit()
        for attempt in range(retries):
            try:
                self.request_count += 1
                resp = self.session.get(url, params=params, timeout=30)
                self._update_rate_info(resp)
                return resp
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ReadTimeout) as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    [retry] {type(e).__name__} on attempt {attempt+1}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    [error] {type(e).__name__} after {retries} attempts: {e}")
                    # Return a fake 500 response
                    fake = requests.Response()
                    fake.status_code = 500
                    fake._content = b"{}"
                    return fake

    def get_paginated(self, url: str, params: dict = None) -> list:
        """Fetch all pages of a paginated endpoint."""
        if params is None:
            params = {}
        params["per_page"] = PER_PAGE
        all_items = []

        while url:
            resp = self.get(url, params=params)
            if resp.status_code == 404:
                return []  # repo not found or no tags
            if resp.status_code == 409:
                return []  # empty repo
            if resp.status_code == 403:
                # Could be rate limit or repo access issue
                if self.rate_remaining < 5:
                    self._check_rate_limit()
                    continue
                return []
            if resp.status_code != 200:
                print(f"    [warn] HTTP {resp.status_code} for {url}")
                return all_items

            data = resp.json()
            if isinstance(data, list):
                all_items.extend(data)
            else:
                break

            # Follow pagination via Link header
            link = resp.headers.get("Link", "")
            url = None
            params = {}  # params are embedded in the next URL
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split("<")[1].split(">")[0]
                    break

        return all_items

    def get_all_tags(self, owner: str, repo: str) -> list:
        """
        Get all tags for a repo via git/refs/tags endpoint.
        Returns list of dicts: {ref, tag_name, sha, type}
        """
        url = f"{API_BASE}/repos/{owner}/{repo}/git/refs/tags"
        refs = self.get_paginated(url)
        tags = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            ref_name = ref.get("ref", "")
            if not ref_name.startswith("refs/tags/"):
                continue
            tag_name = ref_name[len("refs/tags/"):]
            obj = ref.get("object", {})
            tags.append({
                "tag_name": tag_name,
                "sha": obj.get("sha", ""),
                "type": obj.get("type", ""),  # "tag" (annotated) or "commit" (lightweight)
            })
        return tags

    def get_tag_date(self, owner: str, repo: str, sha: str, obj_type: str) -> tuple:
        """
        Resolve the date of a tag.
        Returns (date_str, tag_type) where tag_type is "annotated" or "lightweight".
        """
        if obj_type == "tag":
            # Annotated tag: get tagger date
            url = f"{API_BASE}/repos/{owner}/{repo}/git/tags/{sha}"
            resp = self.get(url)
            if resp.status_code == 200:
                data = resp.json()
                tagger = data.get("tagger", {})
                date = tagger.get("date", "")
                if date:
                    return (date, "annotated")
                # Fall back: resolve to commit
                commit_sha = data.get("object", {}).get("sha", "")
                if commit_sha:
                    return self._get_commit_date(owner, repo, commit_sha)
            return ("", "annotated")
        else:
            # Lightweight tag: get commit date
            return self._get_commit_date(owner, repo, sha)

    def _get_commit_date(self, owner: str, repo: str, sha: str) -> tuple:
        """Get committer date for a commit."""
        url = f"{API_BASE}/repos/{owner}/{repo}/git/commits/{sha}"
        resp = self.get(url)
        if resp.status_code == 200:
            data = resp.json()
            committer = data.get("committer", {})
            date = committer.get("date", "")
            return (date, "lightweight")
        return ("", "lightweight")


# ---------------------------------------------------------------------------
# Version matching
# ---------------------------------------------------------------------------

def match_version_to_tag(version: str, tag_name: str, package_name: str = "",
                          ecosystem: str = "") -> str:
    """
    Try to match a first_patched_version to a tag_name.
    Returns match_type string or "" if no match.
    """
    v = version.strip()
    t = tag_name.strip()

    if not v or not t:
        return ""

    # 1. Exact match
    if t == v:
        return "exact"

    # 2. v-prefix: tag is "v" + version
    if t == "v" + v:
        return "v-prefix"
    # Also try: version starts with v, tag doesn't
    if v.startswith("v") and t == v[1:]:
        return "v-prefix-inverse"

    # 3. release-prefix patterns
    for prefix in ("release-", "rel-", "release/", "rel/"):
        if t == prefix + v or t == prefix + "v" + v:
            return "release-prefix"

    # 4. Maven artifactId-version pattern
    if ecosystem == "Maven" and ":" in package_name:
        artifact_id = package_name.split(":")[-1]
        # e.g., "spring-core-5.3.1" or "spring-core/5.3.1"
        for sep in ("-", "/", "_"):
            if t == f"{artifact_id}{sep}{v}" or t == f"{artifact_id}{sep}v{v}":
                return "artifact-version"

    # 5. Project-prefix: tag ends with the version after some prefix
    # e.g., "apache-tomcat-9.0.1", "jetty-9.4.1"
    # Match if tag ends with -version or /version or _version
    for sep in ("-", "/", "_"):
        if t.endswith(sep + v):
            return "project-prefix"
        if t.endswith(sep + "v" + v):
            return "project-prefix"

    return ""


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_input_data(path: Path) -> list:
    """Load and filter cross_ecosystem_base.csv."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["has_patch"] == "True" and row["has_repo"] == "True":
                rows.append(row)
    return rows


def build_repo_version_map(rows: list) -> dict:
    """
    Build mapping: github_repo -> set of (ghsa_id, ecosystem, package_name, version) tuples.
    """
    repo_map = defaultdict(set)
    for row in rows:
        repo = row["github_repo"]
        key = (row["ghsa_id"], row["ecosystem"], row["package_name"],
               row["first_patched_version"])
        repo_map[repo].add(key)
    return repo_map


def load_checkpoint(path: Path) -> set:
    """Load set of already-processed repos from checkpoint file."""
    processed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["github_repo"])
    return processed


def load_existing_results(path: Path) -> list:
    """Load existing results from checkpoint."""
    results = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    return results


OUTPUT_FIELDS = [
    "ghsa_id", "ecosystem", "package_name", "first_patched_version",
    "github_repo", "tag_name", "tag_date", "tag_type", "match_type"
]


def save_results(results: list, path: Path):
    """Write results to CSV."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Collect git tag dates from GitHub API")
    parser.add_argument("--token", help="GitHub personal access token")
    args = parser.parse_args()

    # Get token
    token = os.environ.get("GITHUB_TOKEN") or args.token
    if not token:
        print("ERROR: No GitHub token provided. Set GITHUB_TOKEN env var or use --token.")
        sys.exit(1)

    print(f"Loading data from {INPUT_CSV}")
    rows = load_input_data(INPUT_CSV)
    print(f"  Filtered rows (has_patch=True, has_repo=True): {len(rows)}")

    repo_map = build_repo_version_map(rows)
    all_repos = sorted(repo_map.keys())
    print(f"  Unique repos to process: {len(all_repos)}")

    # Ecosystem breakdown
    eco_repos = defaultdict(set)
    for repo, entries in repo_map.items():
        for _, eco, _, _ in entries:
            eco_repos[eco].add(repo)
    for eco in sorted(eco_repos):
        print(f"    {eco}: {len(eco_repos[eco])} repos")

    # Load checkpoint
    processed_repos = load_checkpoint(CHECKPOINT_CSV)
    results = load_existing_results(CHECKPOINT_CSV)
    if processed_repos:
        print(f"  Resuming from checkpoint: {len(processed_repos)} repos already done, {len(results)} results loaded")

    client = GitHubClient(token)

    # Verify token
    resp = client.get(f"{API_BASE}/rate_limit")
    if resp.status_code != 200:
        print(f"ERROR: Token validation failed (HTTP {resp.status_code})")
        sys.exit(1)
    rate_info = resp.json().get("rate", {})
    print(f"  API rate limit: {rate_info.get('remaining')}/{rate_info.get('limit')}")

    # Tag cache: repo -> list of {tag_name, sha, type}
    tag_cache = {}

    repos_done = 0
    repos_skipped = 0
    repos_total = len(all_repos)
    match_count = 0
    no_match_count = 0
    error_count = 0

    start_time = time.time()

    for i, repo in enumerate(all_repos):
        if repo in processed_repos:
            repos_skipped += 1
            continue

        try:
            owner_name = repo.split("/")
            if len(owner_name) != 2:
                print(f"  [skip] Invalid repo format: {repo}")
                repos_done += 1
                processed_repos.add(repo)
                continue
            owner, name = owner_name

            # Get all tags for repo
            if repo not in tag_cache:
                tags = client.get_all_tags(owner, name)
                tag_cache[repo] = tags
            else:
                tags = tag_cache[repo]

            if not tags:
                # No tags found; still mark as processed
                repos_done += 1
                processed_repos.add(repo)
                # Record no-match entries
                for ghsa_id, eco, pkg, ver in repo_map[repo]:
                    results.append({
                        "ghsa_id": ghsa_id,
                        "ecosystem": eco,
                        "package_name": pkg,
                        "first_patched_version": ver,
                        "github_repo": repo,
                        "tag_name": "",
                        "tag_date": "",
                        "tag_type": "",
                        "match_type": "no_tags",
                    })
                    no_match_count += 1

                if (repos_done + repos_skipped) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{repos_done + repos_skipped}/{repos_total}] "
                          f"{elapsed:.0f}s elapsed, "
                          f"{client.request_count} API calls, "
                          f"rate remaining: {client.rate_remaining}, "
                          f"matches: {match_count}")
                continue

            # Build a quick lookup: tag_name -> tag_info
            tag_lookup = {t["tag_name"]: t for t in tags}

            # For each advisory+version associated with this repo, try to match
            entries = repo_map[repo]
            matched_tags_for_repo = {}  # tag_name -> (date, type) cache

            for ghsa_id, eco, pkg, ver in entries:
                best_match = ""
                best_tag = ""

                for tag_info in tags:
                    mt = match_version_to_tag(ver, tag_info["tag_name"], pkg, eco)
                    if mt:
                        # Prefer exact > v-prefix > others
                        priority = {"exact": 0, "v-prefix": 1, "v-prefix-inverse": 2,
                                    "release-prefix": 3, "artifact-version": 4,
                                    "project-prefix": 5}
                        if not best_match or priority.get(mt, 99) < priority.get(best_match, 99):
                            best_match = mt
                            best_tag = tag_info["tag_name"]

                if best_match and best_tag:
                    # Resolve tag date (use cache)
                    if best_tag not in matched_tags_for_repo:
                        ti = tag_lookup[best_tag]
                        date, ttype = client.get_tag_date(owner, name, ti["sha"], ti["type"])
                        matched_tags_for_repo[best_tag] = (date, ttype)

                    date, ttype = matched_tags_for_repo[best_tag]
                    results.append({
                        "ghsa_id": ghsa_id,
                        "ecosystem": eco,
                        "package_name": pkg,
                        "first_patched_version": ver,
                        "github_repo": repo,
                        "tag_name": best_tag,
                        "tag_date": date,
                        "tag_type": ttype,
                        "match_type": best_match,
                    })
                    match_count += 1
                else:
                    results.append({
                        "ghsa_id": ghsa_id,
                        "ecosystem": eco,
                        "package_name": pkg,
                        "first_patched_version": ver,
                        "github_repo": repo,
                        "tag_name": "",
                        "tag_date": "",
                        "tag_type": "",
                        "match_type": "no_match",
                    })
                    no_match_count += 1

            repos_done += 1
            processed_repos.add(repo)

        except Exception as e:
            print(f"  [error] Exception processing {repo}: {type(e).__name__}: {e}")
            error_count += 1
            repos_done += 1
            processed_repos.add(repo)
            # Record error entries
            for ghsa_id, eco, pkg, ver in repo_map[repo]:
                results.append({
                    "ghsa_id": ghsa_id,
                    "ecosystem": eco,
                    "package_name": pkg,
                    "first_patched_version": ver,
                    "github_repo": repo,
                    "tag_name": "",
                    "tag_date": "",
                    "tag_type": "",
                    "match_type": "error",
                })
                no_match_count += 1

        # Progress logging every 50 repos
        total_processed = repos_done + repos_skipped
        if total_processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = repos_done / max(elapsed, 1) * 3600
            print(f"  [{total_processed}/{repos_total}] "
                  f"{elapsed:.0f}s elapsed, "
                  f"{client.request_count} API calls, "
                  f"rate remaining: {client.rate_remaining}, "
                  f"matches: {match_count}, "
                  f"~{rate:.0f} repos/hr")

        # Save checkpoint every 100 newly processed repos
        if repos_done > 0 and repos_done % 100 == 0:
            print(f"  [checkpoint] Saving {len(results)} results...")
            save_results(results, CHECKPOINT_CSV)

    # Final save
    save_results(results, OUTPUT_CSV)
    if CHECKPOINT_CSV.exists():
        save_results(results, CHECKPOINT_CSV)  # also update checkpoint

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Total API requests: {client.request_count}")
    print(f"  Repos processed: {repos_done} (skipped from checkpoint: {repos_skipped})")
    print(f"  Total result rows: {len(results)}")
    print(f"  Matched: {match_count}")
    print(f"  Not matched: {no_match_count}")
    print()

    # Per-ecosystem stats
    eco_match = defaultdict(int)
    eco_total = defaultdict(int)
    eco_nomatch_reasons = defaultdict(lambda: defaultdict(int))
    for r in results:
        eco = r["ecosystem"]
        eco_total[eco] += 1
        if r["match_type"] not in ("no_match", "no_tags", ""):
            eco_match[eco] += 1
        else:
            eco_nomatch_reasons[eco][r["match_type"]] += 1

    print("  Match rates by ecosystem:")
    for eco in sorted(eco_total):
        total = eco_total[eco]
        matched = eco_match[eco]
        pct = matched / total * 100 if total else 0
        print(f"    {eco}: {matched}/{total} ({pct:.1f}%)")
        for reason, count in sorted(eco_nomatch_reasons[eco].items()):
            print(f"      {reason}: {count}")

    # Match type distribution
    print("\n  Match type distribution:")
    mt_counts = defaultdict(int)
    for r in results:
        mt_counts[r["match_type"]] += 1
    for mt, count in sorted(mt_counts.items(), key=lambda x: -x[1]):
        print(f"    {mt}: {count}")

    print(f"\n  Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
