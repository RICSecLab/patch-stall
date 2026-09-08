#!/usr/bin/env python3
"""Collect patch propagation data from deps.dev for PyPI and Maven ecosystems.

Strategy:
  For each advisory with a patched version:
    1. Find a vulnerable version (latest version before the patch)
    2. Query deps.dev internal API for up to 100 direct dependents of that version
    3. For each dependent, use REST API to:
       a. Get version constraint on the upstream package
       b. Check if the constraint auto-accepts the patched version
       c. Find the first downstream release that accepts the patch
    4. Classify propagation: auto / manual / censored

Outputs: data/pypi_propagation.csv and data/maven_propagation.csv
with the same schema as npm_propagation.csv.

Usage:
  python scripts/collect_propagation_depsdev.py --test        # 5+5 advisory test
  python scripts/collect_propagation_depsdev.py               # full collection
  python scripts/collect_propagation_depsdev.py --ecosystem pypi   # single ecosystem
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
BASE_CSV = DATA_DIR / "cross_ecosystem_base.csv"

REST_API = "https://api.deps.dev/v3alpha"
INTERNAL_API = "https://deps.dev/_/s"

OBSERVATION_CUTOFF = "2026-03-01"  # right-censoring date

# When checking subsequent downstream versions for manual propagation,
# limit how many we actually fetch dependencies for (each = 1 API call)
MAX_SUBSEQUENT_VERSIONS_TO_CHECK = 15

OUTPUT_COLUMNS = [
    "downstream_package",
    "downstream_version",
    "downstream_release_date",
    "upstream_package",
    "cve_id",
    "osv_id",
    "cvss_severity",
    "cwe_id",
    "patched_version",
    "patch_release_date",
    "first_safe_release_date",
    "propagation_days",
    "propagation_type",
    "dependency_type",
    "version_constraint_at_t0",
    "constraint_type",
    "requires_major_upgrade",
]

# ── Caches ───────────────────────────────────────────────────────────────
_version_list_cache: dict[str, list[dict]] = {}  # "system/package" -> versions
_deps_cache: dict[str, list[dict]] = {}  # "system/package/version" -> edges

REQUEST_DELAY = 0.15  # seconds between API requests
_api_calls = 0


def _log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── HTTP helpers ─────────────────────────────────────────────────────────

def _get_json(url: str, timeout: int = 15, retries: int = 2) -> dict | None:
    """GET a URL and return parsed JSON, or None on error."""
    global _api_calls
    for attempt in range(retries + 1):
        _api_calls += 1
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", "patch-stall-replication")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                _log(f"    Rate limited (429), sleeping 10s... (attempt {attempt+1})")
                time.sleep(10)
                continue
            _log(f"    HTTP {e.code} for {url[:80]}...")
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries:
                _log(f"    Timeout/network error, retry {attempt+1}/{retries}...")
                time.sleep(2)
                continue
            _log(f"    Network error after {retries+1} attempts: {e}")
            return None
        except Exception as e:
            _log(f"    Unexpected error: {type(e).__name__}: {e}")
            return None
    return None


def _encode(s: str) -> str:
    """URL-encode a string (safe for path segments)."""
    return urllib.parse.quote(s, safe="")


# ── deps.dev API wrappers ────────────────────────────────────────────────

def get_package_versions(system: str, package: str) -> list[dict]:
    """Get all versions of a package with publishedAt timestamps.

    Returns list of {"version": str, "publishedAt": str, "isDefault": bool}
    sorted by publishedAt ascending.
    """
    cache_key = f"{system}/{package}"
    if cache_key in _version_list_cache:
        return _version_list_cache[cache_key]

    url = f"{REST_API}/systems/{system}/packages/{_encode(package)}"
    data = _get_json(url)
    time.sleep(REQUEST_DELAY)

    if not data or "versions" not in data:
        _version_list_cache[cache_key] = []
        return []

    versions = []
    for v in data["versions"]:
        vk = v.get("versionKey", {})
        ver = vk.get("version", "")
        pub = v.get("publishedAt", "")
        if ver and pub:
            versions.append({
                "version": ver,
                "publishedAt": pub,
                "isDefault": v.get("isDefault", False),
            })

    # Sort by publishedAt (ISO 8601 sorts lexicographically)
    versions.sort(key=lambda x: x["publishedAt"])
    _version_list_cache[cache_key] = versions
    return versions


def get_direct_dependents(system: str, package: str, version: str) -> list[dict]:
    """Get up to 100 direct dependents from the internal deps.dev API.

    Returns list of {"package": str, "version": str, "system": str}.
    """
    sys_name = system.lower()
    enc_pkg = _encode(package)
    enc_ver = _encode(version)
    url = f"{INTERNAL_API}/{sys_name}/p/{enc_pkg}/v/{enc_ver}/dependents"

    data = _get_json(url)
    time.sleep(REQUEST_DELAY)

    if not data:
        return []

    result = []
    for dep in data.get("directSample", []):
        pkg_info = dep.get("package", {})
        result.append({
            "package": pkg_info.get("name", ""),
            "version": dep.get("version", ""),
            "system": pkg_info.get("system", system).lower(),
        })

    return result


def get_dependency_edges(system: str, package: str, version: str) -> list[dict]:
    """Get dependency edges for a specific version.

    Returns list of {"dep_package": str, "requirement": str, "dep_type": str}.
    """
    cache_key = f"{system}/{package}/{version}"
    if cache_key in _deps_cache:
        return _deps_cache[cache_key]

    enc_pkg = _encode(package)
    enc_ver = _encode(version)
    url = f"{REST_API}/systems/{system}/packages/{enc_pkg}/versions/{enc_ver}:dependencies"

    data = _get_json(url)
    time.sleep(REQUEST_DELAY)

    if not data:
        _deps_cache[cache_key] = []
        return []

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    result = []
    for edge in edges:
        to_node = edge.get("toNode", 0)
        if to_node < len(nodes):
            dep_vk = nodes[to_node].get("versionKey", {})
            dep_name = dep_vk.get("name", "")
            req = edge.get("requirement", "")
            # Dependency type: check relation field
            dep_type = edge.get("relation", "")
            if "DEV" in dep_type.upper() if dep_type else False:
                dep_type = "devDependencies"
            else:
                dep_type = "dependencies"
            result.append({
                "dep_package": dep_name,
                "requirement": req,
                "dep_type": dep_type,
            })

    _deps_cache[cache_key] = result
    return result


# ── Version comparison and constraint checking ───────────────────────────

def parse_date(s: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp."""
    if not s:
        return ""
    return s[:10]


def _parse_semver_tuple(v: str) -> tuple:
    """Parse a version string into a tuple for comparison."""
    v = v.lstrip("vV")
    parts = re.split(r"[\.\-]", v)
    result = []
    for p in parts:
        m = re.match(r"^(\d+)(.*)", p)
        if m:
            result.append((0, int(m.group(1)), m.group(2)))
        else:
            result.append((1, 0, p))
    return tuple(result)


def _version_lt(a: str, b: str) -> bool:
    return _parse_semver_tuple(a) < _parse_semver_tuple(b)


def _version_ge(a: str, b: str) -> bool:
    return _parse_semver_tuple(a) >= _parse_semver_tuple(b)


def _major_version(v: str) -> str:
    v = v.lstrip("vV")
    m = re.match(r"^(\d+)", v)
    return m.group(1) if m else "0"


def requires_major_upgrade(patched: str, constraint_version: str) -> int:
    if not constraint_version:
        return 0
    return 1 if _major_version(patched) != _major_version(constraint_version) else 0


# ── PyPI constraint checking ─────────────────────────────────────────────

def _check_pypi_specifier(spec_str: str, version: str) -> bool:
    """Check if a version satisfies a PyPI specifier set."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        spec = SpecifierSet(spec_str)
        return Version(version) in spec
    except Exception:
        pass

    # Fallback heuristic
    spec_str = spec_str.strip()
    if not spec_str:
        return True

    specs = [s.strip() for s in spec_str.split(",")]
    for s in specs:
        s = s.strip()
        if s.startswith("=="):
            if version != s[2:].strip():
                return False
        elif s.startswith("!="):
            if version == s[2:].strip():
                return False
        elif s.startswith(">="):
            if _version_lt(version, s[2:].strip()):
                return False
        elif s.startswith("<="):
            target = s[2:].strip()
            if not (_version_lt(version, target) or version == target):
                return False
        elif s.startswith(">") and not s.startswith(">="):
            target = s[1:].strip()
            if not (_version_ge(version, target) and version != target):
                return False
        elif s.startswith("<") and not s.startswith("<="):
            target = s[1:].strip()
            if _version_ge(version, target):
                return False
        elif s.startswith("~="):
            base = s[2:].strip()
            if _version_lt(version, base):
                return False
            base_parts = base.split(".")
            ver_parts = version.split(".")
            if len(base_parts) >= 2 and len(ver_parts) >= 2:
                if base_parts[0] != ver_parts[0]:
                    return False
    return True


def classify_pypi_constraint(req_str: str) -> str:
    if not req_str or req_str.strip() == "":
        return "any"
    req = req_str.strip()
    if req.startswith("=="):
        return "pinned"
    if req.startswith("~="):
        return "compatible"
    if ">=" in req and "<" in req:
        return "range"
    if req.startswith(">=") and "<" not in req:
        return "lower_bound"
    if req.startswith("!="):
        return "exclusion"
    if req.startswith("<="):
        return "upper_bound"
    return "other"


def _extract_pinned_version_pypi(req_str: str) -> str:
    if req_str and req_str.strip().startswith("=="):
        return req_str.strip()[2:].strip()
    # For ranges, extract lower bound
    m = re.search(r">=\s*([\d][^\s,]*)", req_str or "")
    if m:
        return m.group(1)
    return ""


# ── Maven constraint checking ────────────────────────────────────────────

def _check_maven_range(range_str: str, version: str) -> bool:
    """Check if a version satisfies a Maven version range."""
    v = range_str.strip() if range_str else ""
    if not v:
        return True

    # Exact pin: [1.2.3]
    if re.match(r"^\[([^\],]+)\]$", v):
        pinned = re.match(r"^\[([^\],]+)\]$", v).group(1)
        return version == pinned

    # Range: [low,high) or (low,high] etc
    m = re.match(r"^([\[\(])([^,]*),([^)\]]*)([\]\)])$", v)
    if m:
        left_inc = m.group(1) == "["
        low = m.group(2).strip()
        high = m.group(3).strip()
        right_inc = m.group(4) == "]"

        if low:
            if left_inc and _version_lt(version, low):
                return False
            if not left_inc and not (_version_ge(version, low) and version != low):
                return False
        if high:
            if right_inc and not (_version_lt(version, high) or version == high):
                return False
            if not right_inc and not _version_lt(version, high):
                return False
        return True

    # Bare version = effectively pinned in Maven
    if re.match(r"^[\d]", v):
        return version == v

    return False


def classify_maven_constraint(version_str: str) -> str:
    if not version_str or version_str.strip() == "":
        return "any"
    v = version_str.strip()
    if re.match(r"^\[([^\],]+)\]$", v):
        return "pinned"
    if re.match(r"^[\[\(].*,.*[\]\)]$", v):
        return "range"
    if re.match(r"^[\d]", v):
        return "soft"
    return "other"


def _extract_pinned_version_maven(version_str: str) -> str:
    v = version_str.strip() if version_str else ""
    if re.match(r"^[\d]", v):
        return v
    m = re.match(r"^\[([^\],]+)\]$", v)
    if m:
        return m.group(1)
    m = re.match(r"^[\[\(]([^,]*),", v)
    if m and m.group(1):
        return m.group(1)
    return ""


# ── Core propagation logic ───────────────────────────────────────────────

def _version_in_ghsa_range(version: str, vulnerable_range: str) -> bool:
    """Check if a version falls within a GHSA vulnerable_range.

    GHSA ranges look like: ">= 9.0.0, < 9.0.98" or ">= 0, < 5.3.0rc1"
    """
    if not vulnerable_range or vulnerable_range.strip() == ">= 0":
        return True  # matches everything

    parts = [p.strip() for p in vulnerable_range.split(",")]
    for p in parts:
        p = p.strip()
        if p.startswith(">="):
            bound = p[2:].strip()
            if bound == "0":
                continue
            if _version_lt(version, bound):
                return False
        elif p.startswith(">") and not p.startswith(">="):
            bound = p[1:].strip()
            if not (_version_ge(version, bound) and version != bound):
                return False
        elif p.startswith("<") and not p.startswith("<="):
            bound = p[1:].strip()
            if _version_ge(version, bound):
                return False
        elif p.startswith("<="):
            bound = p[2:].strip()
            if not (_version_lt(version, bound) or version == bound):
                return False
    return True


def find_vulnerable_version(
    system: str, package: str, patched_version: str, vulnerable_range: str
) -> str | None:
    """Find the latest vulnerable version (the version just before the patch).

    Uses the vulnerable_range from GHSA to filter to the correct release line.
    For example, if patched=9.0.96 and range=">= 9.0.0.M1, < 9.0.96",
    we find the latest version that matches the range.
    """
    versions = get_package_versions(system, package)
    if not versions:
        return None

    # Strategy 1: Use the vulnerable_range to find matching versions
    # This correctly handles multi-branch packages (e.g., tomcat 9.x vs 10.x)
    if vulnerable_range and vulnerable_range.strip() not in (">= 0", ""):
        candidates = [
            v for v in versions
            if _version_in_ghsa_range(v["version"], vulnerable_range)
        ]
        if candidates:
            # Return the latest (by publishedAt, which is how we sorted)
            return candidates[-1]["version"]

    # Strategy 2: Find the version published just before the patched version
    # within the same major version line
    patched_major = _major_version(patched_version)
    patched_idx = None
    for i, v in enumerate(versions):
        if v["version"] == patched_version:
            patched_idx = i
            break

    if patched_idx is not None:
        # Look backwards for a version in the same major line
        for j in range(patched_idx - 1, -1, -1):
            if _major_version(versions[j]["version"]) == patched_major:
                return versions[j]["version"]
        # If no same-major predecessor, just use previous
        if patched_idx > 0:
            return versions[patched_idx - 1]["version"]

    # Strategy 3: Fallback -- use the latest version with the same major
    same_major = [v for v in versions if _major_version(v["version"]) == patched_major]
    if same_major:
        return same_major[-1]["version"]

    # Final fallback
    if versions:
        return versions[-1]["version"]

    return None


def process_advisory(
    system: str,
    row: dict,
    cutoff_date: str,
    verbose: bool = False,
) -> list[dict]:
    """Process a single advisory: find dependents and measure propagation."""
    package = row["package_name"]
    patched_version = str(row["first_patched_version"])
    vulnerable_range = str(row.get("vulnerable_range", ""))
    ghsa_id = row["ghsa_id"]
    cve_id = row.get("cve_id", "")
    severity_label = row.get("severity_label", "")
    cwe_id = row.get("cwe_primary", row.get("cwe_id", ""))

    if verbose:
        _log(f"    Processing {ghsa_id}: {package} (patched={patched_version})")

    # 1. Find vulnerable version
    vuln_version = find_vulnerable_version(system, package, patched_version, vulnerable_range)
    if not vuln_version:
        if verbose:
            _log(f"      No vulnerable version found, skipping")
        return []

    if verbose:
        _log(f"      Vulnerable version: {vuln_version}")

    # 2. Get patch release date
    patch_pub = ""
    versions = get_package_versions(system, package)
    for v in versions:
        if v["version"] == patched_version:
            patch_pub = parse_date(v["publishedAt"])
            break

    if not patch_pub:
        if verbose:
            _log(f"      No patch release date found, skipping")
        return []

    if verbose:
        _log(f"      Patch release date: {patch_pub}")

    # 3. Get direct dependents of the vulnerable version
    dependents = get_direct_dependents(system, package, vuln_version)
    if not dependents:
        if verbose:
            _log(f"      No dependents found for {package}@{vuln_version}")
        return []

    if verbose:
        _log(f"      Found {len(dependents)} direct dependents")

    results = []

    for dep_idx, dep in enumerate(dependents):
        dep_pkg = dep["package"]
        dep_ver = dep["version"]
        dep_system = dep.get("system", system)

        if not dep_pkg or not dep_ver:
            continue

        # 4a. Get the dependent's dependency edges to find constraint on upstream
        edges = get_dependency_edges(dep_system, dep_pkg, dep_ver)
        constraint = ""
        dep_type = "dependencies"
        for edge in edges:
            if edge["dep_package"] == package:
                constraint = edge["requirement"]
                dep_type = edge["dep_type"]
                break

        if not constraint:
            continue

        # 4b. Check if constraint auto-accepts the patched version
        if system == "pypi":
            auto_accepts = _check_pypi_specifier(constraint, patched_version)
            ctype = classify_pypi_constraint(constraint)
            pinned_ver = _extract_pinned_version_pypi(constraint)
        else:  # maven
            auto_accepts = _check_maven_range(constraint, patched_version)
            ctype = classify_maven_constraint(constraint)
            pinned_ver = _extract_pinned_version_maven(constraint)

        major_upgrade = requires_major_upgrade(patched_version, pinned_ver)

        # 4c. Get downstream package version list
        dep_versions = get_package_versions(dep_system, dep_pkg)

        # Get downstream latest release date
        downstream_release = ""
        if dep_versions:
            downstream_release = parse_date(dep_versions[-1]["publishedAt"])

        if auto_accepts:
            # Auto-propagation: constraint already accepts the patch
            prop_type = "auto"
            prop_days = 0
            first_safe = patch_pub

            results.append({
                "downstream_package": dep_pkg,
                "downstream_version": "",
                "downstream_release_date": downstream_release,
                "upstream_package": package,
                "cve_id": cve_id if cve_id and str(cve_id) != "nan" else "",
                "osv_id": ghsa_id,
                "cvss_severity": severity_label,
                "cwe_id": cwe_id if cwe_id and str(cwe_id) != "nan" else "",
                "patched_version": patched_version,
                "patch_release_date": patch_pub,
                "first_safe_release_date": first_safe,
                "propagation_days": prop_days,
                "propagation_type": prop_type,
                "dependency_type": dep_type,
                "version_constraint_at_t0": constraint,
                "constraint_type": ctype,
                "requires_major_upgrade": major_upgrade,
            })

        else:
            # Not auto-accepted: check subsequent downstream versions
            # Find subsequent versions after dep_ver that are after patch_pub
            found_dep_ver = False
            subsequent = []
            for dv in dep_versions:
                if dv["version"] == dep_ver:
                    found_dep_ver = True
                    continue
                if not found_dep_ver:
                    continue
                dv_date = parse_date(dv["publishedAt"])
                if dv_date >= patch_pub:
                    subsequent.append(dv)

            # Limit how many we check (each requires an API call)
            subsequent = subsequent[:MAX_SUBSEQUENT_VERSIONS_TO_CHECK]

            first_safe_date = ""
            first_safe_ver = ""

            for dv in subsequent:
                dv_date = parse_date(dv["publishedAt"])
                dv_edges = get_dependency_edges(dep_system, dep_pkg, dv["version"])

                new_constraint = ""
                for edge in dv_edges:
                    if edge["dep_package"] == package:
                        new_constraint = edge["requirement"]
                        break

                if not new_constraint:
                    # Dependency removed = safe
                    first_safe_date = dv_date
                    first_safe_ver = dv["version"]
                    break

                if system == "pypi":
                    accepts = _check_pypi_specifier(new_constraint, patched_version)
                else:
                    accepts = _check_maven_range(new_constraint, patched_version)

                if accepts:
                    first_safe_date = dv_date
                    first_safe_ver = dv["version"]
                    break

            if first_safe_date:
                prop_type = "manual"
                prop_days = (
                    datetime.strptime(first_safe_date, "%Y-%m-%d")
                    - datetime.strptime(patch_pub, "%Y-%m-%d")
                ).days
                first_safe = first_safe_date
            else:
                prop_type = "censored"
                prop_days = max(0, (
                    datetime.strptime(cutoff_date, "%Y-%m-%d")
                    - datetime.strptime(patch_pub, "%Y-%m-%d")
                ).days)
                first_safe = cutoff_date
                first_safe_ver = ""

            results.append({
                "downstream_package": dep_pkg,
                "downstream_version": first_safe_ver,
                "downstream_release_date": downstream_release,
                "upstream_package": package,
                "cve_id": cve_id if cve_id and str(cve_id) != "nan" else "",
                "osv_id": ghsa_id,
                "cvss_severity": severity_label,
                "cwe_id": cwe_id if cwe_id and str(cwe_id) != "nan" else "",
                "patched_version": patched_version,
                "patch_release_date": patch_pub,
                "first_safe_release_date": first_safe,
                "propagation_days": prop_days,
                "propagation_type": prop_type,
                "dependency_type": dep_type,
                "version_constraint_at_t0": constraint,
                "constraint_type": ctype,
                "requires_major_upgrade": major_upgrade,
            })

    if verbose and results:
        auto_n = sum(1 for r in results if r["propagation_type"] == "auto")
        _log(f"      -> {len(results)} events ({auto_n} auto)")

    return results


# ── Main collection loop ─────────────────────────────────────────────────

def load_advisories(ecosystem_label: str) -> list[dict]:
    """Load and deduplicate advisories for an ecosystem."""
    import pandas as pd

    df = pd.read_csv(BASE_CSV)
    subset = df[
        (df["ecosystem"] == ecosystem_label)
        & (df["has_patch"] == True)
        & df["first_patched_version"].notna()
        & (df["first_patched_version"].astype(str) != "")
    ].copy()

    # For each ghsa_id x package_name, keep only one row (the widest vulnerable range)
    subset = subset.sort_values("vulnerable_range", ascending=False)
    subset = subset.drop_duplicates(subset=["ghsa_id", "package_name"], keep="first")

    records = subset.to_dict("records")
    _log(f"  {ecosystem_label}: {len(records)} unique advisory-package pairs "
         f"({subset['package_name'].nunique()} unique packages)")
    return records


def select_test_advisories(advisories: list[dict], system: str, n: int = 5) -> list[dict]:
    """Select test advisories that are likely to have dependents.

    Prefer well-known packages (django, flask, requests, etc. for PyPI;
    jackson, log4j, etc. for Maven).
    """
    popular_pypi = {
        "django", "flask", "requests", "pillow", "paramiko", "jinja2",
        "pyyaml", "cryptography", "urllib3", "certifi", "setuptools",
        "numpy", "lxml", "aiohttp", "starlette", "fastapi", "celery",
        "tensorflow", "werkzeug", "tornado",
    }
    popular_maven = {
        "com.fasterxml.jackson.core:jackson-databind",
        "org.apache.logging.log4j:log4j-core",
        "com.google.guava:guava",
        "org.springframework:spring-core",
        "org.springframework:spring-web",
        "org.apache.tomcat.embed:tomcat-embed-core",
        "io.netty:netty-codec-http",
        "commons-io:commons-io",
        "org.apache.commons:commons-lang3",
        "com.google.code.gson:gson",
    }

    popular = popular_pypi if system == "pypi" else popular_maven

    # First try popular packages
    selected = [a for a in advisories if a["package_name"].lower() in popular
                or a["package_name"] in popular]

    # If not enough, add others
    if len(selected) < n:
        remaining = [a for a in advisories if a not in selected]
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def collect_ecosystem(
    ecosystem: str,
    test_mode: bool = False,
    max_advisories: int = 0,
):
    """Collect propagation data for one ecosystem."""
    global _api_calls

    eco_map = {"pypi": "PyPI", "maven": "Maven"}
    eco_label = eco_map[ecosystem]
    system = ecosystem

    output_file = DATA_DIR / f"{ecosystem}_propagation.csv"

    advisories = load_advisories(eco_label)

    if test_mode:
        advisories = select_test_advisories(advisories, system, n=5)
        _log(f"  TEST MODE: selected {len(advisories)} advisories:")
        for a in advisories:
            _log(f"    {a['ghsa_id']} | {a['package_name']} | patched={a['first_patched_version']}")
    elif max_advisories > 0:
        advisories = advisories[:max_advisories]

    # Check for existing progress
    existing_osv_ids = set()
    if not test_mode and output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_osv_ids.add(row.get("osv_id", ""))
        _log(f"  Found existing data with {len(existing_osv_ids)} unique advisories, resuming")

    # In test mode, start fresh
    if test_mode and output_file.exists():
        output_file.unlink()

    all_results = []
    processed = 0
    skipped = 0
    errors = 0
    _api_calls = 0
    start_time = time.time()

    for i, adv in enumerate(advisories):
        ghsa = adv["ghsa_id"]

        if ghsa in existing_osv_ids:
            skipped += 1
            continue

        _log(f"  >> {ghsa}: {adv['package_name']} (patched={adv['first_patched_version']})")

        try:
            events = process_advisory(
                system, adv, OBSERVATION_CUTOFF,
                verbose=test_mode,
            )
            if events:
                all_results.extend(events)
            processed += 1
        except Exception as e:
            _log(f"  ERROR processing {ghsa} ({adv['package_name']}): {e}")
            errors += 1
            processed += 1

        total_idx = i + 1
        if total_idx % 10 == 0 or total_idx == len(advisories):
            elapsed = time.time() - start_time
            auto_n = sum(1 for r in all_results if r["propagation_type"] == "auto")
            manual_n = sum(1 for r in all_results if r["propagation_type"] == "manual")
            censored_n = sum(1 for r in all_results if r["propagation_type"] == "censored")
            _log(
                f"  {eco_label} [{total_idx}/{len(advisories)}] "
                f"proc={processed} skip={skipped} err={errors} "
                f"events={len(all_results)} "
                f"(auto={auto_n}/manual={manual_n}/cens={censored_n}) "
                f"api_calls={_api_calls} elapsed={elapsed:.0f}s"
            )

        # Save progress every 50 processed advisories
        if not test_mode and len(all_results) > 0 and processed % 50 == 0:
            _save_results(output_file, all_results, append=output_file.exists())
            _log(f"  Checkpoint: saved {len(all_results)} events")
            all_results = []

    # Final save
    if all_results:
        _save_results(output_file, all_results, append=output_file.exists())
        _log(f"  Final save: {len(all_results)} events to {output_file.name}")

    _print_summary(output_file, eco_label)
    return output_file


def _save_results(output_file: Path, results: list[dict], append: bool = False):
    """Save results to CSV."""
    mode = "a" if append else "w"
    write_header = not append or not output_file.exists() or output_file.stat().st_size == 0

    with open(output_file, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)


def _print_summary(output_file: Path, eco_label: str):
    """Print summary statistics."""
    if not output_file.exists() or output_file.stat().st_size == 0:
        _log(f"\n  {eco_label}: No data collected")
        return

    import pandas as pd
    df = pd.read_csv(output_file)

    _log(f"\n{'=' * 60}")
    _log(f"  {eco_label} PROPAGATION DATA SUMMARY")
    _log(f"{'=' * 60}")
    _log(f"  Total events: {len(df)}")
    _log(f"  Unique upstream packages: {df['upstream_package'].nunique()}")
    _log(f"  Unique downstream packages: {df['downstream_package'].nunique()}")
    _log(f"  Unique advisories (osv_id): {df['osv_id'].nunique()}")
    _log(f"")
    _log(f"  Propagation type distribution:")
    for pt, count in df["propagation_type"].value_counts().items():
        pct = count / len(df) * 100
        _log(f"    {pt:>10s}: {count:>6d} ({pct:.1f}%)")
    _log(f"")
    _log(f"  Constraint type distribution:")
    for ct, count in df["constraint_type"].value_counts().items():
        pct = count / len(df) * 100
        _log(f"    {ct:>15s}: {count:>6d} ({pct:.1f}%)")
    _log(f"")

    non_auto = df[df["propagation_type"] != "auto"]["propagation_days"]
    if len(non_auto) > 0:
        _log(f"  Propagation days (manual + censored):")
        _log(f"    mean:   {non_auto.mean():.1f}")
        _log(f"    median: {non_auto.median():.1f}")
        _log(f"    min:    {non_auto.min():.0f}")
        _log(f"    max:    {non_auto.max():.0f}")
    _log(f"")
    _log(f"  Requires major upgrade: {df['requires_major_upgrade'].mean():.1%}")

    # Show sample rows
    _log(f"\n  Sample events:")
    for _, r in df.head(5).iterrows():
        _log(f"    {r['downstream_package']:30s} <- {r['upstream_package']:20s} "
             f"| {r['propagation_type']:>8s} | {r['propagation_days']:>5.0f}d "
             f"| {r['constraint_type']:>12s} | {r['version_constraint_at_t0']}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect patch propagation data from deps.dev for PyPI/Maven"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: process only 5 advisories per ecosystem"
    )
    parser.add_argument(
        "--ecosystem", choices=["pypi", "maven"],
        help="Process only one ecosystem (default: both)"
    )
    parser.add_argument(
        "--max-advisories", type=int, default=0,
        help="Maximum advisories to process per ecosystem (0=all)"
    )
    args = parser.parse_args()

    _log("deps.dev Propagation Data Collector")
    _log(f"  Date: {datetime.now().isoformat()}")
    _log(f"  Test mode: {args.test}")
    _log(f"  Observation cutoff: {OBSERVATION_CUTOFF}")

    ecosystems = [args.ecosystem] if args.ecosystem else ["pypi", "maven"]

    for eco in ecosystems:
        _log(f"\n{'#' * 60}")
        _log(f"  ECOSYSTEM: {eco.upper()}")
        _log(f"{'#' * 60}")

        _version_list_cache.clear()
        _deps_cache.clear()

        collect_ecosystem(eco, test_mode=args.test, max_advisories=args.max_advisories)

    _log("\nDone.")


if __name__ == "__main__":
    main()
