# Data Dictionary

Dataset for "Where Patches Stall: Ecosystem-Level Constraint Policies, Not
Upstream Health, Covary with Supply Chain Propagation" (RAID 2026).

**Observation cutoff**: 2026-03-01 (right-censoring date)
**Data freeze**: 2026-03-07 (GHSA/NVD snapshot)
**Ecosystems**: npm, PyPI, Maven

---

## Table: advisories.csv

Advisory master table. One row per GHSA advisory-package pair.

| Column | Type | Description |
|---|---|---|
| `osv_id` | string (PK) | GHSA identifier (e.g., `GHSA-xxxx-xxxx-xxxx`) |
| `cve_id` | string | CVE identifier (e.g., `CVE-2024-1234`); may be empty |
| `ecosystem` | enum | `npm`, `PyPI`, or `Maven` |
| `package_name` | string | Package name (Maven uses `groupId:artifactId`) |
| `severity_label` | enum | `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, or empty |
| `cvss_score` | float | CVSS v3.1 base score (0.0-10.0); empty if unavailable |
| `cwe_primary` | string | Primary CWE ID (e.g., `CWE-79`); empty if unavailable |
| `published_at` | datetime | GHSA publication timestamp (ISO 8601 UTC) |
| `first_patched_version` | string | First version with fix; empty if unpatched |
| `has_patch` | boolean | `True` if `first_patched_version` is non-empty |
| `github_repo` | string | GitHub `owner/repo`; empty if not linked |
| `has_repo` | boolean | `True` if `github_repo` is non-empty |

**Composite key**: (`osv_id`, `ecosystem`, `package_name`, `first_patched_version`)
**Note**: Multiple rows per (`osv_id`, `ecosystem`, `package_name`) when an
advisory lists multiple patched version ranges. Deduplicate before joining
with `propagation_events` (see below).
**Row count**: 26,288

---

## Table: repositories.csv

Repository health metrics. One row per unique GitHub repository.

| Column | Type | Description |
|---|---|---|
| `repo` | string (PK) | GitHub `owner/repo` |
| `ecosystem_list` | string | Comma-separated ecosystems (e.g., `PyPI`, `npm,Maven`) |
| `cve_count` | int | Total CVE count linked to this repo |
| `stars` | float | GitHub star count at data freeze |
| `forks` | float | GitHub fork count |
| `open_issues` | float | Open issue count |
| `contributors_count` | float | Total contributor count |
| `language` | string | Primary language |
| `created_at` | datetime | Repository creation date (ISO 8601) |
| `pushed_at` | datetime | Last push date (ISO 8601) |
| `archived` | boolean | `True` if archived |

**Primary key**: `repo`
**Join**: `advisories.github_repo` -> `repositories.repo`
**Row count**: 3,000

---

## Table: patch_timing.csv

Unified patch release dates from dual-source method (registry + git tag).

| Column | Type | Description |
|---|---|---|
| `osv_id` | string (FK) | GHSA identifier |
| `ecosystem` | enum | `npm`, `PyPI`, or `Maven` |
| `package_name` | string | Package name |
| `first_patched_version` | string | Patched version string |
| `github_repo` | string | GitHub `owner/repo`; empty if not linked |
| `patch_date` | datetime | Unified patch release date (ISO 8601 UTC) |
| `date_source` | enum | `registry` (package registry API) or `git` (tag analysis) |
| `tag_match_type` | string | Git tag matching strategy used (empty if registry source) |

**Foreign key**: `osv_id` -> `advisories.osv_id`
**Row count**: 19,883

### Date source priority
Registry dates are preferred. Git tag dates are used as fallback when
registry metadata is unavailable (critical for Maven, where only 14 of
1,559 observations have registry dates). Bland-Altman validation on
5,945 overlapping observations confirms median difference = 0.00 days
(Pearson r = 0.986).

---

## Table: propagation_events.csv

Downstream patch propagation events across all three ecosystems.

| Column | Type | Description |
|---|---|---|
| `ecosystem` | enum | `npm`, `PyPI`, or `Maven` (same labels as `advisories`) |
| `osv_id` | string (FK) | GHSA identifier of the upstream advisory |
| `cve_id` | string | CVE identifier (convenience; also in advisories) |
| `upstream_package` | string | Upstream package name (convenience; also in advisories) |
| `downstream_package` | string | Downstream dependent package name |
| `downstream_version` | string | First downstream version accepting the patch; empty if censored |
| `downstream_release_date` | date | Release date of first safe downstream version; = cutoff if censored |
| `first_safe_release_date` | date | Same as `downstream_release_date` |
| `propagation_days` | int | Days from upstream patch release to downstream adoption |
| `propagation_type` | enum | `auto` (constraint permits patch), `manual` (explicit update), `censored` (not yet updated) |
| `dependency_type` | string | Dependency scope (e.g., `dependencies`, `devDependencies`) |
| `version_constraint_at_t0` | string | Raw version constraint at time of upstream patch (e.g., `^1.2.3`, `==1.2.0`) |
| `constraint_type` | string | Ecosystem-specific constraint classification |
| `requires_major_upgrade` | int | `1` if patch crosses a major version boundary, `0` otherwise |

**Foreign key**: `osv_id` -> `advisories.osv_id`
**Composite key**: (`ecosystem`, `osv_id`, `downstream_package`)

### Constraint type mapping

| Abstract | npm | PyPI | Maven |
|---|---|---|---|
| permissive | `caret` | `lower_bound` | `hard` (open range) |
| restrictive | `tilde`, `compatible` | `compatible`, `range` | `hard` (bounded) |
| pinned | `exact`, `pinned` | `pinned` | `soft` (bare version) |
| wildcard | `star` | — | — |

### Propagation type definitions

- **auto**: The downstream package's existing version constraint already
  permits the patched version without any change. Propagation time = 0.
- **manual**: The downstream package explicitly updated its constraint
  to accept the patched version. Propagation time = days between
  upstream patch release and downstream release with updated constraint.
- **censored**: No downstream update observed by the observation cutoff
  (2026-03-01). Used as right-censored observation in survival analysis.

### Data collection method

- **npm**: Full census of all 819 advisories with patches. Propagation
  data from npm registry API.
- **PyPI**: Full census of all advisories with patches via deps.dev API.
  Up to 100 downstream dependents per advisory (API limit).
- **Maven**: Full census of all advisories with patches via deps.dev API.
  Up to 100 downstream dependents per advisory (API limit).

---

## Joining tables

### Reconstruct denormalized propagation view

**Important**: The `advisories` table has multiple rows per
(`osv_id`, `ecosystem`, `package_name`) when an advisory lists multiple
patched version ranges. Deduplicate before joining with `propagation_events`
to avoid row multiplication.

```sql
-- Deduplicate advisories before joining
WITH adv_dedup AS (
    SELECT DISTINCT osv_id, ecosystem, package_name,
           severity_label, cvss_score, cwe_primary, github_repo
    FROM advisories
)
SELECT
    p.*,
    a.severity_label,
    a.cvss_score,
    a.cwe_primary,
    r.stars,
    r.contributors_count
FROM propagation_events p
LEFT JOIN adv_dedup a
    ON p.osv_id = a.osv_id
    AND p.ecosystem = a.ecosystem
    AND p.upstream_package = a.package_name
LEFT JOIN repositories r ON a.github_repo = r.repo
```

### Python (pandas)
```python
import pandas as pd

prop = pd.read_csv("propagation_events.csv")
adv = pd.read_csv("advisories.csv")
repos = pd.read_csv("repositories.csv")

# Deduplicate advisories (drop version-range duplicates)
adv_dedup = adv.drop_duplicates(
    subset=["osv_id", "ecosystem", "package_name"],
    keep="first"
)

# Join advisory attributes
merged = prop.merge(
    adv_dedup[["osv_id", "ecosystem", "package_name",
               "severity_label", "cvss_score", "cwe_primary", "github_repo"]],
    left_on=["osv_id", "ecosystem", "upstream_package"],
    right_on=["osv_id", "ecosystem", "package_name"],
    how="left",
)

# Join repository health
merged = merged.merge(repos, left_on="github_repo", right_on="repo", how="left")
```

---

## Provenance

Row counts and SHA-256 hashes of both these tables and the raw CSVs they
are built from are recorded in `manifest.json`, generated by
`scripts/normalize_for_release.py`. Re-run that script to regenerate the
tables and the manifest after any data update.

## License

MIT. See `LICENSE` at the repository root.
