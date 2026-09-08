# patch-stall — Replication Package

Data and code for **"Where Patches Stall: Ecosystem-Level Constraint
Policies, Not Upstream Health, Covary with Supply Chain Propagation"**
(RAID 2026).

The paper asks what determines whether a fix reaches downstream dependents
once a vulnerability is patched upstream, using 26,288 advisory-package
pairs, 3,000 GitHub repositories, and 126,266 propagation events across
npm, PyPI, and Maven. Everything needed to reproduce the reported numbers
and figures is in this repository; no API access or credentials are
required.

**Observation cutoff**: 2026-03-01 (right-censoring date)
**Data freeze**: 2026-03-07 (GHSA/NVD snapshot)

## Quick Start

```bash
pip install -r requirements.txt

# Main analysis: RQ1-RQ3, all reported estimates and figures
python3 scripts/analyze_cross_ecosystem.py

# Supplementary diagnostics: VIF, joint LRT, censoring rates, forest plot
python3 scripts/supplementary_diagnostics.py

# Sensitivity analyses (each is independent of the others)
python3 scripts/sensitivity_model_robustness.py  # multicollinearity, dependent cap, T_patch<0, interaction
python3 scripts/sensitivity_advisory_linkage.py  # GHSA repository-linkage confound
python3 scripts/sensitivity_temporal.py          # year-stratified Cox, unstratified propagation
python3 scripts/sensitivity_tost_margin.py       # TOST across equivalence margins
python3 scripts/sensitivity_pypi_selection.py    # within-PyPI constraint selection bias
python3 scripts/sensitivity_maven_bot_proxy.py   # Maven bot-contamination bound and re-fit
```

Run `analyze_cross_ecosystem.py` first: the other scripts import its
model-building helpers, and `supplementary_diagnostics.py` regenerates one
of its figures.

## Reproducibility

The analysis is deterministic. Two quantities are measured relative to a
reference date—the right-censoring time of advisories that never received a
patch, and repository age—and that date is fixed to the observation cutoff,
2026-03-01. It can be overridden for sensitivity checks:

```bash
PATCH_STALL_ANALYSIS_DATE=2026-04-15 python3 scripts/analyze_cross_ecosystem.py
```

**Difference from the published paper.** In the code that produced the
manuscript, both quantities were measured against the wall clock at run
time rather than against the observation cutoff, so the estimates that
depend on them drifted with the date the analysis happened to be run, and
that date was never recorded. Everything downstream of the propagation data
is unaffected, but 33 of the 595 values in `results_main.json` move, all of
them in the RQ1 patch-speed analysis:

| Quantity | Paper | This artifact |
|---|---|---|
| Log-rank npm vs. PyPI | χ² = 662.4 | χ² = 661.6 |
| Log-rank PyPI vs. Maven | χ² = 677.3 | χ² = 680.1 |
| Log-rank npm vs. Maven | p = 0.641 | p = 0.644 |
| Cox HIGH severity (Table 3) | HR 1.050 [1.011, 1.090], p = 0.011 | HR 1.053 [1.014, 1.093], p = 0.008 |
| Other Table 3 rows | — | third-decimal shifts in HR or CI |
| Schoenfeld diagnostics | — | shifts, all conclusions unchanged |

Every sample size and event count reproduces exactly, as does every RQ2 and
RQ3 estimate, including the TOST equivalence results the central claim
rests on. No hazard ratio crosses 1, no p-value crosses 0.05 or 0.001, and
the only threshold any statistic crosses is HIGH severity becoming more
significant rather than less.

The analysis itself is otherwise fully deterministic: fixing the reference
date, it produces bit-identical output on Python 3.10 with lifelines 0.30.0,
NumPy 1.26 and SciPy 1.15, and on Python 3.13 with lifelines 0.30.3,
NumPy 2.4 and SciPy 1.17.

## Directory Structure

```
patch-stall/
├── README.md                    ← This file
├── LICENSE                      ← MIT
├── requirements.txt             ← Python dependencies (3.10+)
├── data/
│   ├── cross_ecosystem_base.csv     26,288 advisory-package pairs
│   ├── cross_ecosystem_repos.csv     3,000 repositories
│   ├── github_repo_health.csv        3,000 repo health metrics
│   ├── patch_dates_merged.csv       19,883 dual-source patch dates
│   ├── patch_release_dates.csv      31,687 registry patch dates
│   ├── git_tag_dates.csv            15,929 git tag dates
│   ├── osv_severity_index.csv      240,026 OSV severity records
│   ├── npm_propagation.csv          18,102 npm propagation events
│   ├── pypi_propagation.csv         32,822 PyPI propagation events
│   ├── maven_propagation.csv        75,342 Maven propagation events
│   ├── results_main.json            ← output of analyze_cross_ecosystem.py
│   ├── results_supplementary.json   ← output of supplementary_diagnostics.py
│   ├── sensitivity_*.json           ← outputs of the sensitivity scripts
│   └── release/                 ← Normalized tables (for reuse)
│       ├── advisories.csv           Advisory master (26,288 rows)
│       ├── repositories.csv         Repository health (3,000 rows)
│       ├── patch_timing.csv         Unified patch dates (19,883 rows)
│       ├── propagation_events.csv   Merged propagation (126,266 rows)
│       ├── manifest.json            SHA-256 hashes and row counts
│       └── DATA_DICTIONARY.md       Full schema and join instructions
├── scripts/
│   ├── analyze_cross_ecosystem.py       Main analysis (all RQs + figures)
│   ├── supplementary_diagnostics.py     VIF, joint tests, forest plot
│   ├── sensitivity_*.py                 Sensitivity analyses (see table below)
│   ├── normalize_for_release.py         Builds data/release/ from the raw CSVs
│   └── collect/                         Data collection (requires API access)
│       ├── build_cross_eco_base.py
│       ├── collect_git_tags.py
│       ├── collect_github_metadata.py
│       ├── collect_patch_dates_fast.py
│       ├── collect_propagation_depsdev.py
│       ├── classify_bot_updates.py      Dependabot/Renovate classification
│       ├── collect_monthly_health.py    Time-varying health metrics
│       └── validate_dual_source.py
└── paper/
    └── fig/                     ← Generated figures (PDF)
```

## Data

### Raw Data (`data/`)

The analysis scripts read these files directly. All data are frozen as of
March 7, 2026.

| File | Rows | Description |
|---|---|---|
| `cross_ecosystem_base.csv` | 26,288 | One row per GHSA advisory × affected package |
| `cross_ecosystem_repos.csv` | 3,000 | Unique upstream repositories |
| `github_repo_health.csv` | 3,000 | Stars, forks, contributors, open issues, age, archived |
| `patch_dates_merged.csv` | 19,883 | Unified patch dates (registry preferred, git tag fallback) |
| `patch_release_dates.csv` | 31,687 | Registry API patch dates |
| `git_tag_dates.csv` | 15,929 | Git tag dates from the GitHub API |
| `osv_severity_index.csv` | 240,026 | CVSS/CWE records from NVD and OSV |
| `npm_propagation.csv` | 18,102 | npm downstream propagation events |
| `pypi_propagation.csv` | 32,822 | PyPI downstream propagation events |
| `maven_propagation.csv` | 75,342 | Maven downstream propagation events |

### Normalized Tables (`data/release/`)

The same data deduplicated and linked by foreign keys, intended for reuse
in other studies rather than for reproducing this paper. Built by
`scripts/normalize_for_release.py`. See `DATA_DICTIONARY.md` for the full
schema, join instructions, and the constraint type mapping.

## Scripts

### Analysis (reproduces all paper results)

| Script | Output | Description |
|---|---|---|
| `analyze_cross_ecosystem.py` | `data/results_main.json` + figures | RQ1-RQ3: Kaplan-Meier, ecosystem-stratified Cox PH, negative binomial, TOST |
| `supplementary_diagnostics.py` | `data/results_supplementary.json` | VIF and correlations, joint LRT for health covariates, censoring rates, forest plot |
| `sensitivity_model_robustness.py` | `data/sensitivity_model_robustness.json` | Multicollinearity, deps.dev 100-dependent cap, pre-advisory patches, permissive-constraint interaction |
| `sensitivity_advisory_linkage.py` | `data/sensitivity_advisory_linkage.json` | Patch rates and Cox model restricted to repository-linked advisories |
| `sensitivity_temporal.py` | `data/sensitivity_temporal.json` | Year-stratified Cox model; unstratified propagation model |
| `sensitivity_tost_margin.py` | `data/sensitivity_tost_margin.json` | TOST at δ_HR ∈ {0.10, 0.15, 0.20, 0.25, 0.30} |
| `sensitivity_pypi_selection.py` | `data/sensitivity_pypi_selection.json` | Within-PyPI covariate balance (SMD) between permissive and pinned constraints |
| `sensitivity_maven_bot_proxy.py` | `data/sensitivity_maven_bot_proxy.json` | One-day bot proxy for Maven, worst-case bound, and per-ecosystem TOST re-fit excluding ≤1-day manual events |

### Data Collection (optional, requires API access)

Scripts in `scripts/collect/` were used to gather the raw data from GitHub,
npm, PyPI, Maven Central, and deps.dev. They are included for transparency
but are **not required** to reproduce the analysis—all collected data is
provided.

Two of them feed the sensitivity analyses rather than the main dataset:

- `classify_bot_updates.py` classifies a random sample of "manual"
  propagation events as Dependabot/Renovate-generated or human by
  inspecting GitHub commit metadata. Output:
  `data/sensitivity_bot_classification.json` (requires a GitHub token).

  ```bash
  GITHUB_TOKEN=ghp_xxx python3 scripts/collect/classify_bot_updates.py \
      --sample-size 500 --token $GITHUB_TOKEN
  ```

- `collect_monthly_health.py` collects monthly contributor counts for
  time-varying Cox covariates (requires a GitHub token; future work).

## Data Sources

- **GitHub Security Advisories (GHSA)** via the GitHub API
- **NVD** (CVE/CVSS/CWE) via the NVD API
- **OSV** severity records
- **Package registries**: npm, PyPI (JSON API), Maven Central
- **deps.dev** (Google Open Source Insights) for propagation data
- **GitHub REST API** for repository health and git tag dates

## Integrity Verification

```bash
cd data/release
python3 -c "
import json, hashlib, pathlib
m = json.load(open('manifest.json'))
for name, info in m['tables'].items():
    h = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    ok = 'OK' if h == info['sha256'] else 'MISMATCH'
    print(f'{ok}  {name}  ({info[\"rows\"]:,} rows)')
"
```

## Citation

```bibtex
@inproceedings{kurogome2026patchstall,
  author    = {Kurogome, Yuma},
  title     = {Where Patches Stall: Ecosystem-Level Constraint Policies,
               Not Upstream Health, Covary with Supply Chain Propagation},
  booktitle = {Research in Attacks, Intrusions and Defenses (RAID)},
  year      = {2026},
}
```

## Acknowledgments

This paper is based on results obtained from a project, JPNP24003,
commissioned by the New Energy and Industrial Technology Development
Organization (NEDO).

## License

MIT. See `LICENSE`.
