# CLAUDE.md - A/B Test ML Project

## Project Overview

Binary classification / causal inference project analyzing A/B test click-through data. The goal is to determine whether an experimental treatment significantly increases click-through rate and to identify which user segments respond most to the treatment.

## Environment

- **Python**: 3.12.2
- **Virtual environment**: `.venv/` (activate with `source .venv/bin/activate`)
- **Jupyter kernel**: `ml-project` (registered for VS Code notebooks)
- **Dependencies**: pinned in `requirements.txt` (173 packages, includes statsmodels + patsy)
- **Key packages**: pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, shap, mlflow, joblib, matplotlib, seaborn, statsmodels, ipykernel, jupyter
- **System dependency**: `libomp` (installed via Homebrew, required by XGBoost on macOS)

## Dataset

### Raw Data

- **File**: `data/raw/ab_test_dataset.csv`
- **Shape**: 200,020 rows × 6 columns

### Cleaned Data

- **File**: `data/cleaned.csv`
- **Shape**: 198,019 rows × 6 columns
- **Produced by**: `src/features/cleaning.py`
- **Cleaning applied**: duplicate removal (31 rows), group label normalization, device_type lowercasing, referral_source typo/whitespace fixes, missing group rows dropped (1,970), missing referral_source imputed as "unknown", session_time capped at 99th percentile (~48.2 min)

### Columns (cleaned)

| Column | Dtype | Description |
|---|---|---|
| `click` | int64 | **Target variable**. Binary 0/1 indicating whether user clicked |
| `group` | object | A/B test group: `exp` (100,017) or `con` (98,002) |
| `session_time` | float64 | Session duration in minutes. Capped at ~48.2 min (99th percentile) |
| `click_time` | object | Impression timestamp (2024-01-01 to 2024-01-14). 1,971 missing values evenly distributed across click=0/1 — this is when the impression was served, not when a click happened |
| `device_type` | object | `mobile` / `desktop` |
| `referral_source` | object | `search` / `email` / `social` / `direct` / `ads` / `unknown` |

### Class Balance (cleaned)

- `click=0`: 128,947 (65.1%)
- `click=1`: 69,072 (34.9%)

### Treatment Effect

- `exp` group click rate: 49.3%
- `con` group click rate: 20.1%

## Completed Work

### Phase 1: EDA (completed)

- **Script**: `src/eda.py`
- **Outputs**: `reports/eda/` — `distributions.png`, `correlations.png`, `class_balance.png`, `missing_values.png`, `eda_summary.md`
- Explored raw data distributions, missing values, class balance, and identified all data quality issues

### Phase 2: Data Cleaning (completed)

- **Script**: `src/features/cleaning.py`
- **Output**: `data/cleaned.csv`
- Resolved all 6 known data quality issues documented above

### Phase 3: A/B Test Statistical Analysis (completed)

- **Script**: `src/ab_test_analysis.py`
- **Report**: `reports/ab_test_results.md`
- **Plots**: `reports/ab_test/experiment_health.png`, `reports/ab_test/click/*.png` (5 plots), `reports/ab_test/session_time/*.png` (4 plots)

#### Key Findings

| Metric | Result |
|---|---|
| **Click rate (primary)** | Exp 49.3% vs Con 20.1%, absolute lift 29.2pp, relative lift 145%, p ≈ 0, **significant** after Bonferroni |
| **Adjusted OR** | 3.87 (95% CI: 3.79–3.94) — odds of clicking ~4x higher in Exp |
| **Cohen's h** | 0.63 (large effect) |
| **Session time (secondary)** | Mean diff = −0.039 min, p = 0.78, **not significant** |
| **Novelty effect** | None detected (p = 0.22), effect stable over 14 days |
| **SRM** | Flagged (p < 0.01) — slight imbalance (100k vs 98k), covariates well-balanced |
| **Subgroups** | Treatment effect uniform across all device types and referral sources (no significant interactions) |
| **Recommendation** | Ship the treatment (HIGH confidence) |

#### Analysis Sections

1. Experiment health check — SRM test, covariate balance (chi-squared + Cramér's V), temporal balance
2. Time-series stability — daily/hourly click rates, cumulative effect, novelty assessment (first 2 vs last 5 days)
3. Click analysis — proportion z-test, covariate-adjusted logistic regression, subgroup analysis with interaction tests, Cohen's h
4. Session time analysis — Mann-Whitney U, Welch's t-test (log), bootstrap CIs, OLS with covariates, clickers-only analysis with collider bias caveat
5. Multiple testing correction — Bonferroni (α = 0.025 for 2 primary hypotheses)
6. Report generation — stakeholder-facing markdown report with inline plot references

## Project Status: Analysis Complete

The A/B test analysis is **done**. The recommendation is to **ship the treatment** (HIGH confidence).

### Possible Future Work (if needed)

If ML modeling for click prediction is desired, the next steps would be:

1. **Data splitting** — stratified 70/15/15, parse `click_time` into temporal features
2. **Feature engineering** — `src/features/feature_engineering.py`, sklearn Pipeline
3. **Baseline model** — logistic regression, tracked with MLflow
4. **Advanced models** — XGBoost, LightGBM with hyperparameter tuning
5. **SHAP interpretation** — feature importance, calibration curves
6. **Final evaluation** — held-out test set, save to `models/`

## Methodology Rules

### Data Cleaning

- Document every cleaning decision and its rationale
- Handle duplicates, missing values, label inconsistencies, and outliers before analysis
- Cap outliers at a defensible percentile (e.g., 99th) rather than removing rows

### A/B Test Statistical Analysis

- Validate experiment health first: SRM test, covariate balance, temporal balance
- Use both unadjusted tests (z-test, chi-squared) and covariate-adjusted models (logistic regression, OLS)
- Apply Bonferroni correction when testing multiple primary hypotheses
- Flag subgroup analyses as exploratory (no multiple testing correction)
- Use non-parametric tests (Mann-Whitney U) for skewed continuous outcomes
- Bootstrap confidence intervals for difference in means/medians
- Check for novelty effects by comparing early vs late treatment lifts
- Flag collider bias when conditioning on post-treatment outcomes

### Reproducibility

- Set random seeds everywhere: `random_state=42`
- Pin all package versions in `requirements.txt`

### Code Standards

- Use the `.venv` environment for all work
- Do not commit large files (models, data) to git — use `.gitignore`

## Project Structure

```
claude-abtest/
├── data/
│   ├── raw/                        ← Original unmodified source files
│   │   └── ab_test_dataset.csv     ← 200,020 rows raw data
│   └── cleaned.csv                 ← 198,019 rows cleaned data
├── src/
│   ├── features/
│   │   └── cleaning.py             ← Data cleaning pipeline
│   ├── eda.py                      ← Exploratory data analysis
│   ├── ab_test_analysis.py         ← Statistical A/B test analysis
│   ├── models/                     ← Training, tuning, and evaluation (TODO)
│   └── monitor/                    ← Drift detection and performance monitoring (TODO)
├── reports/
│   ├── eda/                        ← EDA plots and summaries
│   ├── ab_test/                    ← A/B test analysis plots
│   │   ├── experiment_health.png
│   │   ├── click/                  ← Click metric plots (5 files)
│   │   └── session_time/           ← Session time plots (4 files)
│   └── ab_test_results.md          ← A/B test findings report
├── models/                         ← Saved pipeline artifacts (TODO)
├── tests/                          ← Unit tests (TODO)
├── notebooks/                      ← Ad hoc exploration only
├── .venv/                          ← Python virtual environment
├── requirements.txt                ← Pinned dependencies
└── CLAUDE.md                       ← This file
```
