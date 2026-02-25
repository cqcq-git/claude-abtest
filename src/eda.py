"""Exploratory Data Analysis for A/B test click-through dataset."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_PATH = Path("data/raw/ab_test_dataset.csv")
OUTPUT_DIR = Path("reports/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "click"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def print_overview(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Column dtypes:")
    for col in df.columns:
        print(f"  {col:20s} {str(df[col].dtype):10s} (nunique={df[col].nunique()})")

    print(f"\nMissing values:")
    missing = df.isnull().sum()
    for col in df.columns:
        count = missing[col]
        pct = count / len(df) * 100
        marker = " ***" if count > 0 else ""
        print(f"  {col:20s} {count:>6,} ({pct:.2f}%){marker}")
    print(f"  {'TOTAL':20s} {missing.sum():>6,}")


def print_class_balance(df):
    print("\n" + "=" * 60)
    print("TARGET CLASS BALANCE")
    print("=" * 60)
    counts = df[TARGET].value_counts().sort_index()
    props = df[TARGET].value_counts(normalize=True).sort_index()
    for val in counts.index:
        print(f"  {TARGET}={val}: {counts[val]:>8,} ({props[val]:.1%})")

    majority = counts.max()
    minority = counts.min()
    ratio = majority / minority
    print(f"\n  Imbalance ratio (majority/minority): {ratio:.2f}:1")
    return counts


def plot_class_balance(counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel(TARGET, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Target Class Distribution", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 (no click)", "1 (click)"])
    plt.tight_layout()
    path = OUTPUT_DIR / "class_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_distributions(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]

    if not num_cols:
        print("  No numerical features to plot.")
        return

    n = len(num_cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(num_cols):
        # Raw distribution
        ax = axes[i, 0]
        ax.hist(df[col].dropna(), bins=50, color="#3498db", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{col} — raw", fontsize=11)
        ax.set_ylabel("Count")
        median = df[col].median()
        ax.axvline(median, color="red", linestyle="--", linewidth=1, label=f"median={median:.2f}")
        ax.legend(fontsize=9)

        # Log-transformed distribution (if all positive)
        ax = axes[i, 1]
        valid = df[col].dropna()
        if (valid > 0).all():
            log_vals = np.log1p(valid)
            ax.hist(log_vals, bins=50, color="#9b59b6", edgecolor="black", linewidth=0.3)
            ax.set_title(f"{col} — log1p transformed", fontsize=11)
        else:
            ax.hist(valid, bins=50, color="#9b59b6", edgecolor="black", linewidth=0.3)
            ax.set_title(f"{col} — raw (has non-positive values)", fontsize=11)
        ax.set_ylabel("Count")

    plt.tight_layout()
    path = OUTPUT_DIR / "distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_correlations(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Encode categoricals for correlation analysis
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        cleaned = df_encoded[col].astype(str).str.strip().str.lower()
        df_encoded[col] = cleaned.astype("category").cat.codes

    all_cols = num_cols + cat_cols
    corr = df_encoded[all_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlations (categoricals label-encoded)", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "correlations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_missing_values(df):
    missing = df.isnull()
    if not missing.any().any():
        print("  No missing values — skipping heatmap.")
        return

    # Only show columns that have missing values
    cols_with_missing = [c for c in df.columns if missing[c].any()]
    subset = missing[cols_with_missing]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sample rows if dataset is too large for a readable heatmap
    if len(subset) > 2000:
        sample_idx = np.linspace(0, len(subset) - 1, 2000, dtype=int)
        plot_data = subset.iloc[sample_idx]
    else:
        plot_data = subset

    sns.heatmap(plot_data.T, cbar=False, yticklabels=True, cmap=["#ecf0f1", "#e74c3c"], ax=ax)
    ax.set_title("Missing Value Patterns (red = missing)", fontsize=13)
    ax.set_xlabel("Row index (sampled)" if len(subset) > 2000 else "Row index")
    ax.set_ylabel("")
    plt.tight_layout()
    path = OUTPUT_DIR / "missing_values.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def detect_quality_issues(df):
    """Dynamically detect data quality issues across all columns."""
    issues = []

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count:,} duplicate rows found.")

    for col in df.columns:
        series = df[col]

        # --- Categorical columns ---
        if series.dtype == "object":
            unique_raw = series.dropna().unique()

            # Skip high-cardinality columns (likely IDs or timestamps)
            if len(unique_raw) > 100:
                continue

            normalized = pd.Series(unique_raw).str.strip().str.lower()

            # Detect whitespace issues: values that change after stripping
            whitespace_variants = [v for v in unique_raw if v != v.strip()]
            if whitespace_variants:
                issues.append(
                    f"`{col}` has values with leading/trailing whitespace: "
                    f"{whitespace_variants}."
                )

            # Detect casing inconsistencies: distinct raw values that collapse after lowercasing
            n_raw = normalized.nunique()
            n_unique_raw = len(unique_raw)
            if n_raw < n_unique_raw:
                collapsed = {}
                for v in unique_raw:
                    key = v.strip().lower()
                    collapsed.setdefault(key, []).append(repr(v))
                dupes = {k: vs for k, vs in collapsed.items() if len(vs) > 1}
                for canonical, variants in dupes.items():
                    issues.append(
                        f"`{col}` has casing/whitespace variants that map to '{canonical}': "
                        f"{', '.join(variants)}."
                    )

            # Detect likely typos via edit distance (only for low-cardinality columns)
            norm_unique = sorted(normalized.unique())
            if len(norm_unique) <= 50:
                for i, a in enumerate(norm_unique):
                    for b in norm_unique[i + 1:]:
                        dist = _levenshtein(a, b)
                        if 0 < dist <= 2 and len(a) > 3 and len(b) > 3:
                            issues.append(
                                f"`{col}` has possible typo: '{a}' vs '{b}' "
                                f"(edit distance {dist})."
                            )

        # --- Numerical columns ---
        elif np.issubdtype(series.dtype, np.number) and col != TARGET:
            desc = series.describe()
            median = desc["50%"]
            col_max = desc["max"]
            col_min = desc["min"]
            skew = series.skew()

            if abs(skew) > 2:
                issues.append(
                    f"`{col}` is highly skewed (skew={skew:.2f}, "
                    f"median={median:.2f}, max={col_max:.2f}). "
                    f"Consider log transform or capping."
                )

            if col_min < 0 and col_min < (desc["25%"] - 3 * (desc["75%"] - desc["25%"])):
                issues.append(f"`{col}` has suspicious negative outliers (min={col_min:.2f}).")

            iqr = desc["75%"] - desc["25%"]
            upper_fence = desc["75%"] + 3 * iqr
            n_upper_outliers = (series > upper_fence).sum()
            if n_upper_outliers > 0:
                issues.append(
                    f"`{col}` has {n_upper_outliers:,} extreme upper outliers "
                    f"(>{upper_fence:.2f}, 3x IQR fence)."
                )

    return issues


def _levenshtein(a, b):
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_key_observations(df):
    """Dynamically compute key observations about the dataset."""
    observations = []

    # Click rate by group (if group column exists)
    if "group" in df.columns:
        grouped = df.copy()
        grouped["group"] = grouped["group"].astype(str).str.strip().str.lower()
        rates = grouped.groupby("group")[TARGET].mean()
        rate_strs = [f"'{g}': {r:.1%}" for g, r in rates.items() if g != "nan"]
        if rate_strs:
            observations.append(f"Click rate by group — {', '.join(rate_strs)}.")

    # Identify continuous vs categorical features
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if num_cols:
        observations.append(f"Continuous features: {', '.join(f'`{c}`' for c in num_cols)}.")
    if cat_cols:
        observations.append(f"Categorical features: {', '.join(f'`{c}`' for c in cat_cols)}.")

    # Flag datetime-like string columns
    for col in cat_cols:
        sample = df[col].dropna().head(100)
        parsed = pd.to_datetime(sample, errors="coerce")
        frac_parsed = parsed.notna().mean()
        if frac_parsed > 0.8:
            observations.append(
                f"`{col}` looks like a datetime string ({frac_parsed:.0%} parseable). "
                f"Extract temporal features or drop."
            )

    return observations


def write_summary(df, counts):
    missing = df.isnull().sum()
    majority = counts.max()
    minority = counts.min()
    ratio = majority / minority

    lines = [
        "# EDA Summary",
        "",
        "## Dataset",
        f"- **Rows**: {df.shape[0]:,}",
        f"- **Columns**: {df.shape[1]}",
        "",
        "## Target Variable (`click`)",
        f"- Class 0 (no click): {counts[0]:,} ({counts[0]/len(df):.1%})",
        f"- Class 1 (click): {counts[1]:,} ({counts[1]/len(df):.1%})",
        f"- Imbalance ratio: {ratio:.2f}:1",
        "",
        "## Missing Values",
    ]
    for col in df.columns:
        if missing[col] > 0:
            lines.append(f"- `{col}`: {missing[col]:,} ({missing[col]/len(df):.2%})")
    lines.append(f"- **Total**: {missing.sum():,}")

    # Dynamic data quality issues
    issues = detect_quality_issues(df)
    lines += ["", "## Data Quality Issues"]
    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- No data quality issues detected.")

    # Dynamic key observations
    observations = compute_key_observations(df)
    lines += ["", "## Key Observations"]
    if observations:
        for obs in observations:
            lines.append(f"- {obs}")
    else:
        lines.append("- No additional observations.")

    lines += [
        "",
        "## Plots",
        "",
        "### Class Distribution",
        "![Class Balance](class_balance.png)",
        "",
        "### Numerical Feature Distributions",
        "![Distributions](distributions.png)",
        "",
        "### Correlation Heatmap",
        "![Correlations](correlations.png)",
        "",
        "### Missing Value Patterns",
        "![Missing Values](missing_values.png)",
    ]

    path = OUTPUT_DIR / "eda_summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def main():
    print("Loading data...")
    df = load_data()

    print_overview(df)
    counts = print_class_balance(df)

    print("\nGenerating plots...")
    plot_class_balance(counts)
    plot_distributions(df)
    plot_correlations(df)
    plot_missing_values(df)

    print("\nWriting summary...")
    write_summary(df, counts)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
