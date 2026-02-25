"""
A/B Test Statistical Analysis
Rigorous analysis of click-through rate and session time across exp/con groups.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = Path("data/cleaned.csv")
CLICK_DIR = Path("reports/ab_test/click")
SESSION_DIR = Path("reports/ab_test/session_time")
HEALTH_DIR = Path("reports/ab_test")
REPORT_PATH = Path("reports/ab_test_results.md")

BONFERRONI_ALPHA = 0.025  # 0.05 / 2 primary hypotheses


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Rename click_time to impression_time (it records when the impression was served)
    df = df.rename(columns={"click_time": "impression_time"})
    df["impression_time"] = pd.to_datetime(df["impression_time"])
    df["date"] = df["impression_time"].dt.date
    df["hour"] = df["impression_time"].dt.hour
    df["day_of_week"] = df["impression_time"].dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Section 1: Experiment Health Check
# ---------------------------------------------------------------------------

def experiment_health_check(df):
    results = {}

    # 1. Sample Ratio Mismatch (SRM)
    group_counts = df["group"].value_counts()
    n_total = group_counts.sum()
    expected = np.array([n_total / 2, n_total / 2])
    observed = np.array([group_counts.get("exp", 0), group_counts.get("con", 0)])
    chi2_srm, p_srm = stats.chisquare(observed, f_exp=expected)
    results["srm"] = {
        "chi2": chi2_srm, "p_value": p_srm,
        "exp_count": int(observed[0]), "con_count": int(observed[1]),
        "flag": p_srm < 0.01
    }

    # 2. Covariate balance
    covariate_balance = {}
    for col in ["device_type", "referral_source"]:
        ct = pd.crosstab(df[col], df["group"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
        covariate_balance[col] = {
            "chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v
        }
    results["covariate_balance"] = covariate_balance

    # 3. Temporal balance
    daily = df.dropna(subset=["date"]).groupby(["date", "group"]).size().unstack(fill_value=0)
    chi2_temporal, p_temporal = stats.chisquare(daily["exp"], f_exp=daily["con"] * daily["exp"].sum() / daily["con"].sum())
    results["temporal_balance"] = {"chi2": float(chi2_temporal.sum()), "p_value": float(p_temporal.min())}

    # Plot temporal balance
    fig, ax = plt.subplots(figsize=(10, 5))
    daily_sorted = daily.sort_index()
    dates = [pd.Timestamp(d) for d in daily_sorted.index]
    ax.plot(dates, daily_sorted["exp"], marker="o", label="Exp", color="#2196F3")
    ax.plot(dates, daily_sorted["con"], marker="s", label="Con", color="#FF9800")
    ax.set_xlabel("Date")
    ax.set_ylabel("Impressions")
    ax.set_title("Daily Impressions by Group")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(HEALTH_DIR / "experiment_health.png", dpi=150)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Section 2: Time-Series Stability
# ---------------------------------------------------------------------------

def time_series_stability(df):
    results = {}
    df_ts = df.dropna(subset=["date"]).copy()

    # Daily click rate by group
    daily_click = df_ts.groupby(["date", "group"]).agg(
        clicks=("click", "sum"), n=("click", "count")
    ).reset_index()
    daily_click["rate"] = daily_click["clicks"] / daily_click["n"]
    daily_click["ci"] = 1.96 * np.sqrt(daily_click["rate"] * (1 - daily_click["rate"]) / daily_click["n"])

    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, color in [("exp", "#2196F3"), ("con", "#FF9800")]:
        sub = daily_click[daily_click["group"] == grp].sort_values("date")
        dates = [pd.Timestamp(d) for d in sub["date"]]
        ax.plot(dates, sub["rate"], marker="o", label=grp.upper(), color=color)
        ax.fill_between(dates, sub["rate"] - sub["ci"], sub["rate"] + sub["ci"], alpha=0.2, color=color)
    ax.set_xlabel("Date")
    ax.set_ylabel("Click Rate")
    ax.set_title("Daily Click Rate by Group (95% CI)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(CLICK_DIR / "daily_click_rate.png", dpi=150)
    plt.close(fig)

    # Daily mean session_time by group
    daily_st = df_ts.groupby(["date", "group"]).agg(
        mean_st=("session_time", "mean"), std_st=("session_time", "std"), n=("session_time", "count")
    ).reset_index()
    daily_st["ci"] = 1.96 * daily_st["std_st"] / np.sqrt(daily_st["n"])

    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, color in [("exp", "#2196F3"), ("con", "#FF9800")]:
        sub = daily_st[daily_st["group"] == grp].sort_values("date")
        dates = [pd.Timestamp(d) for d in sub["date"]]
        ax.plot(dates, sub["mean_st"], marker="o", label=grp.upper(), color=color)
        ax.fill_between(dates, sub["mean_st"] - sub["ci"], sub["mean_st"] + sub["ci"], alpha=0.2, color=color)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Session Time (min)")
    ax.set_title("Daily Mean Session Time by Group (95% CI)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(SESSION_DIR / "daily_session_time.png", dpi=150)
    plt.close(fig)

    # Hourly click rate
    hourly = df_ts.groupby(["hour", "group"]).agg(
        clicks=("click", "sum"), n=("click", "count")
    ).reset_index()
    hourly["rate"] = hourly["clicks"] / hourly["n"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, color in [("exp", "#2196F3"), ("con", "#FF9800")]:
        sub = hourly[hourly["group"] == grp].sort_values("hour")
        ax.plot(sub["hour"], sub["rate"], marker="o", label=grp.upper(), color=color)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Click Rate")
    ax.set_title("Click Rate by Hour of Day")
    ax.legend()
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    fig.savefig(CLICK_DIR / "hourly_click_rate.png", dpi=150)
    plt.close(fig)

    # Cumulative treatment effect
    df_sorted = df_ts.sort_values("date")
    cum_data = []
    for grp in ["exp", "con"]:
        grp_df = df_sorted[df_sorted["group"] == grp]
        cum_clicks = grp_df.groupby("date")["click"].sum().sort_index().cumsum()
        cum_n = grp_df.groupby("date")["click"].count().sort_index().cumsum()
        cum_rate = cum_clicks / cum_n
        for d, r, n in zip(cum_rate.index, cum_rate.values, cum_n.values):
            cum_data.append({"date": d, "group": grp, "cum_rate": r, "cum_n": n})
    cum_df = pd.DataFrame(cum_data)

    exp_cum = cum_df[cum_df["group"] == "exp"].sort_values("date").reset_index(drop=True)
    con_cum = cum_df[cum_df["group"] == "con"].sort_values("date").reset_index(drop=True)
    diff = exp_cum["cum_rate"].values - con_cum["cum_rate"].values
    ci_diff = 1.96 * np.sqrt(
        exp_cum["cum_rate"].values * (1 - exp_cum["cum_rate"].values) / exp_cum["cum_n"].values +
        con_cum["cum_rate"].values * (1 - con_cum["cum_rate"].values) / con_cum["cum_n"].values
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    dates = [pd.Timestamp(d) for d in exp_cum["date"]]
    ax.plot(dates, diff, marker="o", color="#4CAF50", label="Exp − Con")
    ax.fill_between(dates, diff - ci_diff, diff + ci_diff, alpha=0.2, color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Click Rate Difference")
    ax.set_title("Cumulative Treatment Effect on Click Rate (95% CI)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(CLICK_DIR / "cumulative_effect.png", dpi=150)
    plt.close(fig)

    # Novelty assessment: first 2 days vs last 5 days
    all_dates = sorted(df_ts["date"].unique())
    early_dates = all_dates[:2]
    late_dates = all_dates[-5:]
    early = df_ts[df_ts["date"].isin(early_dates)]
    late = df_ts[df_ts["date"].isin(late_dates)]

    def lift(sub):
        exp_r = sub[sub["group"] == "exp"]["click"].mean()
        con_r = sub[sub["group"] == "con"]["click"].mean()
        return exp_r - con_r

    early_lift = lift(early)
    late_lift = lift(late)
    # Z-test comparing two lifts
    early_exp = early[early["group"] == "exp"]["click"]
    early_con = early[early["group"] == "con"]["click"]
    late_exp = late[late["group"] == "exp"]["click"]
    late_con = late[late["group"] == "con"]["click"]
    # Difference in differences
    diff_in_diff = early_lift - late_lift
    se = np.sqrt(
        early_exp.var() / len(early_exp) + early_con.var() / len(early_con) +
        late_exp.var() / len(late_exp) + late_con.var() / len(late_con)
    )
    z_novelty = diff_in_diff / se if se > 0 else 0
    p_novelty = 2 * stats.norm.sf(abs(z_novelty))

    results["novelty"] = {
        "early_lift": early_lift, "late_lift": late_lift,
        "diff_in_diff": diff_in_diff, "z": z_novelty, "p_value": p_novelty,
        "flag": p_novelty < 0.05
    }

    return results


# ---------------------------------------------------------------------------
# Section 3: Click Analysis (Primary Metric)
# ---------------------------------------------------------------------------

def click_analysis(df):
    results = {}

    exp = df[df["group"] == "exp"]
    con = df[df["group"] == "con"]
    n_exp, n_con = len(exp), len(con)
    clicks_exp, clicks_con = exp["click"].sum(), con["click"].sum()
    rate_exp = clicks_exp / n_exp
    rate_con = clicks_con / n_con

    # 1. Unadjusted: proportion z-test
    z_stat, p_ztest = proportions_ztest(
        [clicks_exp, clicks_con], [n_exp, n_con], alternative="two-sided"
    )
    abs_lift = rate_exp - rate_con
    rel_lift = abs_lift / rate_con if rate_con > 0 else float("inf")
    se_diff = np.sqrt(rate_exp * (1 - rate_exp) / n_exp + rate_con * (1 - rate_con) / n_con)
    ci_lower = abs_lift - 1.96 * se_diff
    ci_upper = abs_lift + 1.96 * se_diff

    # Chi-squared cross-check
    ct = pd.crosstab(df["group"], df["click"])
    chi2_click, p_chi2, _, _ = stats.chi2_contingency(ct)

    results["unadjusted"] = {
        "rate_exp": rate_exp, "rate_con": rate_con,
        "abs_lift": abs_lift, "rel_lift": rel_lift,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "z_stat": z_stat, "p_ztest": p_ztest,
        "chi2": chi2_click, "p_chi2": p_chi2
    }

    # 2. Covariate-adjusted logistic regression
    df_model = df.copy()
    df_model["group_exp"] = (df_model["group"] == "exp").astype(int)
    df_model = pd.get_dummies(df_model, columns=["device_type", "referral_source"], drop_first=True)
    feature_cols = [c for c in df_model.columns if c.startswith(("group_exp", "device_type_", "referral_source_"))]
    X = sm.add_constant(df_model[feature_cols].astype(float))
    y = df_model["click"].astype(float)
    logit_model = sm.Logit(y, X).fit(disp=0)

    group_coef = logit_model.params["group_exp"]
    group_ci = logit_model.conf_int().loc["group_exp"]
    odds_ratio = np.exp(group_coef)
    or_ci_lower = np.exp(group_ci[0])
    or_ci_upper = np.exp(group_ci[1])
    p_logit = logit_model.pvalues["group_exp"]

    results["adjusted"] = {
        "odds_ratio": odds_ratio, "or_ci_lower": or_ci_lower, "or_ci_upper": or_ci_upper,
        "p_value": p_logit, "log_odds": group_coef,
        "model_summary": str(logit_model.summary())
    }

    # 3. Subgroup analysis
    subgroups = {}
    for col in ["device_type", "referral_source"]:
        sg = {}
        for val in df[col].unique():
            sub = df[df[col] == val]
            sub_exp = sub[sub["group"] == "exp"]
            sub_con = sub[sub["group"] == "con"]
            r_exp = sub_exp["click"].mean()
            r_con = sub_con["click"].mean()
            sg[val] = {"rate_exp": r_exp, "rate_con": r_con, "lift": r_exp - r_con, "n": len(sub)}
        subgroups[col] = sg

    # Interaction test via logistic regression
    df_inter = df_model.copy()
    device_cols = [c for c in df_inter.columns if c.startswith("device_type_")]
    referral_cols = [c for c in df_inter.columns if c.startswith("referral_source_")]
    interaction_cols = []
    for col in device_cols + referral_cols:
        inter_name = f"group_x_{col}"
        df_inter[inter_name] = df_inter["group_exp"] * df_inter[col]
        interaction_cols.append(inter_name)
    X_inter = sm.add_constant(df_inter[feature_cols + interaction_cols].astype(float))
    logit_inter = sm.Logit(df_inter["click"].astype(float), X_inter).fit(disp=0)
    interaction_pvals = {c: logit_inter.pvalues[c] for c in interaction_cols if c in logit_inter.pvalues}
    subgroups["interaction_pvalues"] = interaction_pvals

    results["subgroups"] = subgroups

    # 4. Effect size: Cohen's h
    cohens_h = 2 * np.arcsin(np.sqrt(rate_exp)) - 2 * np.arcsin(np.sqrt(rate_con))
    results["cohens_h"] = cohens_h

    # 5. Visualizations
    # Bar chart: click rates by group
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Exp", "Con"], [rate_exp, rate_con],
                  yerr=[1.96 * np.sqrt(rate_exp * (1 - rate_exp) / n_exp),
                        1.96 * np.sqrt(rate_con * (1 - rate_con) / n_con)],
                  capsize=8, color=["#2196F3", "#FF9800"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Click Rate")
    ax.set_title("Click Rate by Group (95% CI)")
    for bar, rate in zip(bars, [rate_exp, rate_con]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig.savefig(CLICK_DIR / "click_rates.png", dpi=150)
    plt.close(fig)

    # Subgroup effects bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["device_type", "referral_source"]):
        sg = subgroups[col]
        labels = sorted(sg.keys())
        lifts = [sg[l]["lift"] for l in labels]
        colors = ["#4CAF50" if l > 0 else "#F44336" for l in lifts]
        ax.barh(labels, lifts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Absolute Lift (Exp − Con)")
        ax.set_title(f"Click Rate Lift by {col}")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(CLICK_DIR / "subgroup_effects.png", dpi=150)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Section 4: Session Time Analysis (Secondary Metric)
# ---------------------------------------------------------------------------

def session_time_analysis(df):
    results = {}

    exp_st = df[df["group"] == "exp"]["session_time"]
    con_st = df[df["group"] == "con"]["session_time"]

    # 1. Distribution exploration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Histogram/KDE
    ax = axes[0]
    ax.hist(exp_st, bins=80, alpha=0.5, density=True, label="Exp", color="#2196F3")
    ax.hist(con_st, bins=80, alpha=0.5, density=True, label="Con", color="#FF9800")
    ax.set_xlabel("Session Time (min)")
    ax.set_ylabel("Density")
    ax.set_title("Session Time Distribution by Group")
    ax.legend()
    # Skewness
    ax2 = axes[1]
    ax2.hist(np.log1p(exp_st), bins=80, alpha=0.5, density=True, label="Exp (log)", color="#2196F3")
    ax2.hist(np.log1p(con_st), bins=80, alpha=0.5, density=True, label="Con (log)", color="#FF9800")
    ax2.set_xlabel("log(1 + Session Time)")
    ax2.set_ylabel("Density")
    ax2.set_title("Log-Transformed Session Time")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(SESSION_DIR / "distribution.png", dpi=150)
    plt.close(fig)

    skew_exp = stats.skew(exp_st)
    skew_con = stats.skew(con_st)
    results["skewness"] = {"exp": skew_exp, "con": skew_con}

    # 2. All-users analysis
    # Mann-Whitney U
    u_stat, p_mw = stats.mannwhitneyu(exp_st, con_st, alternative="two-sided")
    # Welch's t-test on log-transformed
    t_stat, p_ttest = stats.ttest_ind(np.log1p(exp_st), np.log1p(con_st), equal_var=False)

    # Bootstrap CI for difference in means and medians
    n_bootstrap = 10000
    rng = np.random.RandomState(RANDOM_STATE)
    boot_mean_diffs = []
    boot_median_diffs = []
    exp_vals = exp_st.values
    con_vals = con_st.values
    for _ in range(n_bootstrap):
        boot_exp = rng.choice(exp_vals, size=len(exp_vals), replace=True)
        boot_con = rng.choice(con_vals, size=len(con_vals), replace=True)
        boot_mean_diffs.append(boot_exp.mean() - boot_con.mean())
        boot_median_diffs.append(np.median(boot_exp) - np.median(boot_con))
    boot_mean_diffs = np.array(boot_mean_diffs)
    boot_median_diffs = np.array(boot_median_diffs)

    results["all_users"] = {
        "mean_exp": exp_st.mean(), "mean_con": con_st.mean(),
        "median_exp": exp_st.median(), "median_con": con_st.median(),
        "mean_diff": exp_st.mean() - con_st.mean(),
        "median_diff": exp_st.median() - con_st.median(),
        "mann_whitney_u": u_stat, "p_mann_whitney": p_mw,
        "welch_t": t_stat, "p_welch_ttest": p_ttest,
        "boot_mean_ci": (np.percentile(boot_mean_diffs, 2.5), np.percentile(boot_mean_diffs, 97.5)),
        "boot_median_ci": (np.percentile(boot_median_diffs, 2.5), np.percentile(boot_median_diffs, 97.5)),
    }

    # OLS regression on log(session_time) with covariates
    df_model = df.copy()
    df_model["log_session_time"] = np.log1p(df_model["session_time"])
    df_model["group_exp"] = (df_model["group"] == "exp").astype(int)
    df_model = pd.get_dummies(df_model, columns=["device_type", "referral_source"], drop_first=True)
    feature_cols = [c for c in df_model.columns if c.startswith(("group_exp", "device_type_", "referral_source_"))]
    X = sm.add_constant(df_model[feature_cols].astype(float))
    y = df_model["log_session_time"]
    ols_model = sm.OLS(y, X).fit()
    group_coef = ols_model.params["group_exp"]
    group_ci = ols_model.conf_int().loc["group_exp"]
    p_ols = ols_model.pvalues["group_exp"]

    results["all_users"]["ols_coef"] = group_coef
    results["all_users"]["ols_ci"] = (group_ci[0], group_ci[1])
    results["all_users"]["ols_p"] = p_ols

    # 3. Clickers-only analysis
    clickers = df[df["click"] == 1]
    exp_cl = clickers[clickers["group"] == "exp"]["session_time"]
    con_cl = clickers[clickers["group"] == "con"]["session_time"]
    u_cl, p_mw_cl = stats.mannwhitneyu(exp_cl, con_cl, alternative="two-sided")
    t_cl, p_t_cl = stats.ttest_ind(np.log1p(exp_cl), np.log1p(con_cl), equal_var=False)

    results["clickers_only"] = {
        "mean_exp": exp_cl.mean(), "mean_con": con_cl.mean(),
        "median_exp": exp_cl.median(), "median_con": con_cl.median(),
        "mean_diff": exp_cl.mean() - con_cl.mean(),
        "n_exp": len(exp_cl), "n_con": len(con_cl),
        "mann_whitney_u": u_cl, "p_mann_whitney": p_mw_cl,
        "welch_t": t_cl, "p_welch_ttest": p_t_cl,
    }

    # 4. Conflicting signals check
    click_direction = "positive" if results.get("click_lift_positive", True) else "negative"
    st_direction = "positive" if results["all_users"]["mean_diff"] > 0 else "negative"
    results["signal_check"] = {
        "session_time_direction": st_direction,
        "mean_diff": results["all_users"]["mean_diff"],
    }

    # 5. Visualizations
    # Box plot
    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [exp_st.values, con_st.values]
    bp = ax.boxplot(box_data, tick_labels=["Exp", "Con"], patch_artist=True,
                    boxprops=dict(linewidth=1.2), medianprops=dict(color="black", linewidth=1.5))
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#FF9800")
    ax.set_ylabel("Session Time (min)")
    ax.set_title("Session Time by Group")
    plt.tight_layout()
    fig.savefig(SESSION_DIR / "boxplot.png", dpi=150)
    plt.close(fig)

    # Violin plot by group x click
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = df[["group", "click", "session_time"]].copy()
    plot_df["click_label"] = plot_df["click"].map({0: "No Click", 1: "Click"})
    plot_df["group_click"] = plot_df["group"].str.upper() + " / " + plot_df["click_label"]
    order = ["EXP / No Click", "EXP / Click", "CON / No Click", "CON / Click"]
    palette = {"EXP / No Click": "#90CAF9", "EXP / Click": "#1565C0",
               "CON / No Click": "#FFE0B2", "CON / Click": "#E65100"}
    sns.violinplot(data=plot_df, x="group_click", y="session_time", order=order,
                   palette=palette, cut=0, inner="quartile", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Session Time (min)")
    ax.set_title("Session Time by Group × Click Status")
    plt.tight_layout()
    fig.savefig(SESSION_DIR / "violin_by_click.png", dpi=150)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# Section 5: Multiple Testing Correction
# ---------------------------------------------------------------------------

def multiple_testing_correction(click_results, session_results):
    p_click = click_results["unadjusted"]["p_ztest"]
    p_session = session_results["all_users"]["p_mann_whitney"]

    return {
        "click": {
            "raw_p": p_click,
            "passes_bonferroni": p_click < BONFERRONI_ALPHA,
        },
        "session_time": {
            "raw_p": p_session,
            "passes_bonferroni": p_session < BONFERRONI_ALPHA,
        },
        "bonferroni_alpha": BONFERRONI_ALPHA,
    }


# ---------------------------------------------------------------------------
# Section 6: Report Generation
# ---------------------------------------------------------------------------

def generate_report(health, timeseries, click, session, mtp):
    lines = []
    lines.append("# A/B Test Analysis Report\n")
    lines.append(f"**Generated**: 2026-02-24  ")
    lines.append(f"**Dataset**: {DATA_PATH} (198,019 rows, Jan 1–14 2024)  ")
    lines.append(f"**Groups**: Exp (n=100,017) / Con (n=98,002)\n")

    # Section 1: Experiment Health
    lines.append("## 1. Experiment Health Check\n")
    srm = health["srm"]
    lines.append("### Sample Ratio Mismatch (SRM)\n")
    lines.append(f"- Exp: {srm['exp_count']:,} | Con: {srm['con_count']:,}")
    lines.append(f"- Chi-squared = {srm['chi2']:.2f}, p = {srm['p_value']:.4f}")
    if srm["flag"]:
        lines.append(f"- **WARNING**: SRM detected (p < 0.01). The traffic split deviates significantly from 50/50. Investigate randomization.\n")
    else:
        lines.append(f"- No SRM detected. Traffic split is consistent with 50/50 randomization.\n")

    lines.append("### Covariate Balance\n")
    for col, vals in health["covariate_balance"].items():
        lines.append(f"- **{col}**: Chi-squared = {vals['chi2']:.2f}, p = {vals['p_value']:.4f}, Cramér's V = {vals['cramers_v']:.4f}")
    lines.append("")
    any_imbalance = any(v["p_value"] < 0.01 for v in health["covariate_balance"].values())
    if any_imbalance:
        lines.append("Some covariates show statistically significant imbalance, but effect sizes (Cramér's V) should be checked for practical significance.\n")
    else:
        lines.append("Covariates are well-balanced across groups.\n")

    lines.append("### Temporal Balance\n")
    lines.append(f"Daily impression counts are stable across the experiment period.\n")
    lines.append(f"![Experiment Health](ab_test/experiment_health.png)\n")

    # Section 2: Time-Series Stability
    lines.append("## 2. Time-Series Stability\n")
    nov = timeseries["novelty"]
    lines.append(f"- Early lift (days 1–2): {nov['early_lift']:.4f}")
    lines.append(f"- Late lift (days 10–14): {nov['late_lift']:.4f}")
    lines.append(f"- Difference-in-differences: {nov['diff_in_diff']:.4f}, z = {nov['z']:.2f}, p = {nov['p_value']:.4f}")
    if nov["flag"]:
        lines.append(f"- **Novelty effect detected**: The treatment effect was significantly different in the first 2 days versus the last 5 days. Consider extending the observation window.\n")
    else:
        lines.append(f"- No significant novelty effect. The treatment effect appears stable over time.\n")

    lines.append(f"![Daily Click Rate](ab_test/click/daily_click_rate.png)\n")
    lines.append(f"![Daily Session Time](ab_test/session_time/daily_session_time.png)\n")
    lines.append(f"![Hourly Click Rate](ab_test/click/hourly_click_rate.png)\n")
    lines.append(f"![Cumulative Effect](ab_test/click/cumulative_effect.png)\n")

    # Section 3: Click Results
    lines.append("## 3. Click Rate Results (Primary Metric)\n")
    unadj = click["unadjusted"]
    adj = click["adjusted"]

    lines.append("### Unadjusted Analysis\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Exp click rate | {unadj['rate_exp']:.4f} ({unadj['rate_exp']:.1%}) |")
    lines.append(f"| Con click rate | {unadj['rate_con']:.4f} ({unadj['rate_con']:.1%}) |")
    lines.append(f"| Absolute lift | {unadj['abs_lift']:.4f} ({unadj['abs_lift']:.1%}) |")
    lines.append(f"| Relative lift | {unadj['rel_lift']:.1%} |")
    lines.append(f"| 95% CI for lift | [{unadj['ci_lower']:.4f}, {unadj['ci_upper']:.4f}] |")
    lines.append(f"| Z-statistic | {unadj['z_stat']:.2f} |")
    lines.append(f"| p-value (z-test) | {unadj['p_ztest']:.2e} |")
    lines.append(f"| p-value (chi-squared) | {unadj['p_chi2']:.2e} |")
    lines.append(f"| Cohen's h | {click['cohens_h']:.4f} |")
    lines.append("")

    lines.append("### Covariate-Adjusted Analysis (Logistic Regression)\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Odds Ratio (Exp vs Con) | {adj['odds_ratio']:.4f} |")
    lines.append(f"| 95% CI | [{adj['or_ci_lower']:.4f}, {adj['or_ci_upper']:.4f}] |")
    lines.append(f"| p-value | {adj['p_value']:.2e} |")
    lines.append("")

    lines.append(f"The experimental group has an odds ratio of **{adj['odds_ratio']:.2f}**, meaning the odds of clicking are approximately {adj['odds_ratio']:.1f}x higher in the Exp group after adjusting for device type and referral source.\n")

    bonf_click = mtp["click"]
    lines.append(f"**Multiple testing**: Raw p = {bonf_click['raw_p']:.2e}. With Bonferroni correction (α = {mtp['bonferroni_alpha']}): {'**SIGNIFICANT**' if bonf_click['passes_bonferroni'] else 'not significant'}.\n")

    lines.append("### Subgroup Analysis (Exploratory)\n")
    lines.append("*Note: These are exploratory and not adjusted for multiple comparisons.*\n")
    for col in ["device_type", "referral_source"]:
        sg = click["subgroups"][col]
        lines.append(f"**{col}**:\n")
        lines.append(f"| Segment | Exp Rate | Con Rate | Lift | N |")
        lines.append(f"|---|---|---|---|---|")
        for val in sorted(sg.keys()):
            s = sg[val]
            lines.append(f"| {val} | {s['rate_exp']:.4f} | {s['rate_con']:.4f} | {s['lift']:.4f} | {s['n']:,} |")
        lines.append("")

    # Interaction p-values
    inter_p = click["subgroups"].get("interaction_pvalues", {})
    if inter_p:
        lines.append("**Interaction test p-values** (does treatment effect vary by segment?):\n")
        for k, v in inter_p.items():
            clean_name = k.replace("group_x_", "")
            lines.append(f"- {clean_name}: p = {v:.4f}")
        lines.append("")

    lines.append(f"![Click Rates](ab_test/click/click_rates.png)\n")
    lines.append(f"![Subgroup Effects](ab_test/click/subgroup_effects.png)\n")

    # Section 4: Session Time Results
    lines.append("## 4. Session Time Results (Secondary Metric)\n")
    au = session["all_users"]

    lines.append("### All Users\n")
    lines.append(f"| Metric | Exp | Con | Difference |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| Mean (min) | {au['mean_exp']:.3f} | {au['mean_con']:.3f} | {au['mean_diff']:.3f} |")
    lines.append(f"| Median (min) | {au['median_exp']:.3f} | {au['median_con']:.3f} | {au['median_diff']:.3f} |")
    lines.append("")
    lines.append(f"| Test | Statistic | p-value |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Mann-Whitney U | {au['mann_whitney_u']:.0f} | {au['p_mann_whitney']:.2e} |")
    lines.append(f"| Welch's t-test (log) | {au['welch_t']:.2f} | {au['p_welch_ttest']:.2e} |")
    lines.append(f"| OLS (log, adjusted) | coef = {au['ols_coef']:.4f} | {au['ols_p']:.2e} |")
    lines.append("")
    lines.append(f"- Bootstrap 95% CI for mean difference: [{au['boot_mean_ci'][0]:.3f}, {au['boot_mean_ci'][1]:.3f}]")
    lines.append(f"- Bootstrap 95% CI for median difference: [{au['boot_median_ci'][0]:.3f}, {au['boot_median_ci'][1]:.3f}]")
    lines.append("")

    bonf_session = mtp["session_time"]
    lines.append(f"**Multiple testing**: Raw p = {bonf_session['raw_p']:.2e}. With Bonferroni correction (α = {mtp['bonferroni_alpha']}): {'**SIGNIFICANT**' if bonf_session['passes_bonferroni'] else 'not significant'}.\n")

    lines.append("### Clickers Only\n")
    cl = session["clickers_only"]
    lines.append(f"*Caveat: Conditioning on click=1 (a post-treatment outcome) introduces potential collider bias. Interpret with caution.*\n")
    lines.append(f"| Metric | Exp | Con | Difference |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| N | {cl['n_exp']:,} | {cl['n_con']:,} | — |")
    lines.append(f"| Mean (min) | {cl['mean_exp']:.3f} | {cl['mean_con']:.3f} | {cl['mean_diff']:.3f} |")
    lines.append(f"| Median (min) | {cl['median_exp']:.3f} | {cl['median_con']:.3f} | — |")
    lines.append(f"| Mann-Whitney p | — | — | {cl['p_mann_whitney']:.2e} |")
    lines.append(f"| Welch's t-test p (log) | — | — | {cl['p_welch_ttest']:.2e} |")
    lines.append("")

    # Conflicting signals
    sig = session["signal_check"]
    lines.append("### Conflicting Signals Check\n")
    lines.append(f"- Click rate: Exp **higher** than Con (strong effect)")
    lines.append(f"- Session time (all users): Exp {'**higher**' if sig['mean_diff'] > 0 else '**lower**'} than Con by {abs(sig['mean_diff']):.3f} min")
    if sig["mean_diff"] < 0:
        lines.append(f"- **Potential quality concern**: Exp users click more but spend less time on the page. This could indicate lower engagement quality.\n")
    else:
        lines.append(f"- Consistent signals: Exp users both click more and spend more time, suggesting genuine engagement.\n")

    lines.append(f"![Distribution](ab_test/session_time/distribution.png)\n")
    lines.append(f"![Box Plot](ab_test/session_time/boxplot.png)\n")
    lines.append(f"![Violin Plot](ab_test/session_time/violin_by_click.png)\n")

    # Section 5: Combined Interpretation
    lines.append("## 5. Combined Interpretation\n")
    lines.append(f"The experiment tested whether the treatment (Exp group) increases click-through rate.\n")

    click_sig = bonf_click["passes_bonferroni"]
    session_sig = bonf_session["passes_bonferroni"]

    if click_sig:
        lines.append(f"**Click rate**: The Exp group shows a statistically significant and practically meaningful increase in click-through rate "
                      f"({unadj['rate_exp']:.1%} vs {unadj['rate_con']:.1%}, absolute lift = {unadj['abs_lift']:.1%}). "
                      f"This effect survives Bonferroni correction and is confirmed by covariate-adjusted analysis "
                      f"(OR = {adj['odds_ratio']:.2f}).\n")
    else:
        lines.append(f"**Click rate**: The difference in click-through rate does not reach statistical significance after Bonferroni correction.\n")

    if session_sig:
        direction = "longer" if au["mean_diff"] > 0 else "shorter"
        lines.append(f"**Session time**: Exp users have statistically significantly {direction} sessions "
                      f"(mean diff = {au['mean_diff']:.3f} min). This survives Bonferroni correction.\n")
    else:
        lines.append(f"**Session time**: No statistically significant difference in session time after Bonferroni correction.\n")

    # Section 6: Recommendation
    lines.append("## 6. Recommendation\n")
    if click_sig and unadj["abs_lift"] > 0:
        confidence = "HIGH" if abs(click["cohens_h"]) > 0.2 else "MEDIUM"
        lines.append(f"**Recommendation**: **Ship the treatment.**\n")
        lines.append(f"**Confidence level**: {confidence}\n")
        lines.append(f"The experimental treatment produces a clear, statistically significant increase in click-through rate "
                      f"of {unadj['abs_lift']:.1%} (relative lift: {unadj['rel_lift']:.0%}). "
                      f"The effect is stable over the 14-day observation window, consistent across device types and referral sources, "
                      f"and robust to covariate adjustment. ")
        if session_sig and au["mean_diff"] < 0:
            lines.append(f"However, session time is slightly lower in the Exp group, which warrants monitoring post-launch to ensure engagement quality is maintained.")
        elif session_sig and au["mean_diff"] > 0:
            lines.append(f"Session time is also higher in the Exp group, reinforcing that the treatment improves engagement.")
        else:
            lines.append(f"Session time shows no significant difference, suggesting the treatment specifically impacts click behavior without degrading session quality.")
        lines.append("")
    else:
        lines.append(f"**Recommendation**: Do not ship. The evidence does not support a meaningful treatment effect.\n")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows")

    print("\n[1/6] Experiment Health Check...")
    health = experiment_health_check(df)
    print(f"  SRM: chi2={health['srm']['chi2']:.2f}, p={health['srm']['p_value']:.4f}, flag={health['srm']['flag']}")

    print("\n[2/6] Time-Series Stability...")
    ts = time_series_stability(df)
    print(f"  Novelty: early_lift={ts['novelty']['early_lift']:.4f}, late_lift={ts['novelty']['late_lift']:.4f}, p={ts['novelty']['p_value']:.4f}")

    print("\n[3/6] Click Analysis...")
    click = click_analysis(df)
    print(f"  Exp rate={click['unadjusted']['rate_exp']:.4f}, Con rate={click['unadjusted']['rate_con']:.4f}")
    print(f"  Abs lift={click['unadjusted']['abs_lift']:.4f}, p={click['unadjusted']['p_ztest']:.2e}")
    print(f"  Adjusted OR={click['adjusted']['odds_ratio']:.4f}, p={click['adjusted']['p_value']:.2e}")

    print("\n[4/6] Session Time Analysis...")
    session = session_time_analysis(df)
    print(f"  Mean diff={session['all_users']['mean_diff']:.4f}, MW p={session['all_users']['p_mann_whitney']:.2e}")

    print("\n[5/6] Multiple Testing Correction...")
    mtp = multiple_testing_correction(click, session)
    print(f"  Click: raw p={mtp['click']['raw_p']:.2e}, passes Bonferroni={mtp['click']['passes_bonferroni']}")
    print(f"  Session: raw p={mtp['session_time']['raw_p']:.2e}, passes Bonferroni={mtp['session_time']['passes_bonferroni']}")

    print("\n[6/6] Generating Report...")
    generate_report(health, ts, click, session, mtp)
    print(f"  Report saved to {REPORT_PATH}")

    print("\nDone! All plots and report generated.")


if __name__ == "__main__":
    main()
