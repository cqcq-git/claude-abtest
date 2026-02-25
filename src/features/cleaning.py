"""Data cleaning pipeline for A/B test dataset."""

import pandas as pd


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def normalize_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df["group"].str.strip().str.lower()
    df["group"] = df["group"].replace({"a": "exp"})
    return df


def normalize_device_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["device_type"] = df["device_type"].str.strip().str.lower()
    return df


def normalize_referral_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["referral_source"] = df["referral_source"].str.strip().str.lower()
    df["referral_source"] = df["referral_source"].replace({"seach": "search"})
    return df


def handle_missing_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["group"])


def handle_missing_referral_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["referral_source"] = df["referral_source"].fillna("unknown")
    return df


def cap_session_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cap = df["session_time"].quantile(0.99)
    df["session_time"] = df["session_time"].clip(upper=cap)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to the raw DataFrame and return a cleaned copy."""
    df = drop_duplicates(df)
    df = normalize_group(df)
    df = normalize_device_type(df)
    df = normalize_referral_source(df)
    df = handle_missing_group(df)
    df = handle_missing_referral_source(df)
    df = cap_session_time(df)
    return df


def main():
    raw_path = "data/raw/ab_test_dataset.csv"
    out_path = "data/cleaned.csv"

    raw = pd.read_csv(raw_path)
    print(f"Raw shape: {raw.shape}")
    print(f"Raw missing:\n{raw.isnull().sum()}\n")

    cleaned = clean(raw)

    cleaned.to_csv(out_path, index=False)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Rows removed: {len(raw) - len(cleaned)}")
    print(f"\nRemaining missing:\n{cleaned.isnull().sum()}\n")
    print(f"group values: {sorted(cleaned['group'].unique())}")
    print(f"device_type values: {sorted(cleaned['device_type'].unique())}")
    print(f"referral_source values: {sorted(cleaned['referral_source'].unique())}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
