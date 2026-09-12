import pandas as pd
from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM - FEATURE ENGINEERING
# =========================================================


def load_data(file_path):
    """Load processed sales dataset."""

    print("\nLoading processed dataset...")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    print("Dataset loaded successfully!")
    print(f"Original dataset shape: {df.shape}")

    return df


def create_date_features(df):
    """Create features from the Date column."""

    print("\nCreating date features...")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter

    return df


def create_lag_features(df):
    """
    Create historical sales lag features separately
    for each Product_ID and Store_ID combination.
    """

    print("Creating lag features...")

    group_columns = ["Product_ID", "Store_ID"]

    df["Lag_1"] = (
        df.groupby(group_columns)["Units_Sold"]
        .shift(1)
    )

    df["Lag_7"] = (
        df.groupby(group_columns)["Units_Sold"]
        .shift(7)
    )

    df["Lag_14"] = (
        df.groupby(group_columns)["Units_Sold"]
        .shift(14)
    )

    df["Lag_30"] = (
        df.groupby(group_columns)["Units_Sold"]
        .shift(30)
    )

    return df


def create_rolling_features(df):
    """
    Create rolling averages using only previous data.

    shift(1) is important because it prevents
    data leakage from the current day's sales.
    """

    print("Creating rolling mean features...")

    group_columns = ["Product_ID", "Store_ID"]

    previous_sales = (
        df.groupby(group_columns)["Units_Sold"]
        .shift(1)
    )

    df["Rolling_Mean_7"] = (
        previous_sales
        .groupby(
            [
                df["Product_ID"],
                df["Store_ID"]
            ]
        )
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=7
            ).mean()
        )
    )

    df["Rolling_Mean_30"] = (
        previous_sales
        .groupby(
            [
                df["Product_ID"],
                df["Store_ID"]
            ]
        )
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30
            ).mean()
        )
    )

    return df


def remove_missing_lag_rows(df):
    """
    Remove rows where lag features cannot be created.

    The first 30 days for each Product_ID and Store_ID
    combination will not have complete historical data.
    """

    print("\nRemoving rows with incomplete historical features...")

    rows_before = len(df)

    required_features = [
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30"
    ]

    df = df.dropna(
        subset=required_features
    ).reset_index(
        drop=True
    )

    rows_after = len(df)

    print(
        f"Rows removed: {rows_before - rows_after}"
    )

    print(
        f"Remaining rows: {rows_after}"
    )

    return df


def save_feature_data(df, output_path):
    """Save feature-engineered dataset."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print(f"Final dataset shape: {df.shape}")

    print(
        f"Saved to:\n{output_path}"
    )


def main():

    # -----------------------------------------------------
    # PROJECT PATHS
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "data"
        / "processed"
        / "processed_sales.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "sales_features.csv"
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_data(input_path)

    # Sort before creating historical features
    df = df.sort_values(
        by=["Product_ID", "Store_ID", "Date"]
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # CREATE FEATURES
    # -----------------------------------------------------

    df = create_date_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    # -----------------------------------------------------
    # REMOVE INCOMPLETE RECORDS
    # -----------------------------------------------------

    df = remove_missing_lag_rows(df)

    # -----------------------------------------------------
    # FINAL SORT BY DATE
    # -----------------------------------------------------

    df = df.sort_values(
        by=["Date", "Product_ID", "Store_ID"]
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # DISPLAY INFORMATION
    # -----------------------------------------------------

    print("\nFinal columns:")

    print(
        df.columns.tolist()
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    # -----------------------------------------------------
    # SAVE DATA
    # -----------------------------------------------------

    save_feature_data(
        df,
        output_path
    )


if __name__ == "__main__":

    main()