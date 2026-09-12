import pandas as pd
from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM - MODEL DATA PREPARATION
# =========================================================


def load_feature_data(file_path):
    """Load feature-engineered dataset."""

    print("\nLoading feature-engineered dataset...")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df


def chronological_split(df, test_size=0.20):
    """
    Split data chronologically.

    First dates -> Training
    Last dates  -> Testing
    """

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL TRAIN-TEST SPLIT")
    print("=" * 60)

    # Get unique dates in chronological order
    unique_dates = sorted(
        df["Date"].unique()
    )

    # Calculate split position
    split_index = int(
        len(unique_dates) * (1 - test_size)
    )

    # Split date
    split_date = unique_dates[split_index]

    # Training data
    train_df = df[
        df["Date"] < split_date
    ].copy()

    # Testing data
    test_df = df[
        df["Date"] >= split_date
    ].copy()

    print(f"\nTotal unique dates: {len(unique_dates)}")

    print(
        f"\nTraining date range:\n"
        f"{train_df['Date'].min().date()} "
        f"to "
        f"{train_df['Date'].max().date()}"
    )

    print(
        f"\nTesting date range:\n"
        f"{test_df['Date'].min().date()} "
        f"to "
        f"{test_df['Date'].max().date()}"
    )

    print(
        f"\nTraining records: {len(train_df)}"
    )

    print(
        f"Testing records: {len(test_df)}"
    )

    return train_df, test_df


def display_feature_information():
    """Display model input features."""

    print("\n" + "=" * 60)
    print("MODEL FEATURES")
    print("=" * 60)

    print(
        """
Target Variable:
    Units_Sold

Input Features:
    Product_ID
    Store_ID
    Promotion
    Holiday
    Year
    Month
    Day
    Day_of_Week
    Quarter
    Lag_1
    Lag_7
    Lag_14
    Lag_30
    Rolling_Mean_7
    Rolling_Mean_30
        """
    )


def save_split_data(train_df, test_df, train_path, test_path):
    """Save training and testing datasets."""

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("TRAIN AND TEST DATA SAVED")
    print("=" * 60)

    print(f"\nTraining data:\n{train_path}")

    print(f"\nTesting data:\n{test_path}")


def main():

    # -----------------------------------------------------
    # PROJECT ROOT
    # -----------------------------------------------------

    project_root = Path(
        __file__
    ).resolve().parent.parent


    # -----------------------------------------------------
    # FILE PATHS
    # -----------------------------------------------------

    input_path = (
        project_root
        / "data"
        / "processed"
        / "sales_features.csv"
    )

    train_path = (
        project_root
        / "data"
        / "processed"
        / "train_data.csv"
    )

    test_path = (
        project_root
        / "data"
        / "processed"
        / "test_data.csv"
    )


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_feature_data(
        input_path
    )


    # -----------------------------------------------------
    # SORT DATA BY DATE
    # -----------------------------------------------------

    df = df.sort_values(
        by=[
            "Date",
            "Product_ID",
            "Store_ID"
        ]
    ).reset_index(
        drop=True
    )


    # -----------------------------------------------------
    # DISPLAY FEATURES
    # -----------------------------------------------------

    display_feature_information()


    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------

    train_df, test_df = chronological_split(
        df,
        test_size=0.20
    )


    # -----------------------------------------------------
    # SAVE DATA
    # -----------------------------------------------------

    save_split_data(
        train_df,
        test_df,
        train_path,
        test_path
    )


    print("\n" + "=" * 60)
    print("MODEL PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":

    main()