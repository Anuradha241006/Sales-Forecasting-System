import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM - DATA PREPROCESSING
# =========================================================


def load_data(file_path):
    """Load the raw sales dataset."""
    
    print("\nLoading dataset...")
    
    df = pd.read_csv(file_path)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    return df


def check_missing_values(df):
    """Check for missing values."""
    
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    
    missing_values = df.isnull().sum()
    
    print(missing_values)
    
    return missing_values


def check_duplicates(df):
    """Check for duplicate records."""
    
    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)
    
    duplicate_count = df.duplicated().sum()
    
    print(f"Number of duplicate records: {duplicate_count}")
    
    return duplicate_count


def preprocess_data(df):
    """Perform data cleaning and preprocessing."""
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    
    # -----------------------------------------------------
    # 1. Convert Date column
    # -----------------------------------------------------
    
    print("\nConverting Date column to datetime...")
    
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )
    
    
    # Remove rows where Date conversion failed
    invalid_dates = df["Date"].isnull().sum()
    
    if invalid_dates > 0:
        
        print(
            f"Removing {invalid_dates} rows with invalid dates..."
        )
        
        df = df.dropna(
            subset=["Date"]
        )
    
    
    # -----------------------------------------------------
    # 2. Remove duplicate records
    # -----------------------------------------------------
    
    duplicates_before = df.duplicated().sum()
    
    df = df.drop_duplicates()
    
    duplicates_after = df.duplicated().sum()
    
    print(
        f"Duplicates removed: "
        f"{duplicates_before - duplicates_after}"
    )
    
    
    # -----------------------------------------------------
    # 3. Sort data
    # -----------------------------------------------------
    
    print("\nSorting data by Date...")
    
    df = df.sort_values(
        by=["Date", "Product_ID", "Store_ID"]
    ).reset_index(
        drop=True
    )
    
    
    # -----------------------------------------------------
    # 4. Check for missing values
    # -----------------------------------------------------
    
    numerical_columns = [
        "Units_Sold",
        "Revenue",
        "Promotion",
        "Holiday"
    ]
    
    for column in numerical_columns:
        
        missing_count = df[column].isnull().sum()
        
        if missing_count > 0:
            
            median_value = df[column].median()
            
            df[column] = df[column].fillna(
                median_value
            )
    
    
    # -----------------------------------------------------
    # 5. Validate numerical values
    # -----------------------------------------------------
    
    print("\nChecking invalid numerical values...")
    
    
    # Units sold cannot be negative
    invalid_units = (
        df["Units_Sold"] < 0
    ).sum()
    
    if invalid_units > 0:
        
        print(
            f"Removing {invalid_units} rows "
            f"with negative Units_Sold..."
        )
        
        df = df[
            df["Units_Sold"] >= 0
        ]
    
    
    # Revenue cannot be negative
    invalid_revenue = (
        df["Revenue"] < 0
    ).sum()
    
    if invalid_revenue > 0:
        
        print(
            f"Removing {invalid_revenue} rows "
            f"with negative Revenue..."
        )
        
        df = df[
            df["Revenue"] >= 0
        ]
    
    
    # Promotion should contain only 0 or 1
    invalid_promotion = (
        ~df["Promotion"].isin([0, 1])
    ).sum()
    
    if invalid_promotion > 0:
        
        print(
            f"Found {invalid_promotion} invalid "
            f"Promotion values."
        )
    
    
    # Holiday should contain only 0 or 1
    invalid_holiday = (
        ~df["Holiday"].isin([0, 1])
    ).sum()
    
    if invalid_holiday > 0:
        
        print(
            f"Found {invalid_holiday} invalid "
            f"Holiday values."
        )
    
    
    return df


def save_processed_data(df, output_path):
    """Save cleaned dataset."""
    
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    df.to_csv(
        output_path,
        index=False
    )
    
    print("\n" + "=" * 60)
    print("PROCESSED DATA SAVED")
    print("=" * 60)
    
    print(
        f"Saved to: {output_path}"
    )
    
    print(
        f"Final dataset shape: {df.shape}"
    )


def main():
    
    # Project root directory
    project_root = Path(__file__).resolve().parent.parent
    
    
    # File paths
    input_path = (
        project_root
        / "data"
        / "raw"
        / "sales_data.csv"
    )
    
    output_path = (
        project_root
        / "data"
        / "processed"
        / "processed_sales.csv"
    )
    
    
    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    
    df = load_data(input_path)
    
    
    # -----------------------------------------------------
    # INITIAL DATA CHECKS
    # -----------------------------------------------------
    
    check_missing_values(df)
    
    check_duplicates(df)
    
    
    # -----------------------------------------------------
    # PREPROCESS DATA
    # -----------------------------------------------------
    
    df_processed = preprocess_data(df)
    
    
    # -----------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------
    
    print("\n" + "=" * 60)
    print("FINAL DATASET INFORMATION")
    print("=" * 60)
    
    print(df_processed.info())
    
    
    print("\nFirst 5 rows:")
    
    print(
        df_processed.head()
    )
    
    
    # -----------------------------------------------------
    # SAVE DATA
    # -----------------------------------------------------
    
    save_processed_data(
        df_processed,
        output_path
    )


if __name__ == "__main__":
    
    main()