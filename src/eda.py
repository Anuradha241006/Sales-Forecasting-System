import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM - EXPLORATORY DATA ANALYSIS
# =========================================================


def load_data(file_path):
    """Load processed sales data."""

    print("\nLoading processed dataset...")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df


def save_graph(filename, output_dir):
    """Save the current graph."""

    output_path = output_dir / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graph saved: {filename}")


def daily_sales_trend(df, output_dir):
    """Plot total daily units sold."""

    daily_sales = df.groupby("Date")["Units_Sold"].sum()

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_sales.index,
        daily_sales.values
    )

    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.grid(True)

    save_graph(
        "01_daily_sales_trend.png",
        output_dir
    )


def revenue_trend(df, output_dir):
    """Plot total daily revenue."""

    daily_revenue = df.groupby("Date")["Revenue"].sum()

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily_revenue.index,
        daily_revenue.values
    )

    plt.title("Daily Revenue Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Revenue")
    plt.grid(True)

    save_graph(
        "02_revenue_trend.png",
        output_dir
    )


def monthly_sales_pattern(df, output_dir):
    """Plot monthly average sales."""

    monthly_sales = (
        df.groupby(df["Date"].dt.month)["Units_Sold"]
        .mean()
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        month_names,
        monthly_sales.values
    )

    plt.title("Monthly Average Sales Pattern")
    plt.xlabel("Month")
    plt.ylabel("Average Units Sold")

    save_graph(
        "03_monthly_sales_pattern.png",
        output_dir
    )


def product_wise_sales(df, output_dir):
    """Plot total sales by product."""

    product_sales = (
        df.groupby("Product_ID")["Units_Sold"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        product_sales.index,
        product_sales.values
    )

    plt.title("Product-wise Total Sales")
    plt.xlabel("Product ID")
    plt.ylabel("Total Units Sold")

    save_graph(
        "04_product_wise_sales.png",
        output_dir
    )


def store_wise_sales(df, output_dir):
    """Plot total sales by store."""

    store_sales = (
        df.groupby("Store_ID")["Units_Sold"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        store_sales.index,
        store_sales.values
    )

    plt.title("Store-wise Total Sales")
    plt.xlabel("Store ID")
    plt.ylabel("Total Units Sold")

    save_graph(
        "05_store_wise_sales.png",
        output_dir
    )


def promotion_impact(df, output_dir):
    """Compare sales with and without promotion."""

    promotion_sales = (
        df.groupby("Promotion")["Units_Sold"]
        .mean()
    )

    labels = [
        "No Promotion",
        "Promotion"
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        promotion_sales.values
    )

    plt.title("Promotion Impact on Sales")
    plt.xlabel("Promotion Status")
    plt.ylabel("Average Units Sold")

    save_graph(
        "06_promotion_impact.png",
        output_dir
    )


def holiday_impact(df, output_dir):
    """Compare sales on holidays and normal days."""

    holiday_sales = (
        df.groupby("Holiday")["Units_Sold"]
        .mean()
    )

    labels = [
        "Normal Day",
        "Holiday"
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        holiday_sales.values
    )

    plt.title("Holiday Impact on Sales")
    plt.xlabel("Day Type")
    plt.ylabel("Average Units Sold")

    save_graph(
        "07_holiday_impact.png",
        output_dir
    )


def sales_distribution(df, output_dir):
    """Plot distribution of units sold."""

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["Units_Sold"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Distribution of Units Sold")
    plt.xlabel("Units Sold")
    plt.ylabel("Frequency")

    save_graph(
        "08_sales_distribution.png",
        output_dir
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

    output_dir = (
        project_root
        / "outputs"
        / "graphs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = load_data(input_path)

    # -----------------------------------------------------
    # GENERATE EDA GRAPHS
    # -----------------------------------------------------

    print("\nGenerating EDA graphs...\n")

    daily_sales_trend(df, output_dir)

    revenue_trend(df, output_dir)

    monthly_sales_pattern(df, output_dir)

    product_wise_sales(df, output_dir)

    store_wise_sales(df, output_dir)

    promotion_impact(df, output_dir)

    holiday_impact(df, output_dir)

    sales_distribution(df, output_dir)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(
        f"\nAll graphs saved to:\n{output_dir}"
    )


if __name__ == "__main__":

    main()