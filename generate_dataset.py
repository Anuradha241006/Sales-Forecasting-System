import pandas as pd
import numpy as np
from pathlib import Path

# Reproducible results
np.random.seed(42)

# Create date range: 3 years of historical daily data
dates = pd.date_range(
    start="2023-01-01",
    end="2025-12-31",
    freq="D"
)

# Products and stores
products = ["P001", "P002", "P003", "P004", "P005"]
stores = ["S001", "S002", "S003"]

data = []

# Base demand for each product
product_base_sales = {
    "P001": 120,
    "P002": 150,
    "P003": 100,
    "P004": 180,
    "P005": 130
}

# Base price for each product
product_prices = {
    "P001": 200,
    "P002": 250,
    "P003": 150,
    "P004": 300,
    "P005": 180
}

# Store demand multiplier
store_multiplier = {
    "S001": 1.00,
    "S002": 1.15,
    "S003": 0.90
}

for date in dates:

    # Yearly trend
    days_from_start = (date - dates[0]).days
    trend = 1 + (days_from_start * 0.00015)

    # Monthly seasonality
    month = date.month
    seasonal_effect = 1 + 0.15 * np.sin(
        2 * np.pi * month / 12
    )

    # Weekend effect
    if date.weekday() >= 5:
        weekend_effect = 1.10
    else:
        weekend_effect = 1.00

    for product in products:
        for store in stores:

            # Random promotion
            promotion = np.random.choice(
                [0, 1],
                p=[0.75, 0.25]
            )

            # Promotion increases demand
            promotion_effect = 1.20 if promotion == 1 else 1.00

            # Simple holiday indicator
            holiday_dates = [
                "01-01",
                "01-26",
                "08-15",
                "10-02",
                "12-25"
            ]

            holiday = 1 if date.strftime("%m-%d") in holiday_dates else 0

            # Holiday sales effect
            holiday_effect = 1.25 if holiday == 1 else 1.00

            # Random market noise
            noise = np.random.normal(1, 0.08)

            # Calculate units sold
            units_sold = (
                product_base_sales[product]
                * store_multiplier[store]
                * trend
                * seasonal_effect
                * weekend_effect
                * promotion_effect
                * holiday_effect
                * noise
            )

            units_sold = max(1, round(units_sold))

            # Revenue
            revenue = units_sold * product_prices[product]

            data.append([
                date,
                product,
                store,
                units_sold,
                revenue,
                promotion,
                holiday
            ])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[
        "Date",
        "Product_ID",
        "Store_ID",
        "Units_Sold",
        "Revenue",
        "Promotion",
        "Holiday"
    ]
)

# Create output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Save dataset
output_file = output_dir / "sales_data.csv"

df.to_csv(output_file, index=False)

print("Dataset created successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Saved to: {output_file}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())