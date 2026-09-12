import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# SALES FORECASTING SYSTEM - MOVING AVERAGE BASELINE MODEL
# =========================================================


def load_data(file_path):
    """Load testing dataset."""

    print("\nLoading testing dataset...")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    print("Dataset loaded successfully!")
    print(f"Testing dataset shape: {df.shape}")

    return df


def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    return mae, mse, rmse


def evaluate_baseline(test_df):
    """
    Evaluate the 7-day moving average baseline.

    Rolling_Mean_7 was created using only
    previous sales data, preventing data leakage.
    """

    print("\n" + "=" * 60)
    print("MOVING AVERAGE BASELINE MODEL")
    print("=" * 60)

    y_true = test_df[
        "Units_Sold"
    ]

    y_pred = test_df[
        "Rolling_Mean_7"
    ]

    mae, mse, rmse = calculate_metrics(
        y_true,
        y_pred
    )

    print("\nMODEL PERFORMANCE")

    print(f"MAE  : {mae:.4f}")

    print(f"MSE  : {mse:.4f}")

    print(f"RMSE : {rmse:.4f}")

    return y_pred, mae, mse, rmse


def save_predictions(test_df, predictions, output_path):
    """Save actual vs predicted values."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = test_df[
        [
            "Date",
            "Product_ID",
            "Store_ID",
            "Units_Sold"
        ]
    ].copy()

    results.rename(
        columns={
            "Units_Sold": "Actual_Units_Sold"
        },
        inplace=True
    )

    results[
        "Predicted_Units_Sold"
    ] = predictions.values

    results.to_csv(
        output_path,
        index=False
    )

    print("\nPredictions saved successfully!")

    print(
        f"Saved to:\n{output_path}"
    )

    return results


def save_metrics(mae, mse, rmse, output_path):
    """Save evaluation metrics."""

    metrics_df = pd.DataFrame(
        {
            "Model": [
                "Moving Average Baseline"
            ],
            "MAE": [
                mae
            ],
            "MSE": [
                mse
            ],
            "RMSE": [
                rmse
            ]
        }
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nMetrics saved to:\n{output_path}"
    )


def main():

    # -----------------------------------------------------
    # PROJECT PATH
    # -----------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )


    # -----------------------------------------------------
    # FILE PATHS
    # -----------------------------------------------------

    test_path = (
        project_root
        / "data"
        / "processed"
        / "test_data.csv"
    )

    prediction_path = (
        project_root
        / "outputs"
        / "predictions"
        / "baseline_predictions.csv"
    )

    metrics_path = (
        project_root
        / "outputs"
        / "metrics"
        / "baseline_metrics.csv"
    )


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    test_df = load_data(
        test_path
    )


    # -----------------------------------------------------
    # EVALUATE BASELINE
    # -----------------------------------------------------

    predictions, mae, mse, rmse = evaluate_baseline(
        test_df
    )


    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    save_predictions(
        test_df,
        predictions,
        prediction_path
    )


    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_metrics(
        mae,
        mse,
        rmse,
        metrics_path
    )


    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":

    main()