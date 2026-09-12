import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# SALES FORECASTING SYSTEM - RANDOM FOREST REGRESSOR
# =========================================================


def load_data(file_path):
    """Load dataset."""

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df


def prepare_features(df):
    """Separate input features and target."""

    feature_columns = [
        "Product_ID",
        "Store_ID",
        "Promotion",
        "Holiday",
        "Year",
        "Month",
        "Day",
        "Day_of_Week",
        "Quarter",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30"
    ]

    X = df[feature_columns].copy()

    y = df["Units_Sold"].copy()

    return X, y


def build_model():
    """Create Random Forest pipeline."""

    categorical_features = [
        "Product_ID",
        "Store_ID"
    ]

    numeric_features = [
        "Promotion",
        "Holiday",
        "Year",
        "Month",
        "Day",
        "Day_of_Week",
        "Quarter",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30"
    ]


    # ---------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )


    # ---------------------------------------------
    # RANDOM FOREST MODEL
    # ---------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )


    # ---------------------------------------------
    # PIPELINE
    # ---------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def evaluate_model(y_true, y_pred):
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

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, mse, rmse, r2


def save_predictions(test_df, predictions, output_path):
    """Save predictions."""

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
    ] = predictions

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nPredictions saved to:\n{output_path}"
    )


def save_metrics(mae, mse, rmse, r2, output_path):
    """Save model metrics."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        {
            "Model": [
                "Random Forest"
            ],
            "MAE": [
                mae
            ],
            "MSE": [
                mse
            ],
            "RMSE": [
                rmse
            ],
            "R2_Score": [
                r2
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
    # PROJECT ROOT
    # -----------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )


    # -----------------------------------------------------
    # INPUT PATHS
    # -----------------------------------------------------

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
    # OUTPUT PATHS
    # -----------------------------------------------------

    model_path = (
        project_root
        / "models"
        / "random_forest_model.pkl"
    )

    prediction_path = (
        project_root
        / "outputs"
        / "predictions"
        / "random_forest_predictions.csv"
    )

    metrics_path = (
        project_root
        / "outputs"
        / "metrics"
        / "random_forest_metrics.csv"
    )


    # -----------------------------------------------------
    # START
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL TRAINING")
    print("=" * 60)


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    train_df = load_data(train_path)

    test_df = load_data(test_path)


    # -----------------------------------------------------
    # PREPARE FEATURES
    # -----------------------------------------------------

    X_train, y_train = prepare_features(
        train_df
    )

    X_test, y_test = prepare_features(
        test_df
    )

    print(
        f"\nTraining feature shape: {X_train.shape}"
    )

    print(
        f"Testing feature shape: {X_test.shape}"
    )


    # -----------------------------------------------------
    # BUILD MODEL
    # -----------------------------------------------------

    print("\nBuilding Random Forest pipeline...")

    model = build_model()


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("Training Random Forest model...")
    print("Please wait, this may take a moment...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed!")


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    mae, mse, rmse, r2 = evaluate_model(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST PERFORMANCE")
    print("=" * 60)

    print(f"MAE      : {mae:.4f}")

    print(f"MSE      : {mse:.4f}")

    print(f"RMSE     : {rmse:.4f}")

    print(f"R² Score : {r2:.4f}")


    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    save_predictions(
        test_df,
        predictions,
        prediction_path
    )

    save_metrics(
        mae,
        mse,
        rmse,
        r2,
        metrics_path
    )


    print("\n" + "=" * 60)
    print("RANDOM FOREST MODEL COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":

    main()