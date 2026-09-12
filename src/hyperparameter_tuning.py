import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# SALES FORECASTING SYSTEM
# RANDOM FOREST HYPERPARAMETER TUNING
# =========================================================


def load_data(file_path):

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    print("Dataset loaded successfully!")

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


def prepare_features(df):

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


    X = df[
        feature_columns
    ].copy()


    y = df[
        "Units_Sold"
    ].copy()


    return X, y


def build_pipeline():

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


    random_forest = RandomForestRegressor(

        random_state=42,

        n_jobs=-1

    )


    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),

            (

                "model",

                random_forest

            )

        ]

    )


    return pipeline


def main():

    # =====================================================
    # PROJECT ROOT
    # =====================================================

    project_root = (

        Path(__file__)

        .resolve()

        .parent

        .parent

    )


    # =====================================================
    # INPUT DATA
    # =====================================================

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


    # =====================================================
    # OUTPUT PATHS
    # =====================================================

    model_path = (

        project_root

        / "models"

        / "best_random_forest_model.pkl"

    )


    prediction_path = (

        project_root

        / "outputs"

        / "predictions"

        / "tuned_random_forest_predictions.csv"

    )


    metrics_path = (

        project_root

        / "outputs"

        / "metrics"

        / "tuned_random_forest_metrics.csv"

    )


    # =====================================================
    # START
    # =====================================================

    print("\n" + "=" * 60)

    print(
        "RANDOM FOREST HYPERPARAMETER TUNING"
    )

    print("=" * 60)


    # =====================================================
    # LOAD DATA
    # =====================================================

    train_df = load_data(
        train_path
    )


    test_df = load_data(
        test_path
    )


    # =====================================================
    # FEATURES
    # =====================================================

    X_train, y_train = prepare_features(
        train_df
    )


    X_test, y_test = prepare_features(
        test_df
    )


    print(
        f"\nTraining samples: {len(X_train)}"
    )


    print(
        f"Testing samples: {len(X_test)}"
    )


    # =====================================================
    # BUILD PIPELINE
    # =====================================================

    pipeline = build_pipeline()


    # =====================================================
    # TIME SERIES CROSS VALIDATION
    # =====================================================

    tscv = TimeSeriesSplit(

        n_splits=4

    )


    # =====================================================
    # PARAMETER SEARCH SPACE
    # =====================================================

    param_distributions = {

        "model__n_estimators": [

            200,

            300,

            400

        ],

        "model__max_depth": [

            None,

            10,

            20,

            30

        ],

        "model__min_samples_split": [

            2,

            5,

            10

        ],

        "model__min_samples_leaf": [

            1,

            2,

            4

        ],

        "model__max_features": [

            0.7,

            1.0

        ]

    }


    # =====================================================
    # RANDOMIZED SEARCH
    # =====================================================

    random_search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_distributions,

        n_iter=12,

        scoring="neg_root_mean_squared_error",

        cv=tscv,

        verbose=2,

        random_state=42,

        n_jobs=-1,

        refit=True

    )


    # =====================================================
    # TRAIN
    # =====================================================

    print(

        "\nStarting hyperparameter tuning..."

    )


    print(

        "This may take a few minutes..."

    )


    random_search.fit(

        X_train,

        y_train

    )


    # =====================================================
    # BEST PARAMETERS
    # =====================================================

    print("\n" + "=" * 60)

    print(
        "BEST PARAMETERS"
    )

    print("=" * 60)


    for parameter, value in (

        random_search.best_params_.items()

    ):

        print(

            f"{parameter}: {value}"

        )


    # =====================================================
    # BEST MODEL
    # =====================================================

    best_model = (

        random_search.best_estimator_

    )


    # =====================================================
    # PREDICTIONS
    # =====================================================

    predictions = best_model.predict(

        X_test

    )


    # =====================================================
    # EVALUATION
    # =====================================================

    mae = mean_absolute_error(

        y_test,

        predictions

    )


    mse = mean_squared_error(

        y_test,

        predictions

    )


    rmse = np.sqrt(

        mse

    )


    r2 = r2_score(

        y_test,

        predictions

    )


    print("\n" + "=" * 60)

    print(
        "TUNED RANDOM FOREST PERFORMANCE"
    )

    print("=" * 60)


    print(
        f"MAE      : {mae:.4f}"
    )

    print(
        f"MSE      : {mse:.4f}"
    )

    print(
        f"RMSE     : {rmse:.4f}"
    )

    print(
        f"R² Score : {r2:.4f}"
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    joblib.dump(

        best_model,

        model_path

    )


    print(

        f"\nBest model saved to:\n{model_path}"

    )


    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    prediction_path.parent.mkdir(

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

            "Units_Sold":

            "Actual_Units_Sold"

        },

        inplace=True

    )


    results[

        "Predicted_Units_Sold"

    ] = predictions


    results.to_csv(

        prediction_path,

        index=False

    )


    print(

        f"\nPredictions saved to:\n{prediction_path}"

    )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_path.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    metrics_df = pd.DataFrame(

        {

            "Model": [

                "Tuned Random Forest"

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

        metrics_path,

        index=False

    )


    print(

        f"\nMetrics saved to:\n{metrics_path}"

    )


    print("\n" + "=" * 60)

    print(
        "HYPERPARAMETER TUNING COMPLETED!"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()