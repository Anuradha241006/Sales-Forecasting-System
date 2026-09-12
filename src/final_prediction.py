import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM
# FINAL PREDICTION MODULE
# =========================================================


def main():

    print("\n" + "=" * 60)
    print("FINAL SALES PREDICTION MODULE")
    print("=" * 60)


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
    # FILE PATHS
    # -----------------------------------------------------

    model_path = (

        project_root
        / "models"
        / "best_random_forest_model.pkl"

    )


    test_data_path = (

        project_root
        / "data"
        / "processed"
        / "test_data.csv"

    )


    output_folder = (

        project_root
        / "outputs"
        / "predictions"

    )


    graph_folder = (

        project_root
        / "outputs"
        / "graphs"

    )


    output_folder.mkdir(

        parents=True,
        exist_ok=True

    )


    graph_folder.mkdir(

        parents=True,
        exist_ok=True

    )


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("\nLoading best trained model...")

    model = joblib.load(

        model_path

    )

    print("Model loaded successfully!")


    # -----------------------------------------------------
    # LOAD TEST DATA
    # -----------------------------------------------------

    print("\nLoading test dataset...")

    df = pd.read_csv(

        test_data_path

    )


    df["Date"] = pd.to_datetime(

        df["Date"]

    )


    print(

        f"Test dataset shape: {df.shape}"

    )


    # -----------------------------------------------------
    # SELECT PRODUCT AND STORE
    # -----------------------------------------------------

    print("\nAvailable Products:")

    print(

        sorted(

            df["Product_ID"]

            .unique()

        )

    )


    print("\nAvailable Stores:")

    print(

        sorted(

            df["Store_ID"]

            .unique()

        )

    )


    # Default selection

    selected_product = "P001"

    selected_store = "S001"


    print(

        f"\nSelected Product: {selected_product}"

    )

    print(

        f"Selected Store: {selected_store}"

    )


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    filtered_df = df[

        (

            df["Product_ID"]

            == selected_product

        )

        &

        (

            df["Store_ID"]

            == selected_store

        )

    ].copy()


    filtered_df = (

        filtered_df

        .sort_values(

            "Date"

        )

        .reset_index(

            drop=True

        )

    )


    print(

        f"\nFiltered records: {len(filtered_df)}"

    )


    # -----------------------------------------------------
    # DEFINE FEATURES
    # -----------------------------------------------------

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


    X = (

        filtered_df

        [

            feature_columns

        ]

    )


    y_actual = (

        filtered_df

        [

            "Units_Sold"

        ]

    )


    # -----------------------------------------------------
    # GENERATE PREDICTIONS
    # -----------------------------------------------------

    print("\nGenerating predictions...")


    predictions = (

        model.predict(

            X

        )

    )


    # -----------------------------------------------------
    # CREATE RESULTS DATAFRAME
    # -----------------------------------------------------

    results_df = pd.DataFrame(

        {

            "Date":

                filtered_df["Date"],

            "Product_ID":

                filtered_df["Product_ID"],

            "Store_ID":

                filtered_df["Store_ID"],

            "Actual_Units_Sold":

                y_actual,

            "Predicted_Units_Sold":

                predictions

        }

    )


    results_df[

        "Prediction_Error"

    ] = (

        results_df[

            "Actual_Units_Sold"

        ]

        -

        results_df[

            "Predicted_Units_Sold"

        ]

    )


    results_df[

        "Absolute_Error"

    ] = (

        results_df[

            "Prediction_Error"

        ]

        .abs()

    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results_path = (

        output_folder

        / "final_predictions.csv"

    )


    results_df.to_csv(

        results_path,

        index=False

    )


    print(

        f"\nPredictions saved to:\n{results_path}"

    )


    # -----------------------------------------------------
    # DISPLAY SAMPLE RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("SAMPLE PREDICTIONS")

    print("=" * 60)


    print(

        results_df

        .head(10)

        .to_string(

            index=False

        )

    )


    # -----------------------------------------------------
    # CREATE VISUALIZATION
    # -----------------------------------------------------

    plt.figure(

        figsize=(12, 6)

    )


    plt.plot(

        results_df["Date"],

        results_df["Actual_Units_Sold"],

        label="Actual Sales"

    )


    plt.plot(

        results_df["Date"],

        results_df["Predicted_Units_Sold"],

        label="Predicted Sales"

    )


    plt.xlabel(

        "Date"

    )


    plt.ylabel(

        "Units Sold"

    )


    plt.title(

        f"Actual vs Predicted Sales - "

        f"{selected_product} - "

        f"{selected_store}"

    )


    plt.legend()

    plt.grid()

    plt.tight_layout()


    graph_path = (

        graph_folder

        / "actual_vs_predicted.png"

    )


    plt.savefig(

        graph_path,

        dpi=300

    )


    plt.close()


    print(

        f"\nGraph saved to:\n{graph_path}"

    )


    # -----------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("FINAL PREDICTION COMPLETED SUCCESSFULLY!")

    print("=" * 60)


if __name__ == "__main__":

    main()