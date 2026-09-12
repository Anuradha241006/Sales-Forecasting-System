import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM
# TRUE FUTURE SALES FORECASTING
# =========================================================


def main():

    print("\n" + "=" * 60)
    print("TRUE FUTURE SALES FORECASTING")
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
    # PATHS
    # -----------------------------------------------------

    model_path = (
        project_root
        / "models"
        / "best_random_forest_model.pkl"
    )


    data_path = (
        project_root
        / "data"
        / "processed"
        / "sales_features.csv"
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
    # LOAD HISTORICAL DATA
    # -----------------------------------------------------

    print("\nLoading historical data...")

    df = pd.read_csv(
        data_path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    print(
        f"Historical dataset shape: {df.shape}"
    )


    # -----------------------------------------------------
    # SELECT PRODUCT AND STORE
    # -----------------------------------------------------

    selected_product = "P001"

    selected_store = "S001"


    print(
        f"\nSelected Product: {selected_product}"
    )

    print(
        f"Selected Store: {selected_store}"
    )


    # -----------------------------------------------------
    # FILTER PRODUCT + STORE DATA
    # -----------------------------------------------------

    history = df[
        (df["Product_ID"] == selected_product)
        &
        (df["Store_ID"] == selected_store)
    ].copy()


    history = history.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    print(
        f"\nHistorical records: {len(history)}"
    )


    print(
        f"Latest historical date: "
        f"{history['Date'].max().date()}"
    )


    # -----------------------------------------------------
    # FORECAST SETTINGS
    # -----------------------------------------------------

    forecast_days = 30


    print(
        f"\nForecasting next "
        f"{forecast_days} days..."
    )


    # -----------------------------------------------------
    # FEATURE COLUMNS
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


    # -----------------------------------------------------
    # SALES HISTORY
    # -----------------------------------------------------

    sales_history = list(
        history["Units_Sold"]
    )


    last_date = history[
        "Date"
    ].max()


    future_predictions = []


    # -----------------------------------------------------
    # FUTURE FORECAST LOOP
    # -----------------------------------------------------

    for i in range(
        1,
        forecast_days + 1
    ):


        future_date = (
            last_date
            +
            pd.Timedelta(
                days=i
            )
        )


        # ---------------------------------------------
        # FUTURE DATE FEATURES
        # ---------------------------------------------

        year = future_date.year

        month = future_date.month

        day = future_date.day

        day_of_week = (
            future_date.dayofweek
        )

        quarter = (
            future_date.quarter
        )


        # ---------------------------------------------
        # LAG FEATURES
        # ---------------------------------------------

        lag_1 = (
            sales_history[-1]
        )

        lag_7 = (
            sales_history[-7]
        )

        lag_14 = (
            sales_history[-14]
        )

        lag_30 = (
            sales_history[-30]
        )


        # ---------------------------------------------
        # ROLLING FEATURES
        # ---------------------------------------------

        rolling_mean_7 = np.mean(
            sales_history[-7:]
        )

        rolling_mean_30 = np.mean(
            sales_history[-30:]
        )


        # ---------------------------------------------
        # PROMOTION / HOLIDAY
        # ---------------------------------------------
        #
        # IMPORTANT:
        # For future forecasting we do not know
        # actual promotion or holiday information.
        #
        # We use a reasonable default.
        # ---------------------------------------------

        promotion = 0

        holiday = 0


        # ---------------------------------------------
        # CREATE FUTURE INPUT
        # ---------------------------------------------

        future_input = pd.DataFrame(
            {

                "Product_ID":
                    [selected_product],

                "Store_ID":
                    [selected_store],

                "Promotion":
                    [promotion],

                "Holiday":
                    [holiday],

                "Year":
                    [year],

                "Month":
                    [month],

                "Day":
                    [day],

                "Day_of_Week":
                    [day_of_week],

                "Quarter":
                    [quarter],

                "Lag_1":
                    [lag_1],

                "Lag_7":
                    [lag_7],

                "Lag_14":
                    [lag_14],

                "Lag_30":
                    [lag_30],

                "Rolling_Mean_7":
                    [
                        rolling_mean_7
                    ],

                "Rolling_Mean_30":
                    [
                        rolling_mean_30
                    ]

            }
        )


        # ---------------------------------------------
        # GENERATE PREDICTION
        # ---------------------------------------------

        predicted_sales = (

            model.predict(
                future_input
            )[0]

        )


        # ---------------------------------------------
        # PREVENT NEGATIVE SALES
        # ---------------------------------------------

        predicted_sales = max(
            0,
            predicted_sales
        )


        # ---------------------------------------------
        # STORE RESULT
        # ---------------------------------------------

        future_predictions.append(

            {

                "Date":
                    future_date,

                "Product_ID":
                    selected_product,

                "Store_ID":
                    selected_store,

                "Predicted_Units_Sold":
                    predicted_sales,

                "Promotion":
                    promotion,

                "Holiday":
                    holiday

            }

        )


        # ---------------------------------------------
        # ADD PREDICTION TO HISTORY
        # ---------------------------------------------

        sales_history.append(
            predicted_sales
        )


    # -----------------------------------------------------
    # CREATE RESULTS DATAFRAME
    # -----------------------------------------------------

    forecast_df = pd.DataFrame(
        future_predictions
    )


    # -----------------------------------------------------
    # ROUND PREDICTIONS
    # -----------------------------------------------------

    forecast_df[
        "Predicted_Units_Sold"
    ] = (

        forecast_df[
            "Predicted_Units_Sold"
        ]

        .round(

            0

        )

        .astype(

            int

        )

    )


    # -----------------------------------------------------
    # SAVE FORECAST
    # -----------------------------------------------------

    forecast_path = (

        output_folder

        / "future_sales_forecast.csv"

    )


    forecast_df.to_csv(

        forecast_path,

        index=False

    )


    print(

        f"\nFuture forecast saved to:\n"
        f"{forecast_path}"

    )


    # -----------------------------------------------------
    # DISPLAY FORECAST
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("FUTURE SALES FORECAST")

    print("=" * 60)


    print(

        forecast_df.to_string(

            index=False

        )

    )


    # -----------------------------------------------------
    # CREATE FORECAST GRAPH
    # -----------------------------------------------------

    plt.figure(

        figsize=(12, 6)

    )


    plt.plot(

        history["Date"].tail(60),

        history[
            "Units_Sold"
        ].tail(60),

        label="Historical Sales"

    )


    plt.plot(

        forecast_df["Date"],

        forecast_df[
            "Predicted_Units_Sold"
        ],

        label="Forecasted Sales"

    )


    plt.axvline(

        x=last_date,

        linestyle="--",

        label="Forecast Start"

    )


    plt.xlabel(

        "Date"

    )


    plt.ylabel(

        "Units Sold"

    )


    plt.title(

        f"30-Day Sales Forecast - "
        f"{selected_product} - "
        f"{selected_store}"

    )


    plt.legend()

    plt.grid()

    plt.tight_layout()


    graph_path = (

        graph_folder

        / "future_sales_forecast.png"

    )


    plt.savefig(

        graph_path,

        dpi=300

    )


    plt.close()


    print(

        f"\nForecast graph saved to:\n"
        f"{graph_path}"

    )


    # -----------------------------------------------------
    # FORECAST SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("FORECAST SUMMARY")

    print("=" * 60)


    print(

        f"Forecast Period: "

        f"{forecast_df['Date'].min().date()} "

        f"to "

        f"{forecast_df['Date'].max().date()}"

    )


    print(

        f"Total Forecasted Sales: "

        f"{forecast_df['Predicted_Units_Sold'].sum()}"

    )


    print(

        f"Average Daily Forecast: "

        f"{forecast_df['Predicted_Units_Sold'].mean():.2f}"

    )


    print(

        f"Maximum Predicted Sales: "

        f"{forecast_df['Predicted_Units_Sold'].max()}"

    )


    print(

        f"Minimum Predicted Sales: "

        f"{forecast_df['Predicted_Units_Sold'].min()}"

    )


    print("\n" + "=" * 60)

    print("TRUE FUTURE FORECASTING COMPLETED SUCCESSFULLY!")

    print("=" * 60)


if __name__ == "__main__":

    main()