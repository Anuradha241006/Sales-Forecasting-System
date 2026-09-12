import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM
# INTERACTIVE SALES FORECASTING MODULE
# =========================================================


def get_user_choice(prompt, valid_options):

    while True:

        choice = input(prompt).strip().upper()

        if choice in valid_options:

            return choice

        print(
            f"Invalid choice. Please choose from: "
            f"{', '.join(valid_options)}"
        )


def get_forecast_days():

    valid_days = [7, 14, 30]

    while True:

        try:

            days = int(

                input(
                    "\nEnter forecast duration "
                    "(7 / 14 / 30 days): "
                )

            )

            if days in valid_days:

                return days

            print(
                "Please enter 7, 14, or 30."
            )

        except ValueError:

            print(
                "Invalid input. Please enter a number."
            )


def get_promotion_status():

    while True:

        choice = input(
            "\nWill there be a promotion? "
            "(Yes/No): "
        ).strip().lower()

        if choice in ["yes", "y"]:

            return 1

        elif choice in ["no", "n"]:

            return 0

        else:

            print(
                "Please enter Yes or No."
            )


def main():

    print("\n" + "=" * 60)
    print("INTERACTIVE SALES FORECASTING SYSTEM")
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

    print("\nLoading trained model...")

    model = joblib.load(
        model_path
    )

    print("Model loaded successfully!")


    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nLoading historical data...")

    df = pd.read_csv(
        data_path
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    products = sorted(
        df["Product_ID"].unique()
    )


    stores = sorted(
        df["Store_ID"].unique()
    )


    # -----------------------------------------------------
    # USER INPUT
    # -----------------------------------------------------

    print("\nAvailable Products:")

    print(", ".join(products))


    selected_product = get_user_choice(

        "\nSelect Product ID: ",

        products

    )


    print("\nAvailable Stores:")

    print(", ".join(stores))


    selected_store = get_user_choice(

        "\nSelect Store ID: ",

        stores

    )


    forecast_days = get_forecast_days()


    promotion = get_promotion_status()


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    history = df[

        (
            df["Product_ID"]
            ==
            selected_product
        )

        &

        (
            df["Store_ID"]
            ==
            selected_store
        )

    ].copy()


    history = history.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


    sales_history = list(
        history["Units_Sold"]
    )


    last_date = history[
        "Date"
    ].max()


    print("\n" + "=" * 60)

    print("FORECAST CONFIGURATION")

    print("=" * 60)

    print(
        f"Product: {selected_product}"
    )

    print(
        f"Store: {selected_store}"
    )

    print(
        f"Forecast Days: {forecast_days}"
    )

    print(
        f"Promotion: "
        f"{'Yes' if promotion == 1 else 'No'}"
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


    future_predictions = []


    # -----------------------------------------------------
    # FORECAST LOOP
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


        # Date features

        year = future_date.year

        month = future_date.month

        day = future_date.day

        day_of_week = (
            future_date.dayofweek
        )

        quarter = (
            future_date.quarter
        )


        # Lag features

        lag_1 = sales_history[-1]

        lag_7 = sales_history[-7]

        lag_14 = sales_history[-14]

        lag_30 = sales_history[-30]


        # Rolling means

        rolling_mean_7 = np.mean(
            sales_history[-7:]
        )

        rolling_mean_30 = np.mean(
            sales_history[-30:]
        )


        # -------------------------------------------------
        # HOLIDAY
        # -------------------------------------------------

        holiday = 0


        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        predicted_sales = (

            model.predict(
                future_input
            )[0]

        )


        predicted_sales = max(
            0,
            predicted_sales
        )


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


        # Add prediction to history

        sales_history.append(
            predicted_sales
        )


    # -----------------------------------------------------
    # RESULTS DATAFRAME
    # -----------------------------------------------------

    forecast_df = pd.DataFrame(
        future_predictions
    )


    forecast_df[
        "Predicted_Units_Sold"
    ] = (

        forecast_df[
            "Predicted_Units_Sold"
        ]

        .round()

        .astype(
            int
        )

    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    file_name = (

        f"forecast_"

        f"{selected_product}_"

        f"{selected_store}_"

        f"{forecast_days}days.csv"

    )


    forecast_path = (
        output_folder
        /
        file_name
    )


    forecast_df.to_csv(

        forecast_path,

        index=False

    )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print("SALES FORECAST RESULTS")

    print("=" * 60)


    print(

        forecast_df.to_string(

            index=False

        )

    )


    # -----------------------------------------------------
    # SUMMARY
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

        f"Average Daily Sales: "

        f"{forecast_df['Predicted_Units_Sold'].mean():.2f}"

    )


    print(

        f"Maximum Sales: "

        f"{forecast_df['Predicted_Units_Sold'].max()}"

    )


    print(

        f"Minimum Sales: "

        f"{forecast_df['Predicted_Units_Sold'].min()}"

    )


    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )


    plt.plot(

        history["Date"].tail(60),

        history["Units_Sold"].tail(60),

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

        f"Sales Forecast - "

        f"{selected_product} - "

        f"{selected_store}"

    )


    plt.legend()

    plt.grid()

    plt.tight_layout()


    graph_path = (

        graph_folder

        /

        f"forecast_"

        f"{selected_product}_"

        f"{selected_store}_"

        f"{forecast_days}days.png"

    )


    plt.savefig(

        graph_path,

        dpi=300

    )


    plt.close()


    print(

        f"\nForecast saved to:\n"

        f"{forecast_path}"

    )


    print(

        f"\nForecast graph saved to:\n"

        f"{graph_path}"

    )


    print("\n" + "=" * 60)

    print(
        "INTERACTIVE FORECASTING "
        "COMPLETED SUCCESSFULLY!"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()