
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px


# -------------------------------------------------
# FLASK APPLICATION
# -------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_random_forest_model.pkl"

DATA_PATH = BASE_DIR / "data" / "processed" / "sales_features.csv"

METRICS_PATH = (
    BASE_DIR /
    "outputs" /
    "metrics" /
    "model_comparison.csv"
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# -------------------------------------------------
# HOME PAGE / DASHBOARD
# -------------------------------------------------

@app.route("/")
def home():

    model_metrics = {}

    if METRICS_PATH.exists():

        metrics_df = pd.read_csv(METRICS_PATH)

        best_model = metrics_df.iloc[0]

        model_metrics = {
            "model": best_model["Model"],
            "mae": round(best_model["MAE"], 2),
            "rmse": round(best_model["RMSE"], 2),
            "r2_score": round(best_model["R2_Score"], 4)
        }

    return render_template(
        "index.html",
        model_metrics=model_metrics
    )


# -------------------------------------------------
# FORECAST ROUTE
# -------------------------------------------------

@app.route("/forecast", methods=["POST"])
def forecast():

    # -------------------------------------------------
    # GET USER INPUTS
    # -------------------------------------------------

    product_id = request.form.get("product_id")
    store_id = request.form.get("store_id")

    forecast_days = int(
        request.form.get("forecast_days")
    )

    promotion = int(
        request.form.get("promotion")
    )

    print("\nForecast Request Received")
    print("Product:", product_id)
    print("Store:", store_id)
    print("Days:", forecast_days)
    print("Promotion:", promotion)


    # -------------------------------------------------
    # LOAD HISTORICAL DATA
    # -------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])


    # -------------------------------------------------
    # FILTER SELECTED PRODUCT AND STORE
    # -------------------------------------------------

    filtered_df = df[
        (df["Product_ID"] == product_id) &
        (df["Store_ID"] == store_id)
    ].copy()


    # Check whether data exists
    if filtered_df.empty:

        return "No historical data found for the selected Product and Store."


    # Sort data by date
    filtered_df = filtered_df.sort_values("Date")

    # Historical data
    history = filtered_df.copy()

    # Last historical date
    last_date = history["Date"].max()


    # Store predictions
    forecasts = []


    # -------------------------------------------------
    # RECURSIVE FUTURE FORECASTING
    # -------------------------------------------------

    for i in range(1, forecast_days + 1):

        # Create future date
        future_date = last_date + pd.Timedelta(days=i)


        # Get sales history
        sales_history = history["Units_Sold"].tolist()


        # -------------------------------------------------
        # CREATE LAG FEATURES
        # -------------------------------------------------

        lag_1 = sales_history[-1]

        lag_7 = (
            sales_history[-7]
            if len(sales_history) >= 7
            else lag_1
        )

        lag_14 = (
            sales_history[-14]
            if len(sales_history) >= 14
            else lag_1
        )

        lag_30 = (
            sales_history[-30]
            if len(sales_history) >= 30
            else lag_1
        )


        # -------------------------------------------------
        # CREATE ROLLING MEAN FEATURES
        # -------------------------------------------------

        rolling_mean_7 = (
            sum(sales_history[-7:]) /
            min(7, len(sales_history))
        )

        rolling_mean_30 = (
            sum(sales_history[-30:]) /
            min(30, len(sales_history))
        )


        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame([{
            "Product_ID": product_id,
            "Store_ID": store_id,
            "Promotion": promotion,
            "Holiday": 0,
            "Year": future_date.year,
            "Month": future_date.month,
            "Day": future_date.day,
            "Day_of_Week": future_date.dayofweek,
            "Quarter": future_date.quarter,
            "Lag_1": lag_1,
            "Lag_7": lag_7,
            "Lag_14": lag_14,
            "Lag_30": lag_30,
            "Rolling_Mean_7": rolling_mean_7,
            "Rolling_Mean_30": rolling_mean_30
        }])


        # -------------------------------------------------
        # PREDICT SALES
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]


        # Prevent negative predictions
        prediction = max(
            0,
            round(prediction)
        )


        # -------------------------------------------------
        # SAVE FORECAST
        # -------------------------------------------------

        forecasts.append({
            "Date": future_date.strftime("%Y-%m-%d"),
            "Predicted_Units_Sold": prediction
        })


        # -------------------------------------------------
        # ADD PREDICTION TO HISTORY
        # -------------------------------------------------

        # This allows recursive forecasting.
        # Today's predicted value becomes part of
        # tomorrow's lag features.

        new_row = pd.DataFrame([{
            "Date": future_date,
            "Product_ID": product_id,
            "Store_ID": store_id,
            "Units_Sold": prediction
        }])

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )


    # -------------------------------------------------
    # CREATE FORECAST DATAFRAME
    # -------------------------------------------------

    forecast_df = pd.DataFrame(forecasts)


    # -------------------------------------------------
    # CREATE INTERACTIVE PLOTLY GRAPH
    # -------------------------------------------------

    fig = px.line(
        forecast_df,
        x="Date",
        y="Predicted_Units_Sold",
        markers=True,
        title="Future Sales Forecast"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Units Sold",
        height=500
    )


    # Convert Plotly graph into HTML
    graph_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )


    # -------------------------------------------------
    # FORECAST SUMMARY
    # -------------------------------------------------

    total_sales = int(
        forecast_df[
            "Predicted_Units_Sold"
        ].sum()
    )

    average_sales = round(
        forecast_df[
            "Predicted_Units_Sold"
        ].mean(),
        2
    )

    max_sales = int(
        forecast_df[
            "Predicted_Units_Sold"
        ].max()
    )

    min_sales = int(
        forecast_df[
            "Predicted_Units_Sold"
        ].min()
    )


    # -------------------------------------------------
    # SEND DATA TO RESULT PAGE
    # -------------------------------------------------

    return render_template(
        "result.html",

        product_id=product_id,
        store_id=store_id,

        forecast_days=forecast_days,
        promotion=promotion,

        forecasts=forecast_df.to_dict(
            orient="records"
        ),

        total_sales=total_sales,
        average_sales=average_sales,
        max_sales=max_sales,
        min_sales=min_sales,

        graph_html=graph_html
    )


# -------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )

