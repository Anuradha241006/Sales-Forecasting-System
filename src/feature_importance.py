import pandas as pd
import joblib

from pathlib import Path
import matplotlib.pyplot as plt


# =========================================================
# SALES FORECASTING SYSTEM - FEATURE IMPORTANCE ANALYSIS
# =========================================================


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
    # MODEL PATH
    # -----------------------------------------------------

    model_path = (
        project_root
        / "models"
        / "random_forest_model.pkl"
    )


    # -----------------------------------------------------
    # OUTPUT PATHS
    # -----------------------------------------------------

    csv_output_path = (
        project_root
        / "outputs"
        / "metrics"
        / "feature_importance.csv"
    )

    graph_output_path = (
        project_root
        / "outputs"
        / "graphs"
        / "feature_importance.png"
    )


    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    print("\nLoading Random Forest model...")

    pipeline = joblib.load(model_path)

    print("Model loaded successfully!")


    # -----------------------------------------------------
    # GET PREPROCESSOR AND MODEL
    # -----------------------------------------------------

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]


    # -----------------------------------------------------
    # GET FEATURE NAMES
    # -----------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )


    # -----------------------------------------------------
    # GET FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance = (
        model.feature_importances_
    )


    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance
        }
    )


    # -----------------------------------------------------
    # SORT FEATURES
    # -----------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\nFEATURE IMPORTANCE RANKING")
    print("=" * 60)

    print(
        importance_df.to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # -----------------------------------------------------

    csv_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    graph_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    importance_df.to_csv(
        csv_output_path,
        index=False
    )

    print(
        f"\nFeature importance saved to:\n{csv_output_path}"
    )


    # -----------------------------------------------------
    # CREATE GRAPH
    # -----------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            by="Importance",
            ascending=True
        )
    )


    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.ylabel(
        "Features"
    )

    plt.title(
        "Top 15 Feature Importance - Random Forest"
    )

    plt.tight_layout()

    plt.savefig(
        graph_output_path,
        dpi=300
    )

    plt.close()


    print(
        f"\nFeature importance graph saved to:\n{graph_output_path}"
    )


    # -----------------------------------------------------
    # TOP 5 FEATURES
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP 5 MOST IMPORTANT FEATURES")
    print("=" * 60)

    print(
        importance_df
        .head(5)
        .to_string(
            index=False
        )
    )


    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":

    main()