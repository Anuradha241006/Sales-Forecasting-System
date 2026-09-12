import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# SALES FORECASTING SYSTEM
# MODEL COMPARISON
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
    # METRICS PATH
    # -----------------------------------------------------

    metrics_folder = (
        project_root
        / "outputs"
        / "metrics"
    )


    # -----------------------------------------------------
    # OUTPUT GRAPH FOLDER
    # -----------------------------------------------------

    graphs_folder = (
        project_root
        / "outputs"
        / "graphs"
    )

    graphs_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # LOAD METRICS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)


    baseline_path = (
        metrics_folder
        / "baseline_metrics.csv"
    )

    linear_path = (
        metrics_folder
        / "linear_regression_metrics.csv"
    )

    random_forest_path = (
        metrics_folder
        / "random_forest_metrics.csv"
    )

    tuned_random_forest_path = (
        metrics_folder
        / "tuned_random_forest_metrics.csv"
    )


    print("\nLoading model metrics...")


    baseline_df = pd.read_csv(
        baseline_path
    )

    linear_df = pd.read_csv(
        linear_path
    )

    random_forest_df = pd.read_csv(
        random_forest_path
    )

    tuned_random_forest_df = pd.read_csv(
        tuned_random_forest_path
    )


    # -----------------------------------------------------
    # COMBINE METRICS
    # -----------------------------------------------------

    comparison_df = pd.concat(

        [

            baseline_df,

            linear_df,

            random_forest_df,

            tuned_random_forest_df

        ],

        ignore_index=True

    )


    # -----------------------------------------------------
    # HANDLE BASELINE R2
    # -----------------------------------------------------

    if "R2_Score" not in comparison_df.columns:

        comparison_df["R2_Score"] = None


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\nMODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    print(

        comparison_df.to_string(

            index=False

        )

    )


    # -----------------------------------------------------
    # RANK MODELS BY RMSE
    # -----------------------------------------------------

    ranked_df = (

        comparison_df

        .sort_values(

            by="RMSE",

            ascending=True

        )

        .reset_index(

            drop=True

        )

    )


    ranked_df.insert(

        0,

        "Rank",

        range(

            1,

            len(ranked_df) + 1

        )

    )


    print("\n" + "=" * 60)
    print("MODEL RANKING BASED ON RMSE")
    print("=" * 60)

    print(

        ranked_df.to_string(

            index=False

        )

    )


    # -----------------------------------------------------
    # SAVE COMPARISON
    # -----------------------------------------------------

    comparison_path = (

        metrics_folder

        / "model_comparison.csv"

    )


    comparison_df.to_csv(

        comparison_path,

        index=False

    )


    print(

        f"\nModel comparison saved to:\n{comparison_path}"

    )


    # -----------------------------------------------------
    # MAE GRAPH
    # -----------------------------------------------------

    plt.figure(

        figsize=(10, 6)

    )

    plt.bar(

        comparison_df["Model"],

        comparison_df["MAE"]

    )

    plt.xlabel(

        "Model"

    )

    plt.ylabel(

        "MAE"

    )

    plt.title(

        "Model Comparison - MAE"

    )

    plt.xticks(

        rotation=15

    )

    plt.tight_layout()

    mae_graph_path = (

        graphs_folder

        / "model_comparison_mae.png"

    )

    plt.savefig(

        mae_graph_path,

        dpi=300

    )

    plt.close()


    print(

        f"\nMAE graph saved:\n{mae_graph_path}"

    )


    # -----------------------------------------------------
    # RMSE GRAPH
    # -----------------------------------------------------

    plt.figure(

        figsize=(10, 6)

    )

    plt.bar(

        comparison_df["Model"],

        comparison_df["RMSE"]

    )

    plt.xlabel(

        "Model"

    )

    plt.ylabel(

        "RMSE"

    )

    plt.title(

        "Model Comparison - RMSE"

    )

    plt.xticks(

        rotation=15

    )

    plt.tight_layout()

    rmse_graph_path = (

        graphs_folder

        / "model_comparison_rmse.png"

    )

    plt.savefig(

        rmse_graph_path,

        dpi=300

    )

    plt.close()


    print(

        f"\nRMSE graph saved:\n{rmse_graph_path}"

    )


    # -----------------------------------------------------
    # R2 GRAPH
    # -----------------------------------------------------

    r2_data = (

        comparison_df

        .dropna(

            subset=[

                "R2_Score"

            ]

        )

    )


    if not r2_data.empty:

        plt.figure(

            figsize=(10, 6)

        )

        plt.bar(

            r2_data["Model"],

            r2_data["R2_Score"]

        )

        plt.xlabel(

            "Model"

        )

        plt.ylabel(

            "R² Score"

        )

        plt.title(

            "Model Comparison - R² Score"

        )

        plt.xticks(

            rotation=15

        )

        plt.tight_layout()

        r2_graph_path = (

            graphs_folder

            / "model_comparison_r2.png"

        )

        plt.savefig(

            r2_graph_path,

            dpi=300

        )

        plt.close()


        print(

            f"\nR² Score graph saved:\n{r2_graph_path}"

        )


    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    best_model = (

        ranked_df.iloc[0]

    )


    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(

        f"Model: {best_model['Model']}"

    )

    print(

        f"RMSE: {best_model['RMSE']:.4f}"

    )

    print(

        f"MAE: {best_model['MAE']:.4f}"

    )


    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":

    main()