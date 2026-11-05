from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIRECTORY = ROOT / "results"
PLOTS_DIRECTORY = ROOT / "plots"

MODEL_NAMES = ["mlp", "gru", "transformer"]

NUMBER_OF_DAYS = 7


def save_plot(figure, output_directory, filename):
    output_directory.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_directory / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Zapisano:", output_directory / f"{filename}.png")


def plot_predictions(
    predictions,
    model_name,
    run_name,
    output_directory,
    start,
    end,
):
    selected = predictions.loc[
        (predictions["target_timestamp"] >= start)
        & (predictions["target_timestamp"] < end)
    ]

    if selected.empty:
        print(f"Brak prognoz w wybranym okresie: {model_name}/{run_name}")
        return

    figure, axis = plt.subplots(figsize=(12, 4))

    axis.plot(
        selected["target_timestamp"],
        selected["y_true_kw"],
        label="Actual",
        color="black",
        linewidth=1.5,
    )

    axis.plot(
        selected["target_timestamp"],
        selected["y_pred_kw"],
        label="Predicted",
        color="tab:blue",
        linewidth=1.2,
        linestyle="--",
    )

    axis.set_title(f"{model_name.upper()} — test predictions")
    axis.set_xlabel("Time")
    axis.set_ylabel("Active power [kW]")

    axis.set_xlim(start, end)
    axis.grid(alpha=0.3)
    axis.legend()

    figure.autofmt_xdate()

    save_plot(
        figure,
        output_directory,
        f"{run_name}_predictions",
    )


def plot_learning_curves(
    run_directory,
    model_name,
    output_directory,
):
    results_path = run_directory / "results.csv"

    if not results_path.exists():
        print("Brak zestawienia prób:", results_path)
        return

    results = pd.read_csv(results_path)

    # Zwycięzcę wybieramy według walidacji, tak jak podczas treningu.
    best_index = results["validation_mse_scaled"].idxmin()
    best_result = results.loc[best_index]

    trial_number = int(best_result["trial"])
    best_epoch = int(best_result["best_epoch"])

    history_path = (
        run_directory
        / f"trial_{trial_number:03d}"
        / "history.csv"
    )

    if not history_path.exists():
        print("Brak historii uczenia:", history_path)
        return

    history = pd.read_csv(history_path)

    figure, axis = plt.subplots(figsize=(8, 4))

    axis.plot(
        history["epoch"],
        history["train_mse_scaled"],
        label="Training",
    )

    axis.plot(
        history["epoch"],
        history["validation_mse_scaled"],
        label="Validation",
    )

    axis.axvline(
        best_epoch,
        color="gray",
        linestyle=":",
        label=f"Best epoch: {best_epoch}",
    )

    axis.set_title(
        f"{model_name.upper()} — learning curves, trial {trial_number}"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE — standardized target")

    axis.grid(alpha=0.3)
    axis.legend()

    save_plot(
        figure,
        output_directory,
        f"{run_directory.name}_learning",
    )


def main():
    available_predictions = []

    for model_name in MODEL_NAMES:
        model_directory = RESULTS_DIRECTORY / model_name

        if not model_directory.exists():
            print(f"Brak wyników modelu: {model_name}")
            continue

        run_directories = sorted(
            path
            for path in model_directory.iterdir()
            if path.is_dir()
        )

        for run_directory in run_directories:
            output_directory = PLOTS_DIRECTORY / model_name

            plot_learning_curves(
                run_directory,
                model_name,
                output_directory,
            )

            predictions_path = (
                run_directory / "test_predictions.csv"
            )

            if not predictions_path.exists():
                print("Brak prognoz testowych:", predictions_path)
                continue

            predictions = pd.read_csv(
                predictions_path,
                parse_dates=["target_timestamp"],
            )

            predictions = predictions.sort_values(
                "target_timestamp"
            )

            if predictions.empty:
                print("Pusty plik prognoz:", predictions_path)
                continue

            available_predictions.append(
                {
                    "model_name": model_name,
                    "run_name": run_directory.name,
                    "data": predictions,
                }
            )

    if not available_predictions:
        print("Brak prognoz testowych do narysowania.")
        return

    # Ten sam okres na wszystkich wykresach prognoz.
    start = max(
        item["data"]["target_timestamp"].min()
        for item in available_predictions
    )

    last_common_timestamp = min(
        item["data"]["target_timestamp"].max()
        for item in available_predictions
    )

    end = min(
        start + pd.Timedelta(days=NUMBER_OF_DAYS),
        last_common_timestamp + pd.Timedelta(minutes=15),
    )

    if start >= end:
        raise ValueError(
            "Dostępne eksperymenty nie mają wspólnego okresu testowego."
        )

    print(f"Okres wykresów: od {start} do {end} — bez końca przedziału")

    for item in available_predictions:
        plot_predictions(
            predictions=item["data"],
            model_name=item["model_name"],
            run_name=item["run_name"],
            output_directory=PLOTS_DIRECTORY / item["model_name"],
            start=start,
            end=end,
        )


if __name__ == "__main__":
    main()