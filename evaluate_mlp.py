from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_preprocessing.datasets import (
    create_dataloader,
    create_sliding_windows,
)
from models.mlp import MLPForecaster


ROOT = Path(__file__).resolve().parent

RUN_NAME = "20261026_000611_276183"
RUN_DIRECTORY = ROOT / "results" / "mlp" / RUN_NAME

TARGET_COLUMN = "Active Power (kW)"


def prepare_test_data(checkpoint):
    train_data = pd.read_csv(
        ROOT / "data" / "processed" / "train.csv",
        parse_dates=["timestamp"],
    )

    validation_data = pd.read_csv(
        ROOT / "data" / "processed" / "validation.csv",
        parse_dates=["timestamp"],
    )

    test_data = pd.read_csv(
        ROOT / "data" / "processed" / "test.csv",
        parse_dates=["timestamp"],
    )

    # Łączymy zbiory, aby zachować historię wejściową
    # również dla pierwszych prognoz testowych.
    data = pd.concat(
        [train_data, validation_data, test_data],
        ignore_index=True,
    )

    timestamps = data["timestamp"]

    if (
        timestamps.isna().any()
        or not timestamps.diff().iloc[1:].eq(
            pd.Timedelta(minutes=15)
        ).all()
    ):
        raise ValueError(
            "Dane muszą tworzyć ciągłą siatkę pomiarów co 15 minut."
        )

    sequence_length = checkpoint["sequence_length"]
    forecast_horizon = checkpoint["forecast_horizon"]
    feature_columns = checkpoint["feature_columns"]

    X, y, target_times = create_sliding_windows(
        data=data,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    test_start = (
        test_data["timestamp"].iloc[0].to_datetime64()
    )

    validation_end = (
        validation_data["timestamp"].iloc[-1].to_datetime64()
    )

    # Moment, w którym dostępne jest całe okno wejściowe.
    origin_times = target_times - np.timedelta64(
        15 * forecast_horizon,
        "m",
    )

    test_mask = (
        (target_times >= test_start)
        & (origin_times >= validation_end)
    )

    if not test_mask.any():
        raise ValueError("Brak próbek testowych.")

    loader = create_dataloader(
        X=X[test_mask],
        y=y[test_mask],
        batch_size=checkpoint["configuration"]["batch_size"],
        shuffle=False,
    )

    return (
        loader,
        target_times[test_mask],
        origin_times[test_mask],
    )


def collect_predictions(model, loader, device):
    model.eval()

    targets = []
    predictions = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            batch_predictions = model(X_batch)

            targets.append(y_batch.cpu().numpy())

            predictions.append(
                batch_predictions.cpu().numpy()
            )

    targets = np.concatenate(targets).reshape(-1)
    predictions = np.concatenate(predictions).reshape(-1)

    if not np.isfinite(targets).all():
        raise ValueError("Targety zawierają NaN lub nieskończoność.")

    if not np.isfinite(predictions).all():
        raise ValueError("Prognozy zawierają NaN lub nieskończoność.")

    return targets, predictions


def inverse_target_scaling(values, scaler, feature_columns):
    target_index = feature_columns.index(TARGET_COLUMN)

    target_mean = scaler.mean_[target_index]
    target_scale = scaler.scale_[target_index]

    values = np.asarray(values, dtype=np.float64)

    return values * target_scale + target_mean


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Wczytujemy zwycięski checkpoint wybrany na walidacji.
    checkpoint = torch.load(
        RUN_DIRECTORY / "best_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    configuration = checkpoint["configuration"]
    feature_columns = checkpoint["feature_columns"]

    # Odtwarzamy architekturę i zapisane wagi.
    model = MLPForecaster(
        sequence_length=checkpoint["sequence_length"],
        number_of_features=len(feature_columns),
        hidden_sizes=configuration["hidden_sizes"],
        dropout=configuration["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Używamy scalera zachowanego dla tego eksperymentu.
    scaler = joblib.load(
        RUN_DIRECTORY / "preprocessing" / "scaler.joblib"
    )

    if list(scaler.feature_names_in_) != feature_columns:
        raise ValueError(
            "Kolejność cech scalera różni się od kolejności w modelu."
        )

    loader, target_times, origin_times = (
        prepare_test_data(checkpoint)
    )

    y_true_scaled, y_pred_scaled = collect_predictions(
        model,
        loader,
        device,
    )

    y_true_kw = inverse_target_scaling(
        y_true_scaled,
        scaler,
        feature_columns,
    )

    y_pred_kw = inverse_target_scaling(
        y_pred_scaled,
        scaler,
        feature_columns,
    )

    metrics = {
        "split": "test",
        "samples": len(y_true_kw),
        "best_epoch": checkpoint["epoch"],
        "forecast_horizon": checkpoint["forecast_horizon"],
        "mse_scaled": float(
            mean_squared_error(y_true_scaled, y_pred_scaled)
        ),
        "mae_kw": float(
            mean_absolute_error(y_true_kw, y_pred_kw)
        ),
        "rmse_kw": float(
            np.sqrt(mean_squared_error(y_true_kw, y_pred_kw))
        ),
        "r2": float(
            r2_score(y_true_kw, y_pred_kw)
        ),
    }

    predictions = pd.DataFrame(
        {
            "origin_timestamp": origin_times,
            "target_timestamp": target_times,
            "y_true_kw": y_true_kw,
            "y_pred_kw": y_pred_kw,
            "error_kw": y_pred_kw - y_true_kw,
        }
    )

    predictions.to_csv(
        RUN_DIRECTORY / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame([metrics]).to_csv(
        RUN_DIRECTORY / "test_metrics.csv",
        index=False,
    )

    print("Zbiór: test")
    print("Device:", device)
    print("Konfiguracja:", configuration)
    print("Najlepsza epoka treningu:", checkpoint["epoch"])
    print("Liczba próbek:", metrics["samples"])

    print(f"Test MSE po skalowaniu: {metrics['mse_scaled']:.8f}")
    print(f"Test MAE:  {metrics['mae_kw']:.4f} kW")
    print(f"Test RMSE: {metrics['rmse_kw']:.4f} kW")
    print(f"Test R²:   {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()