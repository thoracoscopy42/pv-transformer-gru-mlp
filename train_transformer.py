from datetime import datetime
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from data_preprocessing.datasets import (
    create_dataloader,
    create_sliding_windows,
)
from model_configs.transformer import MODEL_SPACE, TRAINING_SPACE
from models.transformer import TransformerForecaster
from sample_hyperparameters import sample_hyperparameters


ROOT = Path(__file__).resolve().parent

SEQUENCE_LENGTH = 32
FORECAST_HORIZON = 1

NUMBER_OF_TRIALS = 50
MAX_EPOCHS = 300
PATIENCE = 15

SEARCH_SEED = 42
TRAINING_SEED = 42


def load_training_data():
    processed_directory = ROOT / "data" / "processed"

    train_data = pd.read_csv(
        processed_directory / "train.csv",
        parse_dates=["timestamp"],
    )

    validation_data = pd.read_csv(
        processed_directory / "validation.csv",
        parse_dates=["timestamp"],
    )

    feature_columns = joblib.load(
        ROOT / "results" / "preprocessing" / "feature_columns.joblib"
    )

    return train_data, validation_data, feature_columns


def prepare_windows(train_data, validation_data, feature_columns):
    data = pd.concat(
        [train_data, validation_data],
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

    X, y, target_times = create_sliding_windows(
        data=data,
        feature_columns=feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    validation_start = (
        validation_data["timestamp"].iloc[0].to_datetime64()
    )

    train_end = (
        train_data["timestamp"].iloc[-1].to_datetime64()
    )

    origin_times = target_times - np.timedelta64(
        15 * FORECAST_HORIZON,
        "m",
    )

    train_mask = target_times < validation_start

    validation_mask = (
        (target_times >= validation_start)
        & (origin_times >= train_end)
    )

    if not train_mask.any() or not validation_mask.any():
        raise ValueError(
            "Brak próbek treningowych lub walidacyjnych."
        )

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_validation": X[validation_mask],
        "y_validation": y[validation_mask],
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    number_of_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Nieprawidłowa wartość błędu treningowego."
            )

        loss.backward()
        optimizer.step()

        batch_size = len(X_batch)

        total_loss += loss.item() * batch_size
        number_of_samples += batch_size

    return total_loss / number_of_samples


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    number_of_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Nieprawidłowa wartość błędu walidacyjnego."
                )

            batch_size = len(X_batch)

            total_loss += loss.item() * batch_size
            number_of_samples += batch_size

    return total_loss / number_of_samples


def train_configuration(
    windows,
    feature_columns,
    configuration,
    device,
    output_directory,
):
    torch.manual_seed(TRAINING_SEED)

    model_parameters = {
    name: configuration[name]
    for name in MODEL_SPACE
    }

    model = TransformerForecaster(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=len(feature_columns),
        **model_parameters,
    ).to(device)

    train_loader = create_dataloader(
        X=windows["X_train"],
        y=windows["y_train"],
        batch_size=configuration["batch_size"],
        shuffle=True,
    )

    validation_loader = create_dataloader(
        X=windows["X_validation"],
        y=windows["y_validation"],
        batch_size=configuration["batch_size"],
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )

    output_directory.mkdir(parents=True, exist_ok=False)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "configuration": configuration,
                    "feature_columns": feature_columns,
                    "sequence_length": SEQUENCE_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON,
                    "training_seed": TRAINING_SEED,
                    "epoch": epoch,
                    "validation_mse_scaled": validation_loss,
                },
                output_directory / "best_model.pt",
            )

        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": train_loss,
                "validation_mse_scaled": validation_loss,
            }
        )

        pd.DataFrame(history).to_csv(
            output_directory / "history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:03d}: "
            f"train MSE={train_loss:.6f}, "
            f"val MSE={validation_loss:.6f}"
        )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    return {
        "best_epoch": best_epoch,
        "validation_mse_scaled": best_validation_loss,
        "epochs_completed": len(history),
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_data, validation_data, feature_columns = (
        load_training_data()
    )

    windows = prepare_windows(
        train_data,
        validation_data,
        feature_columns,
    )

    search_space = {
        **MODEL_SPACE,
        **TRAINING_SPACE,
    }

    configurations = sample_hyperparameters(
        search_space=search_space,
        number_of_trials=NUMBER_OF_TRIALS,
        seed=SEARCH_SEED,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = ROOT / "results" / "transformer" / run_id

    output_directory.mkdir(parents=True, exist_ok=False)

    preprocessing_directory = output_directory / "preprocessing"
    preprocessing_directory.mkdir()

    for name in (
        "scaler.joblib",
        "imputer.joblib",
        "feature_columns.joblib",
    ):
        shutil.copy2(
            ROOT / "results" / "preprocessing" / name,
            preprocessing_directory / name,
        )

    experiment_settings = {
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "number_of_trials": NUMBER_OF_TRIALS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "search_seed": SEARCH_SEED,
        "training_seed": TRAINING_SEED,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "configurations": configurations,
    }

    with (output_directory / "settings.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(experiment_settings, file, indent=2)

    print("Device:", device)
    print("Train:", windows["X_train"].shape)
    print("Validation:", windows["X_validation"].shape)
    print("Results:", output_directory)

    results = []

    for trial_number, configuration in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"\nKonfiguracja {trial_number}/{NUMBER_OF_TRIALS}: "
            f"{configuration}"
        )

        trial_directory = (
            output_directory / f"trial_{trial_number:03d}"
        )

        result = train_configuration(
            windows=windows,
            feature_columns=feature_columns,
            configuration=configuration,
            device=device,
            output_directory=trial_directory,
        )

        results.append(
            {
                "trial": trial_number,
                **configuration,
                **result,
            }
        )

        pd.DataFrame(results).to_csv(
            output_directory / "results.csv",
            index=False,
        )

    best_result = min(
        results,
        key=lambda result: result["validation_mse_scaled"],
    )

    best_trial_directory = (
        output_directory / f"trial_{best_result['trial']:03d}"
    )

    shutil.copy2(
        best_trial_directory / "best_model.pt",
        output_directory / "best_model.pt",
    )

    print("\nNajlepszy wynik:")
    print(best_result)


if __name__ == "__main__":
    main()