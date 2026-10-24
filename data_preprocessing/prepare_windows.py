from pathlib import Path

import joblib
import pandas as pd

from .datasets import create_sliding_windows


DATA_DIRECTORY = Path("data/processed")
PREPROCESSING_DIRECTORY = Path("results/preprocessing")

SEQUENCE_LENGTH = 32
FORECAST_HORIZON = 1


def main():
    # Wczytujemy przygotowane i przeskalowane dane.
    train = pd.read_csv(
        DATA_DIRECTORY / "train.csv",
        parse_dates=["timestamp"],
    )
    validation = pd.read_csv(
        DATA_DIRECTORY / "validation.csv",
        parse_dates=["timestamp"],
    )
    test = pd.read_csv(
        DATA_DIRECTORY / "test.csv",
        parse_dates=["timestamp"],
    )

    feature_columns = joblib.load(
        PREPROCESSING_DIRECTORY / "feature_columns.joblib"
    )

    data = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    X, y, target_timestamps = create_sliding_windows(
        data=data,
        feature_columns=feature_columns,
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    
    validation_start = validation["timestamp"].iloc[0]
    test_start = test["timestamp"].iloc[0]

    train_mask = target_timestamps < validation_start
    validation_mask = (
        (target_timestamps >= validation_start)
        & (target_timestamps < test_start)
    )
    test_mask = target_timestamps >= test_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[validation_mask], y[validation_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("Rozmiary tablic:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Oglądamy pierwszą próbkę treningową.
    first_target_time = pd.Timestamp(
        target_timestamps[train_mask][0]
    )
    input_end_time = first_target_time - pd.Timedelta(
        minutes=15 * FORECAST_HORIZON
    )
    input_start_time = input_end_time - pd.Timedelta(
        minutes=15 * (SEQUENCE_LENGTH - 1)
    )

    print("\nPierwsza próbka treningowa:")
    print(f"Wejście od: {input_start_time}")
    print(f"Wejście do: {input_end_time}")
    print(f"Target na:  {first_target_time}")

    print("\nX — wartości po skalowaniu:")
    print(pd.DataFrame(X_train[0], columns=feature_columns))

    print(f"\ny — moc po skalowaniu: {y_train[0, 0]:.6f}")


if __name__ == "__main__":
    main()