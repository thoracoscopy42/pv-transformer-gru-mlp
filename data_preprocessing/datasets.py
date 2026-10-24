from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import (
    TARGET_AVAILABLE_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)


def create_sliding_windows(
    data: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than zero.")

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be greater than zero.")

    required_columns = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        TARGET_AVAILABLE_COLUMN,
        *feature_columns,
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    feature_values = data[feature_columns].to_numpy(
        dtype=np.float32
    )

    target_values = data[TARGET_COLUMN].to_numpy(
        dtype=np.float32
    )

    target_available = data[
        TARGET_AVAILABLE_COLUMN
    ].to_numpy(dtype=bool)

    timestamps = data[
        TIMESTAMP_COLUMN
    ].to_numpy()

    windows = []
    targets = []
    target_timestamps = []

    first_target_index = (
        sequence_length + forecast_horizon - 1
    )

    for target_index in range(
        first_target_index,
        len(data),
    ):
        if not target_available[target_index]:
            continue

        input_end = (
            target_index - forecast_horizon + 1
        )

        input_start = (
            input_end - sequence_length
        )

        window = feature_values[
            input_start:input_end
        ]

        target = target_values[target_index]

        windows.append(window)
        targets.append([target])
        target_timestamps.append(
            timestamps[target_index]
        )

    X = np.asarray(
        windows,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    target_times = np.asarray(
        target_timestamps,
    )

    validate_window_shapes(
        X=X,
        y=y,
        sequence_length=sequence_length,
        number_of_features=len(feature_columns),
    )

    return X, y, target_times


def validate_window_shapes(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
    number_of_features: int,
) -> None:
    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have 3 dimensions, received {X.shape}."
        )

    if y.ndim != 2:
        raise ValueError(
            f"Expected y to have 2 dimensions, received {y.shape}."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y contain different numbers of samples."
        )

    if X.shape[1] != sequence_length:
        raise ValueError(
            "Incorrect sequence length in X."
        )

    if X.shape[2] != number_of_features:
        raise ValueError(
            "Incorrect number of features in X."
        )

    if y.shape[1] != 1:
        raise ValueError(
            "Expected one regression target per sample."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        validate_dataset_arrays(
            X=X,
            y=y,
        )

        self.X = torch.from_numpy(
            np.ascontiguousarray(
                X,
                dtype=np.float32,
            )
        )

        self.y = torch.from_numpy(
            np.ascontiguousarray(
                y,
                dtype=np.float32,
            )
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def validate_dataset_arrays(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape "
            f"[samples, sequence_length, features], "
            f"received {X.shape}."
        )

    if y.ndim != 2:
        raise ValueError(
            f"Expected y with shape [samples, outputs], "
            f"received {y.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "y contains NaN or infinite values."
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    dataset = TimeSeriesDataset(
        X=X,
        y=y,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )