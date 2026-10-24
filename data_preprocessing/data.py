from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.impute import SimpleImputer


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "Active Power (kW)"
TARGET_AVAILABLE_COLUMN = "target_available"


def load_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("Dataset is empty.")

    return data


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.columns = [column.strip() for column in data.columns]

    if "Unnamed: 0" in data.columns:
        data = data.rename(columns={"Unnamed: 0": TIMESTAMP_COLUMN})

    if TIMESTAMP_COLUMN not in data.columns:
        raise ValueError(
            f"Missing timestamp column: {TIMESTAMP_COLUMN}"
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN],
        errors="raise",
    )

    data = data.sort_values(TIMESTAMP_COLUMN)
    data = data.drop_duplicates(subset=TIMESTAMP_COLUMN)
    data = data.reset_index(drop=True)

    return data


def validate_time_index(
    data: pd.DataFrame,
    expected_frequency: str = "15min",
) -> None:
    timestamps = data[TIMESTAMP_COLUMN]

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not sorted chronologically."
        )

    expected_timestamps = pd.date_range(
        start=timestamps.iloc[0],
        end=timestamps.iloc[-1],
        freq=expected_frequency,
    )

    actual_timestamps = pd.DatetimeIndex(timestamps)

    missing_timestamps = expected_timestamps.difference(
        actual_timestamps
    )

    duplicated_timestamps = int(
        timestamps.duplicated().sum()
    )

    print("\nTime index:")
    print(f"First timestamp:       {timestamps.iloc[0]}")
    print(f"Last timestamp:        {timestamps.iloc[-1]}")
    print(f"Expected timestamps:   {len(expected_timestamps)}")
    print(f"Actual timestamps:     {len(actual_timestamps)}")
    print(f"Missing timestamps:    {len(missing_timestamps)}")
    print(f"Duplicated timestamps: {duplicated_timestamps}")

    if len(missing_timestamps) > 0:
        raise ValueError(
            "The time index contains missing timestamps."
        )

    if duplicated_timestamps > 0:
        raise ValueError(
            "The time index contains duplicated timestamps."
        )


def replace_infinite_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()

    numeric_columns = data.select_dtypes(
        include=[np.number]
    ).columns

    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


def add_target_availability(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()

    data[TARGET_AVAILABLE_COLUMN] = (
        data[TARGET_COLUMN].notna()
    )

    return data


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train and validation ratios must sum to less than 1."
        )

    number_of_rows = len(data)

    train_end = int(number_of_rows * train_ratio)

    validation_end = int(
        number_of_rows
        * (train_ratio + validation_ratio)
    )

    train_data = data.iloc[:train_end].copy()

    validation_data = data.iloc[
        train_end:validation_end
    ].copy()

    test_data = data.iloc[
        validation_end:
    ].copy()

    return train_data, validation_data, test_data


def print_missing_values(
    name: str,
    data: pd.DataFrame,
) -> None:
    missing_values = data.isna().sum()
    missing_values = missing_values[
        missing_values > 0
    ]

    print(f"\nMissing values — {name}:")

    if missing_values.empty:
        print("None")
    else:
        print(missing_values.to_string())


def get_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    excluded_columns = {
        TIMESTAMP_COLUMN,
        TARGET_AVAILABLE_COLUMN,
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns were found."
        )

    return feature_columns


def fit_imputer(
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> SimpleImputer:
    
    imputer = SimpleImputer(strategy="median")

    imputer.fit(
        train_data[feature_columns]
    )

    return imputer


def apply_imputer(
    data: pd.DataFrame,
    feature_columns: list[str],
    imputer: SimpleImputer,
) -> pd.DataFrame:
    data = data.copy()

    imputed_values = imputer.transform(
        data[feature_columns]
    )

    data[feature_columns] = imputed_values

    return data


def validate_imputation(
    name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    feature_values = data[
        feature_columns
    ].to_numpy(dtype=np.float64)

    missing_values = int(
        np.isnan(feature_values).sum()
    )

    infinite_values = int(
        np.isinf(feature_values).sum()
    )

    print(f"\nImputation check — {name}:")
    print(
        f"Missing feature values:  {missing_values}"
    )
    print(
        f"Infinite feature values: {infinite_values}"
    )

    if missing_values > 0:
        raise ValueError(
            f"{name} still contains missing feature values."
        )

    if infinite_values > 0:
        raise ValueError(
            f"{name} still contains infinite feature values."
        )


def save_splits(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_csv(
        output_directory / "train.csv",
        index=False,
    )

    validation_data.to_csv(
        output_directory / "validation.csv",
        index=False,
    )

    test_data.to_csv(
        output_directory / "test.csv",
        index=False,
    )

def fit_scaler(
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    scaler = StandardScaler()

    scaler.fit(
        train_data[feature_columns]
    )

    return scaler


def apply_scaler(
    data: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    data = data.copy()

    scaled_values = scaler.transform(
        data[feature_columns]
    )

    data[feature_columns] = scaled_values

    return data


def validate_scaling(
    name: str,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    feature_values = data[
        feature_columns
    ].to_numpy(dtype=np.float64)

    missing_values = int(
        np.isnan(feature_values).sum()
    )

    infinite_values = int(
        np.isinf(feature_values).sum()
    )

    print(f"\nScaling check — {name}:")
    print(f"Missing feature values:  {missing_values}")
    print(f"Infinite feature values: {infinite_values}")

    if missing_values > 0:
        raise ValueError(
            f"{name} contains missing values after scaling."
        )

    if infinite_values > 0:
        raise ValueError(
            f"{name} contains infinite values after scaling."
        )


def print_train_scaling_summary(
    train_data: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    means = train_data[feature_columns].mean()
    standard_deviations = train_data[feature_columns].std(ddof=0)

    print("\nTrain scaling summary:")
    print(
        f"Maximum absolute mean: "
        f"{means.abs().max():.8f}"
    )
    print(
        f"Maximum deviation from std=1: "
        f"{(standard_deviations - 1.0).abs().max():.8f}"
    )


def save_preprocessing_objects(
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: list[str],
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        imputer,
        output_directory / "imputer.joblib",
    )

    joblib.dump(
        scaler,
        output_directory / "scaler.joblib",
    )

    joblib.dump(
        feature_columns,
        output_directory / "feature_columns.joblib",
    )