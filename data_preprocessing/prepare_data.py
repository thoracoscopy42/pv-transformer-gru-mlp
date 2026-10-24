from pathlib import Path

from .data import (
    add_target_availability,
    apply_imputer,
    apply_scaler,
    chronological_split,
    fit_imputer,
    fit_scaler,
    get_feature_columns,
    load_csv,
    prepare_dataframe,
    print_missing_values,
    print_train_scaling_summary,
    replace_infinite_values,
    save_preprocessing_objects,
    save_splits,
    validate_imputation,
    validate_scaling,
    validate_time_index,
)


RAW_DATA_PATH = Path("data/raw/pv_data.csv")
PROCESSED_DATA_DIRECTORY = Path("data/processed")
PREPROCESSING_DIRECTORY = Path("results/preprocessing")

def main() -> None:
    # Loading data
    data = load_csv(RAW_DATA_PATH)
    data = prepare_dataframe(data)
    data = replace_infinite_values(data)
    data = add_target_availability(data)

    print(f"Loaded rows:    {len(data)}")
    print(f"Loaded columns: {len(data.columns)}")

    validate_time_index(data)

    # chronological split
    train_data, validation_data, test_data = (
        chronological_split(data)
    )

    print("\nChronological split:")
    print(f"Train:      {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test:       {len(test_data)}")

    print_missing_values(
        "train before imputation",
        train_data,
    )

    print_missing_values(
        "validation before imputation",
        validation_data,
    )

    print_missing_values(
        "test before imputation",
        test_data,
    )

    feature_columns = get_feature_columns(
        train_data
    )

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"- {column}")

    print(
        f"\nNumber of feature columns: "
        f"{len(feature_columns)}"
    )

    imputer = fit_imputer(
        train_data=train_data,
        feature_columns=feature_columns,
    )

    train_data = apply_imputer(
        data=train_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    validation_data = apply_imputer(
        data=validation_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    test_data = apply_imputer(
        data=test_data,
        feature_columns=feature_columns,
        imputer=imputer,
    )

    validate_imputation(
        name="train",
        data=train_data,
        feature_columns=feature_columns,
    )

    validate_imputation(
        name="validation",
        data=validation_data,
        feature_columns=feature_columns,
    )

    validate_imputation(
        name="test",
        data=test_data,
        feature_columns=feature_columns,
    )


    scaler = fit_scaler(
        train_data=train_data,
        feature_columns=feature_columns,
    )

    train_data = apply_scaler(
        data=train_data,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    validation_data = apply_scaler(
        data=validation_data,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    test_data = apply_scaler(
        data=test_data,
        feature_columns=feature_columns,
        scaler=scaler,
    )

    validate_scaling(
        name="train",
        data=train_data,
        feature_columns=feature_columns,
    )

    validate_scaling(
        name="validation",
        data=validation_data,
        feature_columns=feature_columns,
    )

    validate_scaling(
        name="test",
        data=test_data,
        feature_columns=feature_columns,
    )

    print_train_scaling_summary(
        train_data=train_data,
        feature_columns=feature_columns,
    )


    save_preprocessing_objects(
        imputer=imputer,
        scaler=scaler,
        feature_columns=feature_columns,
        output_directory=PREPROCESSING_DIRECTORY,
    )



    save_splits(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        output_directory=PROCESSED_DATA_DIRECTORY,
    )

    print(
        f"\nSaved processed splits in: "
        f"{PROCESSED_DATA_DIRECTORY}"
    )


if __name__ == "__main__":
    main()