from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from fomo_detector.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MODEL_PATH,
    TIMESTAMP_COLUMN,
)


def load_training_data(data_path=DATA_PATH):
    data_path = Path(data_path)
    frame = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS) | {LABEL_COLUMN, TIMESTAMP_COLUMN}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Training data is missing required columns: {', '.join(missing_columns)}"
        )
    if frame.empty:
        raise ValueError("Training data contains no rows.")

    features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Training features contain missing or non-numeric values.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain infinite values.")

    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    if labels.isna().any() or not np.equal(labels, labels.astype(int)).all():
        raise ValueError("Training labels must be integers from 0 through 3.")
    labels = labels.astype(int)

    expected_labels = {0, 1, 2, 3}
    if set(labels.unique()) != expected_labels:
        raise ValueError("Training data must contain all labels: 0, 1, 2, and 3.")
    small_classes = labels.value_counts().loc[lambda counts: counts < 2]
    if not small_classes.empty:
        raise ValueError("Each class must have at least two rows for a stratified split.")

    return features, labels


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    features, labels = load_training_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("\nModel evaluation:\n")
    print(classification_report(y_test, predictions, zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=model_path.parent,
        prefix=f".{model_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Model saved to {model_path}")
    return model