
"""
utils.py
Shared utility functions for Flipkart Decision Intelligence project.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_FILE = DATA_DIR / "records.csv"


def load_dataset(file_path=DATA_FILE):
    """Load the project dataset from a CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def create_output_directory():
    """Create outputs directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_results(filename, content):
    """Save text results inside the outputs directory."""
    create_output_directory()

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(str(content))

    print(f"Saved: {output_path}")


def save_csv(data, filename):
    """Save a DataFrame or array as a CSV file."""
    create_output_directory()

    output_path = OUTPUT_DIR / filename

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    else:
        pd.DataFrame(data).to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def standardize_data(X):
    """
    Standardize numerical features using mean and standard deviation.

    Formula:
        Z = (X - mean) / standard_deviation

    Returns:
        standardized data, mean, standard deviation
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant columns
    std[std == 0] = 1.0

    X_scaled = (X - mean) / std

    return X_scaled, mean, std


def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate a binary classification confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


from utils import (
    load_dataset,
    save_results,
    save_csv,
    standardize_data,
    calculate_mse,
    calculate_rmse,
    calculate_accuracy
)

# Load dataset
df = load_dataset()

# Select five continuous predictors
features = [
    "customer_tenure_months",
    "complaint_count_90d",
    "discount_order_share",
    "avg_order_value_inr_hundreds",
    "loyalty_score"
]

X = df[features].values

# Standardize features
X_scaled, mean, std = standardize_data(X)

print("Standardized data shape:", X_scaled.shape)

# Save results
save_csv(df, "part3_segments.csv")
