# src/preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load HAR feature names
# -----------------------------
features = pd.read_csv(
    DATA_DIR / "features.txt", sep="\s+", header=None, names=["index", "feature"]
)
feature_names = features["feature"].values
feature_names_unique = [f"{name}_{i}" for i, name in enumerate(feature_names)]

# -----------------------------
# Load activity labels
# -----------------------------
activity_labels = pd.read_csv(
    DATA_DIR / "activity_labels.txt", sep="\s+", header=None, names=["id", "activity"]
)

# -----------------------------
# Load HAR train/test data
# -----------------------------
X_train = pd.read_csv(
    DATA_DIR / "train/X_train.txt", sep="\s+", header=None, names=feature_names_unique
)
y_train = pd.read_csv(
    DATA_DIR / "train/y_train.txt", sep="\s+", header=None, names=["Activity"]
)
y_train["Activity"] = y_train["Activity"].map(
    activity_labels.set_index("id")["activity"]
)

X_test = pd.read_csv(
    DATA_DIR / "test/X_test.txt", sep="\s+", header=None, names=feature_names_unique
)
y_test = pd.read_csv(
    DATA_DIR / "test/y_test.txt", sep="\s+", header=None, names=["Activity"]
)
y_test["Activity"] = y_test["Activity"].map(activity_labels.set_index("id")["activity"])

# -----------------------------
# Select mean() and std() features
# -----------------------------
selected_features = [f for f in feature_names_unique if "mean()" in f or "std()" in f]
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# Missing value check
print("Missing values in X_train:", X_train_selected.isnull().sum().sum())
print("Missing values in X_test:", X_test_selected.isnull().sum().sum())

# Save original selected feature names for later reference
pd.Series(selected_features).to_csv(
    DATA_DIR / "selected_features_list.csv", index=False
)


# -----------------------------
# Standardize features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# -----------------------------
# Save preprocessed data
# -----------------------------
np.save(DATA_DIR / "X_train_scaled.npy", X_train_scaled)
np.save(DATA_DIR / "X_test_scaled.npy", X_test_scaled)
y_train.to_csv(DATA_DIR / "y_train_labels.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test_labels.csv", index=False)
np.save(DATA_DIR / "selected_features.npy", selected_features)
joblib.dump(scaler, DATA_DIR / "scaler.pkl")

# INSERT SUMMARY PRINTS HERE
print("Selected features:", len(selected_features))
print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

print(
    "Preprocessing complete. Scaled features, labels, scaler, and selected features saved."
)
