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
X_train_selected = X_train[selected_features].copy()
X_test_selected = X_test[selected_features].copy()


# -----------------------------
# Compute additional features
# -----------------------------
def add_enhanced_features(df):
    df_enhanced = df.copy()

    # Identify acceleration columns (assuming naming pattern tBodyAcc-xxx)
    acc_cols = [c for c in df.columns if "Acc" in c]

    # Split into X, Y, Z axes
    X_cols = [c for c in acc_cols if c.endswith("_0")]
    Y_cols = [c for c in acc_cols if c.endswith("_1")]
    Z_cols = [c for c in acc_cols if c.endswith("_2")]

    # Vector magnitude
    df_enhanced["Acc_mag"] = np.sqrt(
        df[X_cols].pow(2).sum(axis=1)
        + df[Y_cols].pow(2).sum(axis=1)
        + df[Z_cols].pow(2).sum(axis=1)
    )

    # Jerk (difference)
    df_enhanced["Acc_jerk"] = df_enhanced["Acc_mag"].diff().fillna(0)

    # Rolling features: mean, std, min, max (window=5 samples)
    window = 5
    df_enhanced["Acc_mag_roll_mean"] = (
        df_enhanced["Acc_mag"].rolling(window).mean().fillna(0)
    )
    df_enhanced["Acc_mag_roll_std"] = (
        df_enhanced["Acc_mag"].rolling(window).std().fillna(0)
    )
    df_enhanced["Acc_mag_roll_min"] = (
        df_enhanced["Acc_mag"].rolling(window).min().fillna(0)
    )
    df_enhanced["Acc_mag_roll_max"] = (
        df_enhanced["Acc_mag"].rolling(window).max().fillna(0)
    )

    return df_enhanced


X_train_enhanced = add_enhanced_features(X_train_selected)
X_test_enhanced = add_enhanced_features(X_test_selected)

# -----------------------------
# Standardize features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)

# -----------------------------
# Save preprocessed data
# -----------------------------
np.save(DATA_DIR / "X_train_scaled.npy", X_train_scaled)
np.save(DATA_DIR / "X_test_scaled.npy", X_test_scaled)
y_train.to_csv(DATA_DIR / "y_train_labels.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test_labels.csv", index=False)
joblib.dump(scaler, DATA_DIR / "scaler.pkl")

# Save selected features including new enhanced ones
enhanced_features = list(X_train_enhanced.columns)
np.save(DATA_DIR / "selected_features.npy", enhanced_features)

print("Enhanced preprocessing complete. Features, scaled data, and labels saved.")
