# src/modeling.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib
import json

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../data")
PLOT_DIR = Path("../plot")
PLOT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load preprocessed data
# -----------------------------
X_train = np.load(DATA_DIR / "X_train_scaled.npy")
X_test = np.load(DATA_DIR / "X_test_scaled.npy")
y_train = pd.read_csv(DATA_DIR / "y_train_labels.csv")["Activity"]
y_test = pd.read_csv(DATA_DIR / "y_test_labels.csv")["Activity"]
selected_features = np.load(DATA_DIR / "selected_features.npy", allow_pickle=True)

# -----------------------------
# Encode labels for XGBoost
# -----------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# -----------------------------
# Random Forest Hyperparameter Tuning
# -----------------------------
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("Best RF Params:", rf_grid.best_params_)
print("Best RF CV Score:", rf_grid.best_score_)

# -----------------------------
# XGBoost Hyperparameter Tuning
# -----------------------------
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

best_score = 0
best_params = None
for n in xgb_params["n_estimators"]:
    for d in xgb_params["max_depth"]:
        for lr in xgb_params["learning_rate"]:
            model = XGBClassifier(
                n_estimators=n,
                max_depth=d,
                learning_rate=lr,
                subsample=1.0,
                colsample_bytree=1.0,
                eval_metric="mlogloss",
                random_state=42,
                use_label_encoder=False,
            )
            model.fit(
                X_train_xgb,
                y_train_xgb,
                eval_set=[(X_val_xgb, y_val_xgb)],
                ## early_stopping_rounds=20,
                verbose=False,
            )
            score = model.score(X_val_xgb, y_val_xgb)
            if score > best_score:
                best_score = score
                best_params = (n, d, lr)
                xgb_model = model

print("Best XGB Params:", best_params)
print("Best XGB Score:", best_score)

# -----------------------------
# Save tuned parameters
# -----------------------------
with open(DATA_DIR / "tuned_params.json", "w") as f:
    json.dump({"rf": rf_grid.best_params_, "xgb": best_params}, f)


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, X_test, y_test, le=None, name="Model"):
    if le is not None:
        y_pred = le.inverse_transform(model.predict(X_test))
    else:
        y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy:", acc)
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues" if name == "Random Forest" else "Greens",
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name.lower().replace(' ','_')}_confusion_matrix.png")
    plt.show()


# -----------------------------
# Evaluate RF and XGB
# -----------------------------
evaluate_model(best_rf, X_test, y_test, name="Random Forest")
evaluate_model(xgb_model, X_test, y_test, le=le, name="XGBoost")

# -----------------------------
# Feature Importance (RF)
# -----------------------------
importances = best_rf.feature_importances_
plt.figure(figsize=(14, 6))
plt.bar(range(len(importances)), importances)
plt.title("Random Forest Feature Importance (HAR Dataset)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "rf_feature_importance.png")
plt.show()


# -----------------------------
# Learning Curves
# -----------------------------
def plot_learning_curve(model, X, y, title, filename):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label="Training Accuracy")
    plt.plot(train_sizes, test_mean, label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.show()


plot_learning_curve(
    best_rf, X_train, y_train, "Random Forest Learning Curve", "rf_learning_curve.png"
)
plot_learning_curve(
    xgb_model,
    X_train,
    y_train_encoded,
    "XGBoost Learning Curve",
    "xgb_learning_curve.png",
)

print("✅ Modeling complete. All plots and evaluations saved in 'plot/' folder.")
