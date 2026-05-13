# src/modeling.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, train_test_split
from xgboost import XGBClassifier
import joblib

# Define directories for data and output plots
DATA_DIR = Path("../data")
PLOT_DIR = Path("../plot")
PLOT_DIR.mkdir(exist_ok=True)

# Load preprocessed feature matrices and labels
X_train = np.load(DATA_DIR / "X_train_scaled.npy")
X_test = np.load(DATA_DIR / "X_test_scaled.npy")
y_train = pd.read_csv(DATA_DIR / "y_train_labels.csv")["Activity"]
y_test = pd.read_csv(DATA_DIR / "y_test_labels.csv")["Activity"]
selected_features = np.load(DATA_DIR / "selected_features.npy", allow_pickle=True)

# Encode labels for models requiring integer targets (e.g., XGBoost)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=600, class_weight="balanced", random_state=42
)
rf_model.fit(X_train, y_train_encoded)

# Train XGBoost classifier with validation split
X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
)

xgb_model.fit(
    X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False
)

# Weighted soft-voting ensemble using fixed weights
rf_weight = 0.6
xgb_weight = 0.4

rf_probs = rf_model.predict_proba(X_test)
xgb_probs = xgb_model.predict_proba(X_test)

ensemble_probs = (rf_probs * rf_weight) + (xgb_probs * xgb_weight)
ensemble_pred_encoded = np.argmax(ensemble_probs, axis=1)
ensemble_pred = le.inverse_transform(ensemble_pred_encoded)


# Evaluation function for all models
def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"\n{name} Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues" if "Random" in name else "Greens",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name.lower().replace(' ','_')}_confusion_matrix.png")
    plt.show()


# Evaluate all models
evaluate(y_test, le.inverse_transform(rf_model.predict(X_test)), "Random Forest")
evaluate(y_test, le.inverse_transform(xgb_model.predict(X_test)), "XGBoost")
evaluate(y_test, ensemble_pred, "Weighted Soft Voting Ensemble")

# Random Forest feature importance plot
importances = rf_model.feature_importances_
plt.figure(figsize=(14, 6))
plt.bar(range(len(importances)), importances)
plt.title("Random Forest Feature Importance (Enhanced Features)")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_DIR / "rf_feature_importance.png")
plt.show()

# Learning curve for Random Forest (used as ensemble proxy)
train_sizes, train_scores, test_scores = learning_curve(
    rf_model,
    X_train,
    y_train_encoded,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Accuracy")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Accuracy")
plt.title("Learning Curve (RF as Ensemble Proxy)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ensemble_learning_curve.png")
plt.show()

# Save trained models
joblib.dump(rf_model, DATA_DIR / "rf_model.pkl")
joblib.dump(xgb_model, DATA_DIR / "xgb_model.pkl")

print("Weighted Soft Voting ensemble modelling complete. Outputs saved.")
