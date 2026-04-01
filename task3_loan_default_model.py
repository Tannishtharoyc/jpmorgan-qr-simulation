"""
============================================================
  Loan Default Prediction & Expected Loss Calculator
  Retail Banking — Risk Team Prototype
============================================================

  EL  =  PD  x  LGD  x  EAD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

# =========================
# CONSTANTS
# =========================
RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE

# =========================
# LOAD DATA
# =========================
DATA_PATH = "data/loan_data.csv"
df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)} | Default Rate: {df['default'].mean():.2%}")

# =========================
# FEATURE ENGINEERING
# =========================
def engineer(df):
    d = df.copy()
    d['debt_to_income'] = d['total_debt_outstanding'] / (d['income'] + 1)
    d['loan_to_income'] = d['loan_amt_outstanding'] / (d['income'] + 1)
    d['loan_to_debt'] = d['loan_amt_outstanding'] / (d['total_debt_outstanding'] + 1)
    return d

df = engineer(df)

TARGET = "default"
FEATURES = [c for c in df.columns if c not in ["customer_id", TARGET]]

X = df[FEATURES]
y = df[TARGET]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# MODELS
# =========================
models = {
    "Logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=6)
}

# =========================
# TRAIN + EVALUATE
# =========================
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    results[name] = {
        "model": model,
        "auc": auc,
        "ap": ap,
        "brier": brier
    }

    print(f"{name} → AUC: {auc:.4f}, AP: {ap:.4f}")

# =========================
# BEST MODEL
# =========================
best_name = max(results, key=lambda x: results[x]["auc"])
best_model = results[best_name]["model"]

print(f"\nBest Model: {best_name}")

# =========================
# EXPECTED LOSS
# =========================
probs = best_model.predict_proba(X_test)[:, 1]
ead = X_test["loan_amt_outstanding"]

expected_loss = probs * LGD * ead

print(f"\nAverage Expected Loss: {expected_loss.mean():.2f}")

# =========================
# SIMPLE PLOT
# =========================
plt.hist(expected_loss, bins=30)
plt.title("Expected Loss Distribution")
plt.savefig("fig_expected_loss.png")
print("Plot saved: fig_expected_loss.png")