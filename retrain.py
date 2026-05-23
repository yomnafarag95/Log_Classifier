"""
retrain.py — Firewall Log Classifier
Trains the Random Forest model on Dataset__log2_.csv and saves:
  - model.joblib
  - scaler.joblib
  - label_encoder.joblib

Run:  python retrain.py
Requires: pip install scikit-learn imbalanced-learn pandas joblib
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "Dataset__log2_.csv"

FEATURE_COLS = [
    "Source Port",
    "Destination Port",
    "NAT Source Port",
    "NAT Destination Port",
    "Bytes",
    "Bytes Sent",
    "Bytes Received",
    "Packets",
    "Elapsed Time (sec)",
    "pkts_sent",
    "pkts_received",
]

TARGET_COL = "Action"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load ───────────────────────────────────────────────────────────────────
print(f"Loading {DATASET_PATH} ...")
df = pd.read_csv(DATASET_PATH)
print(f"  Raw rows: {len(df):,}   Columns: {list(df.columns)}")

# ── 2. Verify columns exist ───────────────────────────────────────────────────
missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    print("\nERROR — these columns are missing from the CSV:")
    for m in missing:
        print(f"  '{m}'")
    print("\nAvailable columns:", list(df.columns))
    raise SystemExit(1)

# ── 3. Preprocessing ──────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
print(f"  After dedup: {len(df):,} rows")
# IQR filtering skipped — it eliminates the rare reset-both class entirely.
# SMOTE handles class imbalance instead.

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"  Class distribution:\n{pd.Series(y).value_counts().to_string()}")

# ── 4. Encode labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Classes: {list(le.classes_)}")

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.30, stratify=y_enc, random_state=42
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── 6. SMOTE ──────────────────────────────────────────────────────────────────
print("Applying SMOTE ...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE train size: {len(X_train_res):,}")

# ── 7. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ── 8. Train tuned Random Forest ──────────────────────────────────────────────
print("Training Random Forest (n_estimators=200, max_depth=20) ...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train_res)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test_sc)
acc  = accuracy_score(y_test, y_pred)
mf1  = f1_score(y_test, y_pred, average="macro")
print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"Macro F1      : {mf1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 10. Save artifacts ────────────────────────────────────────────────────────
model_path   = os.path.join(OUTPUT_DIR, "model.joblib")
scaler_path  = os.path.join(OUTPUT_DIR, "scaler.joblib")
le_path      = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

joblib.dump(rf,     model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le,     le_path)

print(f"\nSaved:")
print(f"  {model_path}")
print(f"  {scaler_path}")
print(f"  {le_path}")
print("\nDone! Upload these 3 files to your repo alongside app.py.")