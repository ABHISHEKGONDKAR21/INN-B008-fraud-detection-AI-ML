# ================================================================
#  FRAUD DETECTION AI - GOOGLE COLAB NOTEBOOK (FULL WORKING CODE)
# ================================================================

# ------------------------------------
# 1. Install dependencies
# ------------------------------------
!pip install scikit-learn pandas numpy joblib --quiet

# ------------------------------------
# 2. Upload your creditcard.csv.zip
# ------------------------------------
from google.colab import files
uploaded = files.upload()

import zipfile
import pandas as pd
import io

zip_name = list(uploaded.keys())[0]  # Use uploaded file

# Extract CSV from ZIP
with zipfile.ZipFile(zip_name, 'r') as z:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
    if not csv_files:
        raise Exception("No CSV file found in ZIP.")
    csv_name = csv_files[0]
    df = pd.read_csv(z.open(csv_name))

print("Dataset loaded:", df.shape)
print(df.head())

# ------------------------------------
# 3. Prepare dataset
# ------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Class'])
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Fraud count:", y_train.sum(), " / ", len(y_train))

# ------------------------------------
# 4. Build & train model (fast but accurate)
# ------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

# ------------------------------------
# 5. Evaluate
# ------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\n=== MODEL PERFORMANCE ===")
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------
# 6. Real-time prediction function
# ------------------------------------
import numpy as np

def predict_realtime(transaction_dict):
    """
    transaction_dict must contain the 30 feature fields:
    ['Time','V1','V2',...'V28','Amount']
    """

    df_in = pd.DataFrame([transaction_dict])

    missing = [c for c in X.columns if c not in df_in.columns]
    if missing:
        raise Exception("Missing fields: " + str(missing))

    proba = model.predict_proba(df_in)[0][1]
    pred = model.predict(df_in)[0]

    return {
        "prediction": int(pred),
        "fraud_probability": float(proba)
    }

print("\nReal-time prediction function is ready!")


# ------------------------------------
# 7. Example prediction (DUMMY values)
# ------------------------------------
example = {
    "Time": 10000,
    "V1": -1.1, "V2": 0.3, "V3": -0.9, "V4": 1.2, "V5": 0.2,
    "V6": -0.1, "V7": 0.2, "V8": -0.3, "V9": 0.0, "V10": -0.4,
    "V11": 0.1, "V12": -0.05, "V13": 0.02, "V14": -0.3, "V15": 0.1,
    "V16": -0.2, "V17": 0.0, "V18": 0.01, "V19": -0.01, "V20": 0.2,
    "V21": -0.1, "V22": 0.0, "V23": 0.03, "V24": -0.04, "V25": 0.0,
    "V26": 0.02, "V27": 0.01, "V28": -0.02,
    "Amount": 50
}

print("\nExample transaction → prediction:")
print(predict_realtime(example))
