# Step 0: Install required packages
!pip install xgboost imbalanced-learn shap

# Step 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv("/content/HI774diabetes.csv")
df.head()
df.info()

# Step 3: Separate features & target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Replace impossible zeros with NaN
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in zero_cols:
    X[c] = X[c].replace(0, np.nan)
	
# Step 4: Feature Engineering
def add_features(X_df):
    X_new = X_df.copy()
    eps = 1e-6

    if "BMI" in X_new.columns and "Age" in X_new.columns:
        X_new["BMI_Age"] = X_new["BMI"] * X_new["Age"]

    if "Glucose" in X_new.columns and "Insulin" in X_new.columns:
        X_new["Gluc_Ins_Ratio"] = X_new["Glucose"] / (X_new["Insulin"] + eps)

    return X_new

X = add_features(X)
X.head()

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lr_model = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=2000))
])

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print("=== Logistic Regression ===")
print(classification_report(y_test, lr_pred))
print("ROC AUC:", roc_auc_score(y_test, lr_prob))	

rf_model = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42))
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("=== Random Forest ===")
print(classification_report(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob))

xgb_model = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
])

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("=== XGBoost ===")
print(classification_report(y_test, xgb_pred))
print("ROC AUC:", roc_auc_score(y_test, xgb_prob))

shap.initjs()

# Extract trained classifier
clf = rf_model.named_steps["clf"]

# Prepare test data for SHAP: apply imputer + scaler only (do NOT include SMOTE)
X_test_proc = rf_model.named_steps["imputer"].transform(X_test)
X_test_proc = rf_model.named_steps["scaler"].transform(X_test_proc)

# Feature names (including engineered features)
feature_names = X_test.columns.tolist()

# SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_proc)

# Plot summary
shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names)

ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.show()

