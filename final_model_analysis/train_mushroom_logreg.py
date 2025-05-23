# ============================================================
# Toxic Mushroom Classification – Logistic-Regression Pipeline
# ============================================================

#
# Workflow
# --------
# 1. Load data and handle missing values
# 2. Stratified train-test split
# 3. One-Hot encoding for categorical features
# 4. Compare several baseline models via 5-fold CV
# 5. Grid-search hyper-parameter tuning for Logistic Regression
# 6. Evaluate the best model on the held-out test set
# 7. Save the fitted preprocessing pipeline + model with joblib
# ============================================================

# 1 ── Imports
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 2 ── Load the dataset
DATA_PATH = "secondary_data.csv"          # CSV in the same folder
df = pd.read_csv(DATA_PATH, delimiter=";")

X = df.drop("class", axis=1)
y = df["class"]

# 3 ── Handle missing values (mode imputation)
na_counts = X.isnull().sum()
cols_with_na = na_counts[na_counts > 0].index

print("Columns with missing values:\n", na_counts[na_counts > 0])

for col in cols_with_na:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\nTotal missing values after imputation:",
      int(X.isnull().sum().sum()))

# 4 ── Stratified train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTrain set shape: {X_train.shape}   "
      f"Test set shape: {X_test.shape}")

# 5 ── Pre-processing (One-Hot Encode categorical columns)
categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough",
)

# 6 ── Baseline model comparison (5-fold CV)
models = {
    "LogReg": LogisticRegression(max_iter=1_000),
    "DecTree": DecisionTreeClassifier(),
    "RandForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
}

print("\n=== 5-Fold Cross-Validation Results ===")
for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="accuracy"
    )
    print(f"{name:<10}:  {cv_scores.mean():.4f}")

# 7 ── Hyper-parameter tuning for Logistic Regression
logreg_pipe = Pipeline(
    steps=[("prep", preprocessor), ("clf", LogisticRegression(max_iter=1_000))]
)

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__solver": ["lbfgs", "liblinear"],
}

print("\n>>> Starting grid search for Logistic Regression …")
grid = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)
grid.fit(X_train, y_train)
print(">>> Grid search completed.\n")

print("Best hyper-parameters :", grid.best_params_)
print("Best CV accuracy      :", grid.best_score_)

# 8 ── Best estimator (includes preprocessor + model)
final_model = grid.best_estimator_

# 9 ── Evaluate on the test set
y_pred = final_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(
    y_test, y_pred, target_names=["edible", "poisonous"]
)

print("\n=== Test-Set Results ===")
print(f"Accuracy : {test_acc:.4f}\n")
print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", class_rep)

# 10 ── Save pipeline (preprocess + model)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pipeline_logreg.joblib"

joblib.dump(final_model, MODEL_PATH)
print(f"\n✅ Pipeline saved to  {MODEL_PATH.resolve()}")
