# 1. Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Read the CSV file
df = pd.read_csv('secondary_data.csv', delimiter=';')

# 3. Separate features (X) and target variable (y)
X = df.drop('class', axis=1)
y = df['class']

# 4. Check for missing values
missing_values = X.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing values (initial check):")
print(missing_values)

# 5. Fill missing values with the most frequent value in each column
for column in missing_values.index:
    most_frequent = X[column].mode()[0]
    X[column] = X[column].fillna(most_frequent)

# 6. Check if there are any missing values left
print("Total number of missing values after filling:")
print(X.isnull().sum().sum())

# 7. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Print the shapes of the training and test sets
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Test target shape:", y_test.shape)

# 9. Identify categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# 10. Create a preprocessor with One-Hot Encoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 11. Transform the training and test sets
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# 12. Print the shapes of the encoded data
print("Encoded X_train shape:", X_train_encoded.shape)
print("Encoded X_test shape:", X_test_encoded.shape)

# 13. Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC()
}

# 14. Evaluate each model using 5-Fold Cross Validation
print("\nModel Performance Results (5-Fold Cross Validation):\n")
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_encoded, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} - Mean Accuracy: {scores.mean():.4f}")

# 15. Hyperparameter optimization for Logistic Regression using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

log_reg = LogisticRegression(max_iter=1000)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\nStarting Grid Search for Logistic Regression...\n")
grid_search.fit(X_train_encoded, y_train)
print("\nGrid Search Completed.\n")
print("Best hyperparameters:", grid_search.best_params_)
print("Best validation accuracy:", grid_search.best_score_)

# 16. Train the final model using the best parameters
final_model = LogisticRegression(
    C=grid_search.best_params_['C'],
    solver=grid_search.best_params_['solver'],
    max_iter=1000
)

final_model.fit(X_train_encoded, y_train)

# 17. Make predictions on the test set
y_pred = final_model.predict(X_test_encoded)

# 18. Evaluate the test results
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Set Accuracy Score:", round(accuracy, 4))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['edible', 'poisonous']))
