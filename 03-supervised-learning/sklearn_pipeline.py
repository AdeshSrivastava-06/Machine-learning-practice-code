"""
Sklearn Pipeline Example

Pipeline chains preprocessing steps and a model into a single object,
so you call .fit()/.predict() once instead of manually applying each
transform to train AND test data separately (a common source of bugs
like accidentally fitting the scaler on test data / data leakage).

This example also uses ColumnTransformer to apply different
preprocessing to numeric vs categorical columns, and wraps the whole
thing inside GridSearchCV for hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Create a small synthetic dataset with numeric + categorical
#    columns and some missing values (to justify imputation)

rng = np.random.RandomState(42)
n = 300

df = pd.DataFrame({
    "age": rng.normal(40, 12, n),
    "income": rng.normal(50000, 15000, n),
    "city": rng.choice(["Mumbai", "Delhi", "Bangalore", "Pune"], n),
    "employment": rng.choice(["Salaried", "Self-Employed", "Unemployed"], n),
})

# introduce some missing values
df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
df.loc[rng.choice(n, 15, replace=False), "income"] = np.nan

# synthetic target: rough rule + noise
y = (
    (df["income"].fillna(df["income"].mean()) > 50000).astype(int) ^
    (df["employment"] == "Unemployed").astype(int)
)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)


# 2. Define preprocessing per column type

numeric_features = ["age", "income"]
categorical_features = ["city", "employment"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])


# 3. Full pipeline: preprocessing + model

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])


# 4. Hyperparameter tuning through the pipeline
#    (note the classifier__ prefix to target the model step)

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, None],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)


# 5. Evaluate on test set
#    (all preprocessing is applied automatically and consistently)

y_pred = grid_search.predict(X_test)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
