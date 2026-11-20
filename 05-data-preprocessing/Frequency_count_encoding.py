# Frequency encoding for high cardinality categorical columns.
# Goal: encode columns with many categories (such as Ticket and Cabin in the
# Titanic data) without creating hundreds of one-hot columns.
# Frequency encoding replaces each category with how often it appears in the
# training data. Count encoding is the same idea but uses the raw count.
# Three setups are compared with 5-fold cross validation:
#   1. One-hot encoding of all categories
#   2. One-hot encoding where rare categories (fewer than 10 rows) are grouped
#   3. Frequency encoding
# The encoder is placed inside the pipeline, so the frequencies are learned
# only from the training part of each fold.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = "train.csv"

numeric_cols = ["Pclass", "SibSp", "Parch", "Fare"]
categorical_cols = ["Sex", "Embarked", "Ticket", "Cabin"]

df = pd.read_csv(CSV_PATH)
needed = numeric_cols + categorical_cols + ["Survived"]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Columns not found in {CSV_PATH}: {missing_cols}")

# Missing text values become their own category called Missing.
# Cabin is missing for most passengers, so this category carries information.
df[categorical_cols] = df[categorical_cols].fillna("Missing")
X = df[numeric_cols + categorical_cols]
y = df["Survived"]
print("Dataset:", X.shape[0], "rows")
print("Unique values per categorical column:")
print(X[categorical_cols].nunique().to_string())
print()

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    # Replaces each category with its share of rows in the training data.
    # Set normalize=False to get the raw count instead (count encoding).
    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = {c: X[c].value_counts(normalize=self.normalize) for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        # A category not seen during fit gets the value 0
        return np.column_stack([X[c].map(self.maps_[c]).fillna(0).to_numpy() for c in X.columns])

# Small demonstration of what the encoder produces
demo = FrequencyEncoder().fit(X[["Ticket"]])
sample = X[["Ticket"]].head(6).copy()
sample["frequency"] = demo.transform(sample).ravel().round(4)
sample["count"] = (sample["frequency"] * len(X)).round().astype(int)
print("Example for the Ticket column")
print(sample.to_string(index=False))
print()

def build(encoder):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", encoder, categorical_cols),
    ])

encoders = {
    "One-hot (all categories)": OneHotEncoder(handle_unknown="ignore"),
    "One-hot (rare grouped)": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
    "Frequency encoding": make_pipeline(FrequencyEncoder(), StandardScaler()),
}
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=3000),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"{'Encoding':26s} {'Features':>8s} {'Random Forest':>14s} {'Logistic Reg.':>14s}")
results = {}
for enc_name, encoder in encoders.items():
    # The feature count is taken from a fit on all data, only for display
    n_features = build(encoder).fit_transform(X).shape[1]
    row = []
    for model in models.values():
        pipe = make_pipeline(build(encoder), model)
        row.append(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean())
    results[enc_name] = (n_features, row)
    print(f"{enc_name:26s} {n_features:8d} {row[0]:14.4f} {row[1]:14.4f}")

# Summary
full_name, freq_name = "One-hot (all categories)", "Frequency encoding"
n_full, n_freq = results[full_name][0], results[freq_name][0]
acc_full, acc_freq = results[full_name][1][0], results[freq_name][1][0]
print()
print("Summary:")
print(f"One-hot encoding creates {n_full} features and frequency encoding creates {n_freq}.")
print(f"Random Forest accuracy: one-hot {acc_full:.3f}, frequency {acc_freq:.3f}.")
if acc_full - acc_freq > 0.01:
    print("Frequency encoding lost some accuracy here, because one-hot lets the model")
    print("recognise exact tickets and cabins, which frequency encoding cannot do.")
else:
    print("Frequency encoding kept almost the same accuracy with far fewer features.")
print("Frequency encoding is useful when a column has too many categories for one-hot,")
print("for example when the dataset is large or memory is limited.")
print("It gives the same value to different categories that appear equally often.")
