import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = sns.load_dataset('titanic')

df = df[['age','fare','pclass','sex','embarked','survived']]

X = df.drop(columns='survived')
y = df['survived']

num_cols = ['age','fare','pclass']
cat_cols = ['sex','embarked']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

"""METHOD 1 — CCA (Complete Case Analysis)"""

cca_df = df.dropna()

X1 = cca_df.drop(columns='survived')
y1 = cca_df['survived']

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=2
)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols)
], remainder='passthrough')

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train1, y_train1)
print("CCA Accuracy:", accuracy_score(y_test1, pipe.predict(X_test1)))


"""METHOD 2 — Simple Imputer (Median + Mode)"""

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first'))
    ]), cat_cols)
],remainder='passthrough')

pipe = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
print("Median/Mode Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))


"""METHOD 3 — KNN Imputer"""

preprocessor = ColumnTransformer([
    ('num', KNNImputer(n_neighbors=5), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first'))
    ]), cat_cols)
],remainder='passthrough')

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
print("KNN Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))


"""METHOD 4 — MICE (Iterative Imputer)"""


preprocessor = ColumnTransformer([
    ('num', IterativeImputer(max_iter=10, random_state=0), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first'))
    ]), cat_cols)
],remainder='passthrough')

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
print("MICE Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))
