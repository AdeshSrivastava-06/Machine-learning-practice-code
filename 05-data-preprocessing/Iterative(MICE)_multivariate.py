"""Multivariate-
       KNN imputer
       Iterative imputer(MICE->Multivaraite imputation by chained equations the most imp assumption is the data should be MAR(missing at random),slow,memory probelm but accurate,only applied on input cols)"""
       
'''the following steps are being follwed
step[1]->reaplace the NaN values with the mean of respective cols
step[2]->Remove all cols1 missing values(now that 1col will become the target and the other 2 will become the input to predict it)
step[3]->predict the missing values of col1 using other col
step[4]->simillary for col2,col3
step[5]->we will consider the differencr of iteration0{mean} - iteration2{step3-4} and make a new datadframe of the differencr and will continue until the both the itrations (i.e performing step2,3,4 once more with the new values) the difference is 0'''       


"""
MULTIVARIATE IMPUTATION
ITERATIVE IMPUTER (MICE)

-> assumes MAR (Missing At Random)
-> uses relationships between features
-> slow and memory heavy
-> gives more accurate estimates
-> applied ONLY on input columns
-> commonly used in industry when data is correlated
"""

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Load Titanic Dataset

df = sns.load_dataset('titanic')

df = df[['age','fare','pclass','survived','sex']]
print(df.isnull().sum())

# Train Test Split

x = df.drop(columns='survived')
y = df['survived']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

# Iterative Imputer (MICE)

preprocessor = ColumnTransformer(transformers=[
       ('ohe',OneHotEncoder(),['sex']),
       ('mice',IterativeImputer(max_iter=10,random_state=0),['age'])
],remainder='passthrough')

pip = Pipeline([
       ('preprocessor',preprocessor),
       ('method',LogisticRegression(max_iter=1000))
])

pip.fit(x_train,y_train)

y_pred = pip.predict(x_test)

print(accuracy_score(y_test,y_pred))