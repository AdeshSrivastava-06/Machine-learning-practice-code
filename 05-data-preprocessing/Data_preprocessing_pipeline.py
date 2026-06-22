
"""If we will not use the pipelines then the series of steps that we have done to make the model then that same series of steps we have to make after deploying the model so pipelines helps there"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')

print(df.head())

"""will follow the following steps 1]Filling the missing values in the age and embarked cols 2]One-hot-encoding on sex,embarked 3]scaling the output 4]feature selection(best 5 cols) 5]training the model using a decisionTree"""

df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'])#these cols are not useful

"""step1-> train-test-split"""
X_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)

"""imputation ransformer normalization ohe"""
trf1 = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ]), ['Age','Fare']),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Sex','Embarked'])
],remainder='passthrough')

"""here we have not written the cols name instead we have writeen the index values because the output of the transformer is not a dataframe but a numpy array
and in the one-hot encoder we can do it with both the cols because everything is benn stored and no new array is created"""


trf4 = SelectKBest(score_func=chi2,k=5)

"""train the model"""

trf5 = DecisionTreeClassifier()

"""create a pipelines"""

pipe = Pipeline([
    ('trf1',trf1),
    ('trf4',trf4),
    ('trf5',trf5)
])

"""pipeline requires naming of steps,make_pipeline does not same applies to columnTransformer and make_col_t i.e
pipe = make_pipeline(trf1,trf4,trf5)"""

#train
pipe.fit(X_train,y_train)
"""if we are not training the model then we call fit_transform but here we are goning to train the data so only fit"""

"""display pipeline"""
from sklearn import set_config
set_config(display='diagram')

print(pipe.named_steps)

y_pred = pipe.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuacy-score-->',accuracy_score(y_test,y_pred))

"""cross validation using pipelines"""

from sklearn.model_selection import cross_val_score

print("cross-val-score->",cross_val_score(pipe,X_train,y_train,cv=5,scoring='accuracy').mean())

"""exporting the pipeline"""

import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))

pipe = pickle.load(open('pipe.pkl','rb'))

"""assume user imput"""

test_input = pd.DataFrame(
    [[3, 'male', 22, 1, 0, 7.25, 'S']],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
)

print(pipe.predict(test_input))
